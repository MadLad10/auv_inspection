"""Deep segmenter: per-pixel classes from a MobileNetV2 encoder pretrained on ImageNet.

The encoder is torchvision's MobileNetV2 with its ImageNet weights (IMAGENET1K_V2), read from a file
whose SHA-256 must equal segmenter.backbone_sha256: nothing is downloaded here. A light decoder
(segmenter.decoder_channels) merges the encoder's stride-32, -16 and -8 features, as a feature
pyramid does, into a score per class (background and perception.classes) at stride 8, upsampled to
the image.

Training (`auvinspect train-segmenter`) takes the train split of one or more perception datasets:
simulated frames (dataset.py), LIACi's real images (liaci.py), or both. Images are resized to
segmenter.input_width and padded to input_height (the loss ignores the padding), with horizontal
flips and brightness and contrast jitter. The loss is cross-entropy, each class weighted by its
pixel frequency^-class_weight_power. The model kept is the epoch with the best mean AP over the
datasets' val splits; the test splits are never seen. AdamW with cosine decay, the pretrained
encoder at encoder_lr_factor of the learning rate. The decoder's initial weights, the batch order and
the augmentation are drawn from the seed; the run is repeatable up to the floating-point
nondeterminism of multi-threaded CPU kernels.

Detection (`auvinspect segment`): each connected region of a class with at least
segmenter.min_region_px pixels (at the dataset's resolution) becomes a box, scored by the class's
mean probability over the region, for the perception referee. Neighbouring defects whose regions
touch become one box, and a defect split into pieces gives several, so the box score understates a
segmenter. Each frame's pixels are also counted against the dataset's masks, for a per-class IoU.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time

import numpy as np
from PIL import Image
from scipy import ndimage

from . import perception, registry
from .rng import stream

IGNORE = 255


def _torch():
    try:
        import torch
        import torchvision  # noqa: F401
    except ImportError as e:
        raise ValueError('the deep segmenter needs PyTorch and torchvision: install the `learn` extra') from e
    return torch


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build_net(n_out: int, channels: int, backbone: str | None = None):
    """The network; with backbone (the torchvision MobileNetV2 weights file), its encoder is pretrained."""
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional
    from torchvision.models import mobilenet_v2

    def project(c_in, k):
        return nn.Sequential(nn.Conv2d(c_in, channels, k, padding=k // 2, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            m = mobilenet_v2()
            if backbone:
                m.load_state_dict(torch.load(backbone, map_location='cpu', weights_only=True))
            self.s8, self.s16, self.s32 = m.features[:7], m.features[7:14], m.features[14:18]     # 32, 96, 320 channels
            self.p32, self.p16, self.p8 = project(320, 1), project(96, 1), project(32, 1)
            self.fuse16, self.fuse8 = project(channels, 3), project(channels, 3)
            self.head = nn.Conv2d(channels, n_out, 1)

        def encoder(self):
            return [self.s8, self.s16, self.s32]

        def forward(self, x):
            a = self.s8(x)
            b = self.s16(a)
            y = self.p32(self.s32(b))
            y = self.fuse16(F.interpolate(y, size=b.shape[-2:], mode='bilinear', align_corners=False) + self.p16(b))
            y = self.fuse8(F.interpolate(y, size=a.shape[-2:], mode='bilinear', align_corners=False) + self.p8(a))
            return F.interpolate(self.head(y), size=x.shape[-2:], mode='bilinear', align_corners=False)

    return Net()


# ---------------------------------------------------------------------------------------------- data
def _load_split(ds_dir: str, split: str, cfg: dict) -> tuple[dict, list[str], np.ndarray, np.ndarray]:
    """A dataset's annotations, the split's frame ids, and their images (N, H, W, 3) uint8 and masks
    (N, H, W) uint8 at the network's input size, padded (mask: IGNORE)."""
    s = cfg['segmenter']
    with open(os.path.join(ds_dir, 'annotations.json')) as f:
        ann = json.load(f)
    if ann['classes'] != list(cfg['perception']['classes']):
        raise ValueError(f'{ds_dir}: classes {ann["classes"]} differ from perception.classes')
    fids = ann['splits'][split]
    W, H = s['input_width'], s['input_height']
    imgs = np.zeros((len(fids), H, W, 3), np.uint8)
    masks = np.full((len(fids), H, W), IGNORE, np.uint8)
    for k, fid in enumerate(fids):
        fr = ann['frames'][fid]
        h = min(H, int(round(fr['height'] * W / fr['width'])))
        imgs[k, :h] = np.asarray(Image.open(os.path.join(ds_dir, fr['image'])).convert('RGB').resize((W, h), Image.BILINEAR))
        masks[k, :h] = np.asarray(Image.open(os.path.join(ds_dir, 'masks', split, fid + '.png')).resize((W, h), Image.NEAREST))
    return ann, fids, imgs, masks


def _normalise(torch, imgs: np.ndarray, cfg: dict, gain=None, contrast=None):
    """Images as the encoder expects them; gain and contrast (per image) jitter them first."""
    s = cfg['segmenter']
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    if gain is not None:
        g, c = (torch.from_numpy(v).float().view(-1, 1, 1, 1) for v in (gain, contrast))
        mu = x.mean((1, 2, 3), keepdim=True)
        x = (((x - mu) * c + mu) * g).clamp(0, 1)
    mean = torch.tensor(s['normalise_mean']).view(1, 3, 1, 1)
    std = torch.tensor(s['normalise_std']).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------------------------- training
def train(datasets: list[str], backbone: str, cfg: dict, seed: int = 0, log=print) -> dict:
    """Fit the segmenter on the train splits of the datasets; returns {'state': ..., 'meta': ...}."""
    torch = _torch()
    s = cfg['segmenter']
    digest = _sha256(backbone)
    if digest != s['backbone_sha256']:
        raise ValueError(f'{backbone}: SHA-256 {digest} is not segmenter.backbone_sha256')
    torch.set_num_threads(s['threads'])
    torch.manual_seed(int(stream(seed, 'segmenter_init').integers(2 ** 31)))
    rng = stream(seed, 'segmenter_train')
    n_out = len(cfg['perception']['classes']) + 1
    parts = [_load_split(d, 'train', cfg) for d in datasets]
    imgs = np.concatenate([p[2] for p in parts])
    masks = np.concatenate([p[3] for p in parts])
    vals = [(d,) + _load_split(d, 'val', cfg)[:2] for d in datasets]
    freq = np.bincount(masks[masks != IGNORE].ravel(), minlength=n_out).astype(float)
    weight = np.zeros(n_out)                                # classes absent from the training pixels: weight 0
    weight[freq > 0] = (freq[freq > 0] / freq.sum()) ** -s['class_weight_power']
    weight /= weight[freq > 0].mean()
    net = build_net(n_out, s['decoder_channels'], backbone)
    enc = {id(p) for m in net.encoder() for p in m.parameters()}
    groups = [{'params': [p for p in net.parameters() if id(p) in enc], 'lr': s['learning_rate'] * s['encoder_lr_factor']},
              {'params': [p for p in net.parameters() if id(p) not in enc], 'lr': s['learning_rate']}]
    opt = torch.optim.AdamW(groups, weight_decay=s['weight_decay'])
    steps = s['epochs'] * math.ceil(len(imgs) / s['batch'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32), ignore_index=IGNORE)
    best, history = None, []
    for epoch in range(s['epochs']):
        net.train()
        t0, total = time.time(), 0.0
        order = rng.permutation(len(imgs))
        for b in range(0, len(order), s['batch']):
            idx = order[b:b + s['batch']]
            xi, yi = imgs[idx].copy(), masks[idx].copy()
            flip = rng.uniform(size=len(idx)) < 0.5
            xi[flip], yi[flip] = xi[flip][:, :, ::-1], yi[flip][:, :, ::-1]
            x = _normalise(torch, xi, cfg, rng.uniform(1 - s['brightness'], 1 + s['brightness'], len(idx)),
                           rng.uniform(1 - s['contrast'], 1 + s['contrast'], len(idx)))
            loss = loss_fn(net(x), torch.from_numpy(yi.astype(np.int64)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
        model = {'state': {k: v.detach().clone() for k, v in net.state_dict().items()}, 'meta': {}}
        scores = [_val_map(net, d, ann, fids, cfg) for d, ann, fids in vals]
        val = float(np.mean([v for v in scores if v is not None])) if any(v is not None for v in scores) else 0.0
        history.append({'epoch': epoch + 1, 'loss': round(total / len(imgs), 4), 'val_map': round(val, 4),
                        'seconds': round(time.time() - t0, 1)})
        log(json.dumps(history[-1]))
        if best is None or val > best[0]:
            best = (val, epoch + 1, model['state'])
    meta = {'classes': list(cfg['perception']['classes']), 'settings': {k: v for k, v in s.items() if k != 'threads'},
            'backbone_sha256': digest, 'seed': seed, 'epoch': best[1], 'val_map': round(best[0], 4), 'history': history,
            'trained_on': [{'dataset': os.path.abspath(d), 'annotations_sha256': _sha256(os.path.join(d, 'annotations.json')),
                            'frames': len(p[1])} for d, p in zip(datasets, parts)]}
    return {'state': best[2], 'meta': meta}


def save(model: dict, path: str):
    torch = _torch()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'state': model['state'], 'meta': json.dumps(model['meta'])}, path)


def load(path: str, cfg: dict):
    """The network, its record, and cfg with the model's own input settings."""
    torch = _torch()
    blob = torch.load(registry.resolve(path, cfg), map_location='cpu', weights_only=True)
    meta = json.loads(blob['meta'])
    if meta['classes'] != list(cfg['perception']['classes']):
        raise ValueError(f'{path}: the model was trained for classes {meta["classes"]}, not perception.classes')
    net = build_net(len(meta['classes']) + 1, meta['settings']['decoder_channels'])
    net.load_state_dict(blob['state'])
    net.eval()
    run_cfg = copy.deepcopy(cfg)
    run_cfg['segmenter'].update({k: meta['settings'][k] for k in ('input_width', 'input_height', 'normalise_mean', 'normalise_std')})
    return net, meta, run_cfg


def load_for_benchmark(path: str, cfg: dict) -> dict:
    """A model file loaded for benchmark.run_hull: the network, its record, its input settings and its digest."""
    net, meta, run_cfg = load(path, cfg)
    return {'net': net, 'meta': meta, 'cfg': run_cfg, 'sha256': _sha256(registry.resolve(path, cfg))}


# ---------------------------------------------------------------------------------------------- detection
def _probabilities(net, imgs: np.ndarray, cfg: dict) -> np.ndarray:
    torch = _torch()
    out = []
    with torch.no_grad():
        for b in range(0, len(imgs), cfg['segmenter']['batch']):
            out.append(torch.softmax(net(_normalise(torch, imgs[b:b + cfg['segmenter']['batch']], cfg)), 1).numpy())
    return np.concatenate(out) if out else np.zeros((0,))


def frame_detections(prob: np.ndarray, width: int, height: int, classes: list[str], cfg: dict) -> tuple[list[dict], np.ndarray]:
    """Boxes of one frame from its class probabilities (C, H_in, W_in) at the network's input size, and
    the predicted class mask at the frame's size (0 = background)."""
    s = cfg['segmenter']
    h = min(s['input_height'], int(round(height * s['input_width'] / width)))
    p = prob[:, :h]
    up = np.stack([np.asarray(Image.fromarray(c).resize((width, height), Image.BILINEAR)) for c in p])
    pred = up.argmax(0)
    dets = []
    for k, cls in enumerate(classes, 1):
        lab, n = ndimage.label(pred == k)
        if n == 0:
            continue
        for r, sl in enumerate(ndimage.find_objects(lab), 1):
            region = lab[sl] == r
            if region.sum() < s['min_region_px']:
                continue
            dets.append({'class': cls, 'score': round(float(up[k][sl][region].mean()), 4),
                         'box': [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop], 'area_px': int(region.sum())})
    return dets, pred


def detect(net, ds_dir: str, split: str, cfg: dict, ann: dict | None = None, fids: list[str] | None = None) -> tuple[dict, np.ndarray]:
    """Detections of a dataset split for the perception referee, and the pixel confusion matrix
    (rows: truth, columns: prediction, at the dataset's resolution)."""
    if ann is None:
        with open(os.path.join(ds_dir, 'annotations.json')) as f:
            ann = json.load(f)
        fids = ann['splits'][split]
    classes = ann['classes']
    n = len(classes) + 1
    conf = np.zeros((n, n), np.int64)
    out = {}
    paths = {fid: os.path.join(ds_dir, ann['frames'][fid]['image']) for fid in fids}
    for fid, (dets, pred) in _predict(net, paths, classes, cfg):
        out[fid] = dets
        truth = np.asarray(Image.open(os.path.join(ds_dir, 'masks', split, fid + '.png')))
        conf += np.bincount(truth.ravel().astype(np.int64) * n + pred.ravel(), minlength=n * n).reshape(n, n)
    return out, conf


def run_frames(net, run_dir: str, cfg: dict) -> dict:
    """Detections for a recorded run's inspection-camera frames, keyed by frame name (for the defect
    map and report: `report --detections`). The map leaves the stations' glare zones out itself."""
    recs = [r for r in (json.loads(line) for line in open(os.path.join(run_dir, 'frames.jsonl')))
            if r['camera'] == cfg['survey']['camera'] and r.get('signal', True)]
    paths = {r['frame']: os.path.join(run_dir, 'frames', r['frame'] + '.png') for r in recs}
    return {name: dets for name, (dets, _) in _predict(net, paths, list(cfg['perception']['classes']), cfg)}


def _predict(net, paths: dict, classes: list[str], cfg: dict):
    """(name, (detections, predicted class mask)) for each image path, in batches."""
    W, H = cfg['segmenter']['input_width'], cfg['segmenter']['input_height']
    names = list(paths)
    for b in range(0, len(names), cfg['segmenter']['batch']):
        chunk = names[b:b + cfg['segmenter']['batch']]
        imgs = np.zeros((len(chunk), H, W, 3), np.uint8)
        sizes = []
        for k, name in enumerate(chunk):
            im = Image.open(paths[name]).convert('RGB')
            sizes.append(im.size)
            h = min(H, int(round(im.size[1] * W / im.size[0])))
            imgs[k, :h] = np.asarray(im.resize((W, h), Image.BILINEAR))
        for name, (w, h), prob in zip(chunk, sizes, _probabilities(net, imgs, cfg)):
            yield name, frame_detections(prob, w, h, classes, cfg)


def pixel_iou(conf: np.ndarray, classes: list[str]) -> dict:
    """Per-class IoU from a pixel confusion matrix (classes with no pixels in truth or prediction: None)."""
    tp = np.diag(conf).astype(float)
    union = conf.sum(0) + conf.sum(1) - tp
    out = {c: (round(float(tp[k] / union[k]), 4) if union[k] else None) for k, c in enumerate(['background'] + classes)}
    present = [out[c] for k, c in enumerate(classes, 1) if conf[k].sum() > 0]
    out['mean_over_present'] = round(float(np.mean(present)), 4) if present else None
    return out


def _val_map(net, ds_dir: str, ann: dict, fids: list[str], cfg: dict) -> float | None:
    if not fids:
        return None
    net.eval()
    det, _ = detect(net, ds_dir, 'val', cfg, ann, fids)
    return perception.evaluate(ann, det, cfg, 'val')['map']
