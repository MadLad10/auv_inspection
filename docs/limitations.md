# Limitations

- Simulation only. Numbers describe this simulator and must not be read as real-world performance.
- Fleet: vehicles do not appear in each other's images, and nothing physical stops two vehicles
  meeting: the referee measures their separation, and the rules keep it. The acoustic channel is a
  simple model: perfectly synchronised slots, a loss chance that depends on range only (no
  multipath geometry, no sound-speed profile). The ship's positioning has Gaussian errors with no
  outliers, and the vehicle applies a late fix as the innovation it had when it was measured.
  Handing work over waits for a clear path, so a vehicle may wait while a neighbour finishes.
- Defect appearance in the simulator is procedural, or cut from LIACi's real images (corrosion,
  coating damage, fouling; cracks and missing anodes stay procedural). Real patches do not make
  the rest of the image real: a segmenter trained only on simulated hulls fails on LIACi's images.
  LIACi itself is small, from a few inspections, and not ship-disjoint between training and test.
- The detector's hand-written class rules were written for red-brown paint (on black paint they
  take fouling for corrosion); the region classifier, trained on all three paints, replaced them.
  Paints it has not seen are untested.
- Cracks: a crack 3 to 6 mm wide at 4 mm per pixel shows a pixel contrast (median 3.5 to 8 median
  deviations of a valley filter) within the paint's grain (its 99.9th percentile 8 to 16), and the
  candidate regions' one-pixel opening removes most crack pixels. The region classifier finds 23% of
  findable cracks on the test hulls; the deep segmenter finds 8 to 10 of 13.
- The deep segmenter makes about five false confirmed findings per test hull, against 0.25 for the
  classical detector on procedural defects. The map's confirmation settings were tuned with the
  classical detector. Its weights, trained on LIACi-derived data, cannot go into the repository or
  CI, so the CI gate does not cover it.
- Light: single scattering for backscatter, a single-scattering halo scaled to the multiply
  scattered power for forward scatter, Lambertian surfaces, three colour bands rather than spectra,
  no shadows cast by anodes, no marine snow. Water is uniform.
- Hydrodynamics: added mass from an equivalent ellipsoid, quadratic drag from strip theory, no
  interaction with the nearby hull (no wall effect), no waves.
- The standoff sounder is an ideal ray with Gaussian range noise. Hull-end fixes assume the echo only
  starts or stops at a hull end.
- The controller's drag feed-forward uses the vehicle model's own damping (scaled by
  `vehicle.control.drag_feedforward`). The onboard assumes the phase function and daylight it is
  configured with; it measures attenuation and scattering.
- Revisits are planned for the nominal water; persistent turbidity leaves holes, which are reported.
- The coverage criterion asks whether a faint stain over a 2 cm cell would be seen; it does not ask
  whether a feature narrower than a pixel is resolved. Cracks 3 to 6 mm wide at 4 mm per pixel can
  count as covered yet show too few pixels to find; the map referee lists them separately.
