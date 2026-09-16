# contact-angle-py (`cangle`)

Measure the **static water contact angle** from side-view (sessile-drop)
photographs, e.g. to characterise hydrophilic, hydrophobic and
superhydrophobic self-cleaning coatings.

- Dependencies: `numpy` and `opencv-python-headless` only
- Two independent angle estimates: **circle fit** (spherical-cap model) and
  **polynomial tangent fit** at the left and right contact points
- Automatic baseline (dark substrate, lowest drop point, or reflective
  substrate) or a baseline you supply
- Annotated output images, batch mode to CSV, command-line tool
- Synthetic drop generator with a known angle, used by the test suite

## Install

```bash
git clone https://github.com/mqadimimehr-hash/contact-angle-py
cd contact-angle-py
pip install -e .            # add ".[test]" to get pytest
```

## Command line

```bash
cangle measure drop.png --annotate out.png
cangle measure drop.png --baseline 412.5 --json      # give the baseline yourself
cangle measure drop.png --reflection                 # mirror-like substrate
cangle batch folder/ -o results.csv --annotate-dir annotated/
cangle synth 120 test_drop.png --noise 3             # write a synthetic 120 deg drop
```

Example output:

```
baseline y      : 360.01 px (substrate)
circle fit      : left 120.00  right 120.00  mean 120.00 deg  (rms 0.02 px)
tangent (poly)  : left 119.38  right 119.42  mean 119.40 deg
classification  : hydrophobic
```

Other options: `--edge canny` (Canny edges plus background flood fill
instead of an Otsu threshold) and `--blur SIGMA`. The batch CSV has
columns `file, baseline_y, baseline_source, circle_left, circle_right,
circle_mean, circle_rms_px, tangent_left, tangent_right, tangent_mean,
classification, warnings, error`. An image that fails gets a message in
`error`, and the rest of the batch still runs.

## Python API

```python
import cv2
from cangle import load_image, measure, annotate, batch_measure

img = load_image("drop.png")
m = measure(img)                 # or measure(img, baseline_y=412.5, edge="canny")
print(m.circle.left, m.circle.right, m.circle.mean)
print(m.tangent.left, m.tangent.right, m.tangent.mean)  # m.tangent is None if the fit failed
print(m.classification, m.warnings)
cv2.imwrite("out.png", annotate(img, m))

rows = batch_measure("folder/", csv_path="results.csv")
```

## How it works

1. Convert to grayscale and apply a Gaussian blur (sigma 1 px).
2. Segment the dark drop with an Otsu threshold, or with Canny edges and a
   flood fill of the background.
3. Find the baseline:
   - **user**: the value you pass.
   - **substrate**: the first image row that is more than 90 % dark. The
     position is refined to sub-pixel accuracy from columns left and right
     of the drop.
   - **lowest-point**: if there is no dark substrate band, the bottom edge
     of the drop silhouette.
   - **reflection** (`--reflection`): the row about which the silhouette of
     drop plus reflection is most mirror-symmetric.
4. Extract sub-pixel edge points of the drop above the baseline. The 2 px
   just above the baseline are ignored because blur mixes drop and
   substrate there.
5. Compute the angles:
   - **Circle fit**: an algebraic (Kasa) fit, refined by Gauss-Newton
     iterations. The angle is `theta = arccos((yc - yb) / R)`, the same
     on both sides by construction.
   - **Tangent fit**: a quadratic fitted to the edge points near each
     contact point, using x(y) for 45-135 deg and y(x) otherwise. The fit
     is intersected with the baseline and its slope there gives the angle.
     The neighbourhood is centred on the contact points from the circle
     fit and then refined. Left and right angles are computed separately,
     so the method can show asymmetry.
6. Classify the mean angle: superhydrophilic < 10 deg, hydrophilic < 90 deg,
   hydrophobic 90-150 deg, superhydrophobic > 150 deg.

The drop must be **darker than the background**, which is the usual
back-lit (shadowgraph) setup. Invert the image first if yours is the
other way round.

## Accuracy: what has actually been tested

All numbers below come from **synthetic** images made by
`cangle.synthetic.make_drop`. Each image is a 640x480 px, 4x-supersampled
circular cap on a dark flat substrate, with 0.8 px blur and Gaussian noise
sigma = 3 grey levels. The drop fills about 80 % of the width. These
images are ideal. The drops are exact circles, the baseline is perfectly
horizontal, the contrast is high, and there are no reflections, needle,
dust or gravity flattening. **Real photos will be less accurate.** Treat
these numbers as a check that the implementation is correct, not as the
accuracy you will get on a real sample.

| true angle | circle fit (auto baseline) | tangent L / R | circle, baseline +1 px | circle, baseline -1 px | Canny: circle / tangent L,R |
|---:|---:|---:|---:|---:|---:|
| 30  | 30.00  | 29.74 / 29.76   | 30.22  | 29.77  | 29.97 / 29.82, 29.64 |
| 60  | 60.00  | 60.59 / 60.58   | 60.23  | 59.78  | 59.96 / 59.87, 59.84 |
| 90  | 90.00  | 90.09 / 90.12   | 90.22  | 89.77  | 90.01 / 90.17, 90.23 |
| 120 | 120.00 | 119.34 / 119.33 | 120.32 | 119.68 | 120.01 / 119.20, 119.20 |
| 150 | 150.01 | 149.37 / 149.46 | 150.71 | 149.32 | 149.88 / 150.58, 150.58 |

The test suite requires the following:

- **Circle fit** within **±3 deg** of the true angle at 30/60/90/120/150 deg.
  This holds for an automatic, user-given, lowest-point or reflection
  baseline, with Otsu or Canny segmentation.
- **Tangent fit** within **±5 deg**. This tolerance is looser on purpose.
  The method uses only a few pixels near the contact line, so on real
  images it is much more sensitive to noise, blur and baseline error than
  the circle fit. In separate checks on noisy images at 135-160 deg,
  single-side values deviated by up to about 6 deg.
- Above about **160 deg** the tangent fit usually fails and is reported as
  unavailable. The circle fit still returns a value, with a low-confidence
  warning.

Baseline errors matter more for small drops and high angles. On these
images, 1 px of baseline error moved the circle-fit angle by 0.2-0.7 deg.

## Taking good photos

- **Side view, level camera.** Put the optical axis in the substrate plane
  and level the stage. A slightly raised view hides the contact line and
  biases the result.
- **Back-lighting.** Use a diffuse light source behind the drop so the drop
  looks dark on a bright, even background.
- **Focus on the contact line.** Use a macro lens or telecentric optics if
  you can, fill a large part of the frame with the drop, and avoid motion
  blur.
- **Small drops** (a few microlitres) keep gravity flattening low, which is
  what the spherical-cap model assumes. For large drops use the tangent
  values, or better, full Young-Laplace fitting, which this package does
  not do.
- **Remove the needle** from the frame before capturing (a static angle).
- **Keep the frame clean.** Only the drop should cross the region above the
  baseline. Dust, fibres or droplet satellites can be picked up as part of
  the drop.
- **Repeat.** Measure at least 3-5 drops at different spots and report the
  mean ± standard deviation. Coating heterogeneity usually matters more
  than measurement error.

## Limitations

- **Tilt.** The baseline is assumed to be horizontal. Rotate the image
  first if the stage or camera is tilted. A tilted drop also has
  different left and right angles, which the circle fit cannot show.
- **Reflections.** `--reflection` handles a clean mirror image below the
  contact line. Partial or blurry reflections, or a bright glare band at
  the contact line, can move the automatic baseline. In that case give the
  baseline with `--baseline`.
- **Very high angles (> ~160 deg, superhydrophobic).** The contact line is
  hidden under the drop and only a thin wedge is visible. The result
  depends strongly on the baseline, and the tangent fit is usually
  unavailable. Small differences between near-180 deg surfaces are not
  meaningful with this method. Consider sliding or roll-off angle and
  hysteresis measurements as well.
- **Very low angles (< ~10 deg).** The drop is only a few pixels tall and
  the baseline dominates the error.
- **Needle in frame.** If the needle touches the drop, it becomes part of
  the silhouette and corrupts the fit. A warning is issued when the drop
  touches the top border.
- **Gravity.** Large drops are flattened, so the circle fit is a
  simplification. A high `circle_rms_px` warns about this.
- **Static angle only.** There are no advancing or receding angles, no
  hysteresis and no video tracking.
- Only the synthetic images above have been validated. The package has
  **not** been benchmarked against a commercial goniometer. If you compare
  it with one, please share the results in an issue.

## Development

```bash
pip install -e ".[test]"
pytest
```

## License

MIT © 2026 Mohammad Ghadimimehr
