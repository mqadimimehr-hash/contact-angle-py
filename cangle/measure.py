"""Static contact-angle measurement from side-view sessile-drop images.

Pipeline
--------
1. grayscale -> Gaussian blur
2. segmentation of the dark drop: Otsu threshold (default) or Canny edges
   + flood fill of the background
3. baseline: user-given, auto-detected from the dark substrate band (sub-pixel,
   from columns left/right of the drop), from the lowest drop point, or
   -- for reflective substrates -- from the mirror-symmetry of the
   drop + reflection silhouette
4. sub-pixel edge points of the drop profile above the baseline
5. contact angle by
   (a) circle fit (spherical-cap model, Kasa + geometric refinement)
   (b) local polynomial fit near each contact point (tangent method)
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

__all__ = [
    "AngleResult",
    "DropMeasurement",
    "classify",
    "load_image",
    "measure",
    "measure_file",
    "annotate",
    "batch_measure",
    "IMAGE_EXTENSIONS",
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
EDGE_MARGIN = 2.0  # px above the baseline ignored (blur / meniscus merge zone)


# --------------------------------------------------------------------------- #
# data containers
# --------------------------------------------------------------------------- #
@dataclass
class AngleResult:
    method: str
    left: float
    right: float
    contact_left: Tuple[float, float]
    contact_right: Tuple[float, float]

    @property
    def mean(self) -> float:
        return 0.5 * (self.left + self.right)

    @property
    def classification(self) -> str:
        return classify(self.mean)


@dataclass
class DropMeasurement:
    baseline_y: float
    baseline_source: str
    circle: AngleResult
    tangent: Optional[AngleResult]
    circle_params: Tuple[float, float, float]  # xc, yc, R (image coords)
    circle_rms: float
    points: np.ndarray = field(repr=False)  # (N, 2) sub-pixel edge points x, y
    tangent_fits: dict = field(default_factory=dict, repr=False)
    warnings: List[str] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.circle.mean

    @property
    def classification(self) -> str:
        return self.circle.classification

    def summary(self) -> dict:
        d = {
            "baseline_y": round(self.baseline_y, 2),
            "baseline_source": self.baseline_source,
            "circle_left": round(self.circle.left, 2),
            "circle_right": round(self.circle.right, 2),
            "circle_mean": round(self.circle.mean, 2),
            "circle_rms_px": round(self.circle_rms, 3),
            "classification": self.classification,
        }
        if self.tangent is not None:
            d.update(
                tangent_left=round(self.tangent.left, 2),
                tangent_right=round(self.tangent.right, 2),
                tangent_mean=round(self.tangent.mean, 2),
            )
        else:
            d.update(tangent_left="", tangent_right="", tangent_mean="")
        d["warnings"] = "; ".join(self.warnings)
        return d


def classify(angle_deg: float) -> str:
    """Wettability class of a static water contact angle (degrees)."""
    if angle_deg < 10:
        return "superhydrophilic"
    if angle_deg < 90:
        return "hydrophilic"
    if angle_deg <= 150:
        return "hydrophobic"
    return "superhydrophobic"


# --------------------------------------------------------------------------- #
# image handling / segmentation
# --------------------------------------------------------------------------- #
def load_image(path: Union[str, os.PathLike]) -> np.ndarray:
    """Load an image file as uint8 grayscale."""
    img = cv2.imread(os.fspath(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return to_gray(img)


def to_gray(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    return img


def _segment(blur: np.ndarray, edge: str, canny: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """Return (dark-object mask, grey level used for sub-pixel edges)."""
    t_otsu, otsu_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark, bright = blur[otsu_mask > 0], blur[otsu_mask == 0]
    if dark.size == 0 or bright.size == 0:
        raise ValueError("image has no contrast; cannot segment a drop")
    level = 0.5 * (float(np.median(dark)) + float(np.median(bright)))
    if edge == "otsu":
        return otsu_mask > 0, level
    if edge != "canny":
        raise ValueError("edge must be 'otsu' or 'canny'")
    edges = cv2.Canny(blur, canny[0], canny[1])
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    h, w = edges.shape
    flood = (edges > 0).astype(np.uint8)  # 1 = barrier
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    ff_mask[1:-1, 1:-1] = flood
    canvas = np.zeros((h, w), np.uint8)
    for x in range(0, w, max(1, w // 64)):  # seed along the top row
        if flood[0, x] == 0 and ff_mask[1, x + 1] == 0:
            cv2.floodFill(canvas, ff_mask, (x, 0), 255, flags=4 | (255 << 8))
    background = ff_mask[1:-1, 1:-1] == 255
    return ~background, level


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        raise ValueError("no drop found in image")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == k


def _crossing(g0: float, g1: float, level: float) -> float:
    """Fractional position (0..1) between two samples where the profile crosses level."""
    d = g0 - g1
    if abs(d) < 1e-9:
        return 0.5
    return float(np.clip((g0 - level) / d, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #
def _baseline_from_substrate(blur, mask, level):
    h, w = mask.shape
    frac = mask.mean(axis=1)
    rows = np.flatnonzero(frac > 0.9)
    if rows.size == 0:
        return None
    r0 = int(rows[0])
    if r0 < 3:
        return None
    drop = _largest_component(mask[:r0])
    xs = np.flatnonzero(drop.any(axis=0))
    pad = 10
    cols = [x for x in range(w) if x < xs.min() - pad or x > xs.max() + pad]
    ys = []
    for x in cols:
        col = blur[:, x].astype(float)
        for y in range(max(1, r0 - 4), min(h, r0 + 3)):
            if col[y] < level <= col[y - 1]:
                ys.append(y - 1 + _crossing(col[y - 1], col[y], level))
                break
    if len(ys) >= 5:
        return float(np.median(ys)), "substrate"
    return r0 - 0.5, "substrate"


def _baseline_from_reflection(mask):
    blob = _largest_component(mask)
    rows = np.flatnonzero(blob.any(axis=1))
    top, bot = int(rows[0]), int(rows[-1])
    widths = np.zeros(mask.shape[0])
    for y in range(top, bot + 1):
        xs = np.flatnonzero(blob[y])
        widths[y] = xs[-1] - xs[0] + 1
    hgt = bot - top + 1
    best, best_c = np.inf, None
    for r in range(top + hgt // 4, bot - hgt // 4 + 1):
        k = min(r - top, bot - r - 1, hgt // 4)
        if k < 3:
            continue
        for off in (0, 1):  # centre on row r (off=0) or between r and r+1
            up = widths[r - np.arange(k)]
            dn = widths[r + off + np.arange(k)]
            cost = float(np.mean(np.abs(up - dn)))
            if cost < best:
                best, best_c = cost, r + 0.5 * off
    if best_c is None:
        raise ValueError("reflection mode: silhouette too small")
    return float(best_c), "reflection"


def _detect_baseline(blur, mask, level, reflection):
    if reflection:
        return _baseline_from_reflection(mask)
    res = _baseline_from_substrate(blur, mask, level)
    if res is not None:
        return res
    return _baseline_from_lowest_point(blur, mask, level)


def _baseline_from_lowest_point(blur, mask, level):
    """Baseline = bottom edge of the drop silhouette (sub-pixel, central columns)."""
    blob = _largest_component(mask)
    h = mask.shape[0]
    bot = int(np.flatnonzero(blob.any(axis=1))[-1])
    xs = np.flatnonzero(blob[bot])
    g = blur.astype(float)
    ys = [bot + _crossing(g[bot, x], g[bot + 1, x], level) for x in xs
          if bot + 1 < h and g[bot + 1, x] >= level > g[bot, x]]
    if ys:
        return float(np.median(ys)), "lowest-point"
    return bot + 0.5, "lowest-point"


# --------------------------------------------------------------------------- #
# profile extraction
# --------------------------------------------------------------------------- #
def _edge_points(blur, mask, baseline_y, level):
    h, w = mask.shape
    cut = int(math.floor(baseline_y - EDGE_MARGIN))
    if cut < 3:
        raise ValueError("baseline too close to the top of the image")
    region = mask.copy()
    region[max(cut, 0):] = False
    drop = _largest_component(region)
    g = blur.astype(float)
    pts = []
    for y in np.flatnonzero(drop.any(axis=1)):
        xs = np.flatnonzero(drop[y])
        xl, xr = xs[0], xs[-1]
        if xl > 0:
            pts.append((xl - 1 + _crossing(g[y, xl - 1], g[y, xl], level), y))
        if xr < w - 1:
            pts.append((xr + _crossing(g[y, xr], g[y, xr + 1], level), y))
    for x in np.flatnonzero(drop.any(axis=0)):
        yt = np.flatnonzero(drop[:, x])[0]
        if yt > 0:
            pts.append((x, yt - 1 + _crossing(g[yt - 1, x], g[yt, x], level)))
    return np.asarray(pts, float), drop


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def fit_circle(pts: np.ndarray) -> Tuple[float, float, float, float]:
    """Least-squares circle: algebraic (Kasa) start + Gauss-Newton refinement.

    Returns xc, yc, R, rms residual (px).
    """
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    xc, yc = c[0] / 2, c[1] / 2
    R = math.sqrt(max(c[2] + xc**2 + yc**2, 1e-12))
    p = np.array([xc, yc, R])
    for _ in range(20):
        dx, dy = x - p[0], y - p[1]
        d = np.hypot(dx, dy)
        d[d == 0] = 1e-12
        r = d - p[2]
        J = np.column_stack([-dx / d, -dy / d, -np.ones_like(d)])
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        p += step
        if np.max(np.abs(step)) < 1e-6:
            break
    res = np.hypot(x - p[0], y - p[1]) - p[2]
    return float(p[0]), float(p[1]), float(p[2]), float(np.sqrt(np.mean(res**2)))


def _circle_angle(xc, yc, R, yb) -> AngleResult:
    cosv = (yc - yb) / R
    if abs(cosv) >= 1:
        raise ValueError("fitted circle does not intersect the baseline")
    theta = math.degrees(math.acos(cosv))
    half = math.sqrt(R**2 - (yb - yc) ** 2)
    return AngleResult("circle", theta, theta, (xc - half, yb), (xc + half, yb))


def _tangent_side(pts, x0, yb, side, theta_seed, radius_px, degree):
    """Polynomial fit to profile points near one contact point."""
    x_of_y = 45.0 <= theta_seed <= 135.0
    fit = None
    for _ in range(2):  # re-centre the neighbourhood on the refined contact point
        d = np.hypot(pts[:, 0] - x0, pts[:, 1] - yb)
        sel = pts[d < radius_px]
        if len(sel) < degree + 3:
            raise ValueError("too few edge points near contact point")
        if x_of_y:
            p = np.polyfit(sel[:, 1], sel[:, 0], degree)
            xc = float(np.polyval(p, yb))
            t = np.array([np.polyval(np.polyder(p), yb), 1.0])
        else:
            p = np.polyfit(sel[:, 0], sel[:, 1], degree)
            q = p.copy()
            q[-1] -= yb
            roots = np.roots(q)
            roots = roots[np.abs(roots.imag) < 1e-9].real
            if roots.size == 0:
                raise ValueError("tangent fit does not reach the baseline")
            xc = float(roots[np.argmin(np.abs(roots - x0))])
            t = np.array([1.0, np.polyval(np.polyder(p), xc)])
        u = sel.mean(axis=0) - np.array([xc, yb])
        if np.dot(t, u) < 0:
            t = -t
        fit = dict(coeffs=p, x_of_y=x_of_y, contact=(xc, yb), direction=t / np.linalg.norm(t))
        x0 = xc
    tx, tup = t[0], -t[1]
    if side == "right":
        tx = -tx
    ang = math.degrees(math.atan2(tup, tx))
    return ang, fit


def measure(
    image: np.ndarray,
    baseline_y: Optional[float] = None,
    edge: str = "otsu",
    reflection: bool = False,
    blur_sigma: float = 1.0,
    canny: Tuple[int, int] = (50, 150),
    tangent_degree: int = 2,
    tangent_radius: Optional[float] = None,
) -> DropMeasurement:
    """Measure the static contact angle of a (dark) drop in a side-view image.

    Parameters
    ----------
    image : grayscale or BGR array.
    baseline_y : baseline row in pixels (sub-pixel allowed). Auto if ``None``.
    edge : ``"otsu"`` (threshold) or ``"canny"`` segmentation.
    reflection : substrate is mirror-like; the drop's reflection is visible
        below the contact line (used only for automatic baseline detection).
    tangent_radius : neighbourhood (px) around each contact point used for the
        polynomial tangent fit; default max(15 % of base width, 12 % of cap radius), clamped 8-60 px.
    """
    gray = to_gray(image)
    blur = cv2.GaussianBlur(gray, (0, 0), blur_sigma) if blur_sigma > 0 else gray
    mask, level = _segment(blur, edge, canny)
    warnings: List[str] = []
    if baseline_y is None:
        baseline_y, source = _detect_baseline(blur, mask, level, reflection)
    else:
        baseline_y, source = float(baseline_y), "user"

    pts, drop = _edge_points(blur, mask, baseline_y, level)
    if len(pts) < 10:
        raise ValueError("too few edge points on drop profile")
    h, w = drop.shape
    if drop[0].any():
        warnings.append("drop touches top border (needle in frame?)")
    if drop[:, 0].any() or drop[:, -1].any():
        warnings.append("drop touches left/right border")

    xc, yc, R, rms = fit_circle(pts)
    circ = _circle_angle(xc, yc, R, baseline_y)
    if rms > 1.0:
        warnings.append(f"poor circle fit (rms {rms:.2f} px): drop may not be a spherical cap")
    if circ.mean > 160:
        warnings.append("angle > 160 deg: contact region is barely resolved, low confidence")

    tangent, fits = None, {}
    base = circ.contact_right[0] - circ.contact_left[0]
    rad = tangent_radius or float(np.clip(max(0.15 * base, 0.12 * R), 8.0, 60.0))
    try:
        la, lf = _tangent_side(pts[pts[:, 0] <= xc], circ.contact_left[0], baseline_y,
                               "left", circ.mean, rad, tangent_degree)
        ra, rf = _tangent_side(pts[pts[:, 0] >= xc], circ.contact_right[0], baseline_y,
                               "right", circ.mean, rad, tangent_degree)
        tangent = AngleResult("tangent", la, ra, lf["contact"], rf["contact"])
        fits = {"left": lf, "right": rf}
    except (ValueError, np.linalg.LinAlgError) as exc:
        warnings.append(f"tangent method failed: {exc}")

    return DropMeasurement(baseline_y, source, circ, tangent, (xc, yc, R), rms,
                           pts, fits, warnings)


def measure_file(path, **kwargs) -> DropMeasurement:
    return measure(load_image(path), **kwargs)


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def annotate(image: np.ndarray, m: DropMeasurement) -> np.ndarray:
    """Return a BGR copy of ``image`` with baseline, fit and angles drawn."""
    gray = to_gray(image)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape
    S = 16  # fixed-point shift for sub-pixel drawing
    f = lambda v: int(round(v * S))  # noqa: E731
    yb = m.baseline_y
    cv2.line(out, (0, f(yb)), (f(w - 1), f(yb)), (255, 128, 0), 1, cv2.LINE_AA, 4)
    for x, y in m.points[:: max(1, len(m.points) // 400)]:
        cv2.circle(out, (f(x), f(y)), 1 * S, (0, 200, 255), -1, cv2.LINE_AA, 4)
    xc, yc, R = m.circle_params
    cv2.circle(out, (f(xc), f(yc)), f(R), (0, 200, 0), 1, cv2.LINE_AA, 4)
    if m.tangent is not None:
        L = max(20.0, 0.15 * R)
        for side in ("left", "right"):
            fit = m.tangent_fits[side]
            cx, cy = fit["contact"]
            d = fit["direction"]
            cv2.line(out, (f(cx), f(cy)), (f(cx + L * d[0]), f(cy + L * d[1])),
                     (0, 0, 255), 2, cv2.LINE_AA, 4)
    for cx, cy in (m.circle.contact_left, m.circle.contact_right):
        cv2.circle(out, (f(cx), f(cy)), 3 * S, (255, 0, 255), -1, cv2.LINE_AA, 4)
    lines = [f"circle: L {m.circle.left:.1f}  R {m.circle.right:.1f}  mean {m.circle.mean:.1f} deg"]
    if m.tangent is not None:
        lines.append(f"tangent: L {m.tangent.left:.1f}  R {m.tangent.right:.1f}  "
                     f"mean {m.tangent.mean:.1f} deg")
    lines.append(m.classification)
    scale = max(0.4, w / 1200)
    for i, txt in enumerate(lines):
        org = (10, int(25 * scale * 1.3 * (i + 1)))
        cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 180), 1, cv2.LINE_AA)
    return out


CSV_FIELDS = ["file", "baseline_y", "baseline_source", "circle_left", "circle_right",
              "circle_mean", "circle_rms_px", "tangent_left", "tangent_right",
              "tangent_mean", "classification", "warnings", "error"]


def iter_images(folder) -> List[str]:
    return sorted(
        os.path.join(folder, n) for n in os.listdir(folder)
        if n.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(folder, n))
    )


def batch_measure(folder, csv_path=None, annotate_dir=None, **kwargs) -> List[dict]:
    """Measure every image in ``folder``; optionally write a CSV and annotated images.

    Failures are recorded in the ``error`` column instead of aborting the batch.
    """
    rows = []
    if annotate_dir:
        os.makedirs(annotate_dir, exist_ok=True)
    for path in iter_images(folder):
        row = {k: "" for k in CSV_FIELDS}
        row["file"] = os.path.basename(path)
        try:
            img = load_image(path)
            m = measure(img, **kwargs)
            row.update(m.summary())
            if annotate_dir:
                stem = os.path.splitext(row["file"])[0]
                cv2.imwrite(os.path.join(annotate_dir, stem + "_annotated.png"), annotate(img, m))
        except Exception as exc:  # keep going on bad images
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            wr.writeheader()
            wr.writerows(rows)
    return rows
