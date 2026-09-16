"""Command-line interface: ``cangle measure`` / ``cangle batch`` / ``cangle synth``."""
from __future__ import annotations

import argparse
import json
import sys

import cv2

from . import __version__
from .measure import annotate, batch_measure, load_image, measure
from .synthetic import make_drop


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--baseline", type=float, default=None,
                   help="baseline row in pixels (default: auto-detect)")
    p.add_argument("--edge", choices=("otsu", "canny"), default="otsu",
                   help="segmentation method (default: otsu)")
    p.add_argument("--reflection", action="store_true",
                   help="substrate is reflective (drop mirror image visible)")
    p.add_argument("--blur", type=float, default=1.0, help="Gaussian blur sigma in px")


def _kwargs(a):
    return dict(baseline_y=a.baseline, edge=a.edge, reflection=a.reflection, blur_sigma=a.blur)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cangle", description=__doc__)
    ap.add_argument("--version", action="version", version=f"cangle {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="measure one image")
    m.add_argument("image")
    m.add_argument("--annotate", metavar="OUT", help="save annotated image")
    m.add_argument("--json", action="store_true", help="print JSON instead of text")
    _common(m)

    b = sub.add_parser("batch", help="measure all images in a folder")
    b.add_argument("folder")
    b.add_argument("-o", "--output", default="results.csv", help="CSV path")
    b.add_argument("--annotate-dir", default=None, help="folder for annotated images")
    _common(b)

    s = sub.add_parser("synth", help="write a synthetic drop with a known angle")
    s.add_argument("angle", type=float)
    s.add_argument("output")
    s.add_argument("--noise", type=float, default=0.0)
    s.add_argument("--substrate", choices=("dark", "none", "reflective"), default="dark")

    a = ap.parse_args(argv)

    if a.cmd == "measure":
        img = load_image(a.image)
        try:
            res = measure(img, **_kwargs(a))
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        summary = res.summary()
        if a.json:
            print(json.dumps(summary, indent=2))
        else:
            c, t = res.circle, res.tangent
            print(f"baseline y      : {res.baseline_y:.2f} px ({res.baseline_source})")
            print(f"circle fit      : left {c.left:.2f}  right {c.right:.2f}  "
                  f"mean {c.mean:.2f} deg  (rms {res.circle_rms:.2f} px)")
            if t is not None:
                print(f"tangent (poly)  : left {t.left:.2f}  right {t.right:.2f}  mean {t.mean:.2f} deg")
            print(f"classification  : {res.classification}")
            for w in res.warnings:
                print(f"warning         : {w}")
        if a.annotate:
            cv2.imwrite(a.annotate, annotate(img, res))
        return 0

    if a.cmd == "batch":
        rows = batch_measure(a.folder, csv_path=a.output, annotate_dir=a.annotate_dir, **_kwargs(a))
        bad = sum(1 for r in rows if r["error"])
        print(f"{len(rows)} images, {bad} failed -> {a.output}")
        return 0 if rows and bad < len(rows) else 1

    if a.cmd == "synth":
        cv2.imwrite(a.output, make_drop(a.angle, noise_sigma=a.noise, substrate=a.substrate))
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
