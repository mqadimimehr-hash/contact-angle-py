import csv

import cv2
import numpy as np
import pytest

from cangle import classify, make_drop, measure
from cangle.cli import main

ANGLES = [30, 60, 90, 120, 150]
CIRCLE_TOL = 3.0
TANGENT_TOL = 5.0  # looser: local polynomial fits are noisier (see README)


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("edge", ["otsu", "canny"])
def test_circle_fit_auto_baseline(angle, edge):
    m = measure(make_drop(angle, noise_sigma=3, seed=angle), edge=edge)
    assert m.baseline_source == "substrate"
    assert abs(m.baseline_y - 360) < 0.5
    assert abs(m.circle.mean - angle) <= CIRCLE_TOL
    assert m.classification == classify(m.circle.mean)


@pytest.mark.parametrize("angle", ANGLES)
def test_tangent_method(angle):
    m = measure(make_drop(angle, noise_sigma=3, seed=angle))
    assert m.tangent is not None, m.warnings
    assert abs(m.tangent.left - angle) <= TANGENT_TOL
    assert abs(m.tangent.right - angle) <= TANGENT_TOL


@pytest.mark.parametrize("angle", ANGLES)
def test_user_baseline(angle):
    m = measure(make_drop(angle, noise_sigma=2), baseline_y=360.0)
    assert m.baseline_source == "user"
    assert abs(m.circle.mean - angle) <= CIRCLE_TOL


@pytest.mark.parametrize("angle", ANGLES)
def test_no_substrate_lowest_point(angle):
    m = measure(make_drop(angle, substrate="none", noise_sigma=2))
    assert m.baseline_source == "lowest-point"
    assert abs(m.circle.mean - angle) <= CIRCLE_TOL


@pytest.mark.parametrize("angle", ANGLES)
def test_reflective_substrate(angle):
    img = make_drop(angle, height=640, substrate="reflective", noise_sigma=2)
    m = measure(img, reflection=True)
    assert abs(m.baseline_y - 360) <= 1.0
    assert abs(m.circle.mean - angle) <= CIRCLE_TOL


def test_off_centre_and_colour_input():
    img = make_drop(100, center_x=250, radius=120)
    m = measure(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    assert abs(m.circle.mean - 100) <= CIRCLE_TOL
    assert abs(m.circle_params[0] - 250) < 1
    assert abs(m.circle_params[2] - 120) < 1


@pytest.mark.parametrize(
    "angle,label",
    [(5, "superhydrophilic"), (45, "hydrophilic"), (95, "hydrophobic"),
     (150, "hydrophobic"), (155, "superhydrophobic")],
)
def test_classify(angle, label):
    assert classify(angle) == label


def test_blank_image_raises():
    with pytest.raises(ValueError):
        measure(np.full((100, 100), 200, np.uint8))


def test_synthetic_rejects_bad_angle():
    with pytest.raises(ValueError):
        make_drop(180)


def test_cli_measure_and_batch(tmp_path, capsys):
    for a in (60, 120):
        cv2.imwrite(str(tmp_path / f"drop_{a}.png"), make_drop(a, noise_sigma=2))
    (tmp_path / "notes.txt").write_text("ignored")
    cv2.imwrite(str(tmp_path / "blank.png"), np.full((50, 50), 255, np.uint8))

    out_img = tmp_path.parent / f"{tmp_path.name}_ann.png"
    assert main(["measure", str(tmp_path / "drop_60.png"), "--annotate", str(out_img)]) == 0
    assert "hydrophilic" in capsys.readouterr().out
    assert cv2.imread(str(out_img)).shape == (480, 640, 3)

    csv_path = tmp_path / "results.csv"
    assert main(["batch", str(tmp_path), "-o", str(csv_path)]) == 0
    rows = {r["file"]: r for r in csv.DictReader(open(csv_path))}
    assert set(rows) == {"blank.png", "drop_60.png", "drop_120.png"}
    assert rows["blank.png"]["error"]
    assert abs(float(rows["drop_120.png"]["circle_mean"]) - 120) <= CIRCLE_TOL
    assert rows["drop_120.png"]["classification"] == "hydrophobic"
