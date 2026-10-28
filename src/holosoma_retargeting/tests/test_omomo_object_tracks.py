"""Cleaning OMOMO box tracks: symmetry flips, dropouts and smoothing."""

from __future__ import annotations

import numpy as np

from holosoma_retargeting.data_utils.omomo_objects import BOX_FLIPS, _angle, clean_object_track

FPS = 30.0


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _track(n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """A box sliding at 0.6 m/s while turning at 30 deg/s."""
    trans = np.stack([np.arange(n) * 0.02, np.zeros(n), np.full(n, 0.2)], axis=1)
    rot = np.stack([_rot_z(np.radians(i)) for i in range(n)])
    return trans, rot


def test_a_clean_track_is_unchanged_without_smoothing() -> None:
    trans, rot = _track()
    t2, r2, report = clean_object_track(trans, rot, FPS, 5.0, 900.0, 0.0)
    assert report == {"flips": 0, "dropouts": 0, "residual": 0}
    np.testing.assert_allclose(t2, trans)
    np.testing.assert_allclose(r2, rot)


def test_a_symmetry_flip_is_undone_for_the_rest_of_the_clip() -> None:
    trans, rot = _track()
    rot[30:] = rot[30:] @ BOX_FLIPS[2]
    _, r2, report = clean_object_track(trans, rot, FPS, 5.0, 900.0, 0.0)
    assert report["flips"] == 1 and report["residual"] == 0
    assert np.degrees(_angle(r2[29], r2[30])) < 2.0


def test_a_multi_frame_dropout_is_interpolated() -> None:
    trans, rot = _track()
    trans[20:24] += np.array([1.5, -0.8, 0.0])
    t2, _, report = clean_object_track(trans, rot, FPS, 5.0, 900.0, 0.0)
    assert report["dropouts"] == 1 and report["residual"] == 0
    np.testing.assert_allclose(t2[20:24], _track()[0][20:24], atol=1e-9)


def test_a_jump_that_never_returns_is_reported() -> None:
    trans, rot = _track()
    trans[30:] += np.array([2.0, 0.0, 0.0])
    _, _, report = clean_object_track(trans, rot, FPS, 5.0, 900.0, 0.0)
    assert report["residual"] == 1
