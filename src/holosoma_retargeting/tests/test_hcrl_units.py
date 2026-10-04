"""CPU tests for the hcrl source adapters and solver configuration (no solve, no rendering)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY, G1FK_DEMO_JOINTS, TOE_NAMES_BY_FORMAT
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.terms import SolverTerms, preset_terms
from holosoma_retargeting.hcrl import amass_source, smpl_fk, stance_windows
from holosoma_retargeting.hcrl.limb_retarget import rescale_to_robot_limbs
from holosoma_retargeting.hcrl.source_angles import t1_joint_angle_targets

T1_MODEL = Path(__file__).resolve().parents[1] / "holosoma_retargeting" / "models" / "t1" / "t1_23dof.xml"


def test_rodrigues_rotates_x_onto_y_about_z():
    rot = smpl_fk._rodrigues(np.array([0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(rot @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(smpl_fk._rodrigues(np.zeros(3)), np.eye(3))


def test_to_z_up_is_a_rotation_taking_y_up_to_z_up():
    assert np.linalg.det(smpl_fk._Y_UP_TO_Z_UP) == pytest.approx(1.0)
    np.testing.assert_allclose(smpl_fk.to_z_up(np.array([0.0, 1.0, 0.0])), [0.0, 0.0, 1.0])


def test_resample_picks_nearest_frames():
    values = np.arange(10.0)
    assert amass_source._resample(values, 30.0, 30.0) is values
    np.testing.assert_array_equal(amass_source._resample(values, 60.0, 30.0), [0, 2, 4, 6, 8])


@pytest.mark.parametrize("key", amass_source.FRAME_RATE_KEYS)
def test_frame_rate_reads_every_spelling(key):
    assert amass_source._frame_rate({key: np.array(120)}) == 120.0


def test_frame_rate_refuses_to_guess():
    with pytest.raises(ValueError, match="no frame rate"):
        amass_source._frame_rate({"trans": np.zeros((2, 3))})


@pytest.mark.parametrize(
    ("stored", "gender"),
    [(b"female", "female"), (np.array("Male "), "male"), ("NEUTRAL", "neutral"), ("other", "neutral"), (None, None)],
)
def test_gender_normalizes_its_storage_forms(stored, gender):
    raw = {} if stored is None else {"gender": stored}
    assert amass_source._gender(raw) == (gender or "neutral")


@pytest.mark.parametrize("axis", [1, 2])
def test_up_axis_follows_head_minus_feet(axis):
    joints = np.zeros((4, 16, 3))
    joints[:, 15, axis] = 1.7
    assert amass_source.up_axis(joints) == axis


def test_stance_windows_label_a_lifted_foot_as_swing():
    frames = 60
    src = np.zeros((frames, len(G1FK_DEMO_JOINTS), 3))
    left, right = (G1FK_DEMO_JOINTS.index(n) for n in TOE_NAMES_BY_FORMAT["g1fk"])
    src[:, left] = [0.0, 0.1, 0.02]
    src[:, right] = [0.0, -0.1, 0.02]
    src[30:, left, 2] = 0.2
    masks, windows_left, windows_right = stance_windows.compute(src, np.zeros((0, 6)), flight_tol=0.15)
    assert masks[:28, 0].all() and not masks[32:, 0].any() and masks[:, 1].all()
    assert len(windows_right) == 1 and windows_right[0][:2] == [0, frames - 1]
    assert windows_right[0][4] == pytest.approx(0.02)


def test_limb_rescale_sets_rigid_lengths_and_only_shortens_free_spans():
    names = ["root", "a", "b"]
    keypoints = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.5, 0.0]]])
    out = rescale_to_robot_limbs(
        keypoints, names, {"root": None, "a": "root", "b": "a"}, {"a": 1.0, "b": 1.0}, rigid={"a": True, "b": False}
    )
    np.testing.assert_allclose(out[0, 1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(out[0, 2] - out[0, 1], [0.0, 0.5, 0.0])


def test_limb_rescale_keeps_a_horizontal_segments_drop():
    keypoints = np.array([[[0.0, 0.0, 0.1], [0.05, 0.0, 0.0]]])
    parent = {"ankle": None, "toe": "ankle"}
    out = rescale_to_robot_limbs(keypoints, ["ankle", "toe"], parent, {"toe": 0.2}, horizontal=["toe"])
    assert out[0, 1, 2] == pytest.approx(0.0)
    assert np.linalg.norm(out[0, 1] - out[0, 0]) == pytest.approx(0.2)


@pytest.mark.parametrize("fmt", sorted(DEMO_JOINTS_REGISTRY))
def test_joint_angle_hinges_resolve_by_name(fmt):
    names = DEMO_JOINTS_REGISTRY[fmt]
    targets = t1_joint_angle_targets(np.random.default_rng(0).normal(size=(3, len(names), 3)), names)
    hinges = {"Left_Elbow_Yaw", "Right_Elbow_Yaw", "Left_Knee_Pitch", "Right_Knee_Pitch"}
    assert set(targets) == (set() if fmt == "g1fk" else hinges)


def test_a_right_angle_elbow_reads_pi_over_two_signed_to_its_range():
    names = ["L_Shoulder", "L_Elbow", "L_Wrist", "R_Shoulder", "R_Elbow", "R_Wrist"]
    joints = np.array([[[0, 0, 0], [0, 0, -1], [1, 0, -1], [0, 0, 0], [0, 0, -1], [1, 0, -1]]], dtype=float)
    targets = t1_joint_angle_targets(joints, names)
    assert targets["Left_Elbow_Yaw"][0] == pytest.approx(-np.pi / 2)
    assert targets["Right_Elbow_Yaw"][0] == pytest.approx(np.pi / 2)


def test_preset_terms():
    assert preset_terms(None) == SolverTerms()
    assert preset_terms("t1_keypoint").keypoint_weight == 50.0
    with pytest.raises(ValueError, match="Unknown preset"):
        preset_terms("no_such_preset")


def test_an_explicit_default_wins_over_the_preset():
    from holosoma_retargeting.examples.robot_retarget import parse_config

    args = ["--robot", "t1", "--preset", "t1_keypoint"]
    assert parse_config(args).terms.joint_angle_weight == 20.0
    assert parse_config([*args, "--terms.joint-angle-weight", "5.0"]).terms.joint_angle_weight == 5.0


def test_t1_knee_flexion_is_capped_at_the_urdf_limit():
    model = mujoco.MjModel.from_xml_path(str(T1_MODEL))
    ub = RobotConfig(robot_type="t1").MANUAL_UB
    for knee in ("Left_Knee_Pitch", "Right_Knee_Pitch"):
        adr = int(model.jnt_qposadr[model.joint(knee).id])
        assert ub[str(adr)] == pytest.approx(2.18)
        assert model.jnt_range[model.joint(knee).id][1] > 2.18
