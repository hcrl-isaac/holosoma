"""
Unified robot retargeting script for all task types:
- robot_only: Robot-only retargeting with ground interaction
- object_interaction: Object manipulation retargeting (InterMimic)
- climbing: Climbing retargeting with dynamic terrain
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY, MotionDataConfig, root_keypoint  # noqa: E402
from holosoma_retargeting.config_types.retargeter import RetargeterConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeting import RetargetingConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,  # type: ignore[import-not-found]
)
from holosoma_retargeting.src.utils import (  # noqa: E402
    augment_object_poses,
    calculate_scale_factor,
    create_new_scene_xml_file,
    create_scaled_multi_boxes_urdf,
    create_scaled_multi_boxes_xml,
    estimate_human_orientation,
    extract_foot_sticking_sequence_velocity,
    extract_object_first_moving_frame,
    load_intermimic_data,
    load_object_data,
    preprocess_motion_data,
    transform_from_human_to_world,
    transform_y_up_to_z_up,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _envf(name: str, default: float) -> float:
    """Float override from the environment (the hcrl prior weights; see the v3 block in main())."""
    return float(os.environ.get(name, default))


# ----------------------------- Constants -----------------------------

# Task-specific defaults
DEFAULT_DATA_FORMATS = {
    "robot_only": "smplh",
    "object_interaction": "smplh",
    "climbing": "mocap",
}

DEFAULT_SAVE_DIRS = {
    "robot_only": "demo_results/{robot}/robot_only/omomo",
    "object_interaction": "demo_results/{robot}/object_interaction/omomo",
    "climbing": "demo_results/{robot}/climbing/mocap_climb",
}


# Constants for numpy arrays (not in dataclass to avoid tyro parsing issues)
_OBJECT_SCALE_AUGMENTED = np.array([1.0, 1.0, 1.2])
_OBJECT_SCALE_NORMAL = np.array([1.0, 1.0, 1.0])
_AUGMENTATION_TRANSLATION = np.array([0.2, 0.0, 0.0])

# Toe-step cap floor (m/frame at 30 fps) and how far the source's own toe speed may be exceeded.
DEFAULT_TOE_STEP_CAP = 0.075
TOE_STEP_CAP_SOURCE_SCALE = 1.2


# Type aliases
TaskType = Literal["robot_only", "object_interaction", "climbing"]
# DataFormat is imported from config_types.data_type


# ----------------------------- Helper Functions -----------------------------


def create_task_constants(
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    task_config: TaskConfig,
    task_type: str,
) -> SimpleNamespace:
    """Create combined task constants from robot and motion data configs.

    Args:
        robot_config: Robot configuration
        motion_data_config: Motion data format configuration
        task_config: Task-specific configuration
        task_type: Type of task ("robot_only", "object_interaction", "climbing")

    Returns:
        SimpleNamespace with all task constants
    """
    task_constants = SimpleNamespace()

    # Copy all attributes from robot_config
    for attr in dir(robot_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(task_constants, attr, getattr(robot_config, attr))

    # Copy legacy motion data constants (upper-case for compatibility)
    for attr, value in motion_data_config.legacy_constants().items():
        setattr(task_constants, attr, value)

    # Task-specific object setup
    if task_type == "robot_only":
        obj_name = task_config.object_name or "ground"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = None
        task_constants.OBJECT_MESH_FILE = None
    elif task_type == "object_interaction":
        obj_name = task_config.object_name or "largebox"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = f"models/{obj_name}/{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = f"models/{obj_name}/{obj_name}.obj"
        task_constants.OBJECT_URDF_TEMPLATE = f"models/templates/{obj_name}.urdf.jinja"
    elif task_type == "climbing":
        obj_name = task_config.object_name or "multi_boxes"
        task_constants.OBJECT_NAME = obj_name
        object_dir = task_config.object_dir
        task_constants.OBJECT_DIR = str(object_dir) if object_dir else ""
        task_constants.OBJECT_URDF_FILE = str(object_dir / f"{obj_name}.urdf") if object_dir else f"{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = str(object_dir / f"{obj_name}.obj") if object_dir else f"{obj_name}.obj"
        task_constants.SCENE_XML_FILE = ""  # Will be set later

    return task_constants


def validate_config(cfg: RetargetingConfig) -> None:
    """Validate configuration consistency.

    Args:
        cfg: Configuration arguments

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate that data_format exists in registry (if provided)
    if cfg.data_format is not None and cfg.data_format not in DEMO_JOINTS_REGISTRY:
        available = ", ".join(sorted(DEMO_JOINTS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown data_format: '{cfg.data_format}'. "
            f"Available formats: {available}. "
            f"Add your format to DEMO_JOINTS_REGISTRY in config_types/data_type.py"
        )

    # Task-specific format requirements
    if cfg.task_type == "climbing" and cfg.data_format not in (None, "mocap", "g1fk"):
        raise ValueError("Climbing task requires 'mocap' data format")
    # smplx is the 22-joint body layout our OMOMO sources use; smplh assumes the 52-joint set
    if cfg.task_type == "object_interaction" and cfg.data_format not in (None, "smplh", "smplx"):
        raise ValueError("Object interaction requires 'smplh' data format")
    # robot_only accepts any format in the registry (already validated above)


def create_ground_points(x_range: tuple[float, float], y_range: tuple[float, float], size: int) -> np.ndarray:
    """Create ground point meshgrid.

    Args:
        x_range: (min, max) x-coordinate range
        y_range: (min, max) y-coordinate range
        size: Number of points per dimension

    Returns:
        (N, 3) array of ground points
    """
    x = np.linspace(x_range[0], x_range[1], size)
    y = np.linspace(y_range[0], y_range[1], size)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())], axis=1)


_root_quat_track = None


def load_motion_data(
    task_type: TaskType,
    data_format: str,
    data_path: Path,
    task_name: str,
    constants: SimpleNamespace,
    motion_data_config: MotionDataConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load motion data based on task type and format.

    Args:
        task_type: Type of task
        data_format: Data format ("lafan", "smplh", "mocap")
        data_path: Path to data directory
        task_name: Name of the task/sequence
        constants: Task constants
        motion_data_config: Motion data configuration

    Returns:
        Tuple of (human_joints, object_poses, smpl_scale)
        - human_joints: (T, J, 3) array of joint positions
        - object_poses: (T, 7) array of object poses [qw, qx, qy, qz, x, y, z]
        - smpl_scale: Scaling factor for SMPL compatibility

    Raises:
        FileNotFoundError: If required data files are not found
    """
    global _root_quat_track
    logger.info("Loading motion data for task: %s, format: %s", task_name, data_format)

    if task_type == "robot_only":
        if data_format == "lafan":
            npy_path = data_path / f"{task_name}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"LAFAN data file not found: {npy_path}")

            human_joints = np.load(str(npy_path))
            human_joints = transform_y_up_to_z_up(human_joints)
            spine_joint_idx = constants.DEMO_JOINTS.index(root_keypoint(constants.DEMO_JOINTS))
            # LAFAN-specific spine adjustment
            human_joints[:, spine_joint_idx, -1] -= 0.06
            smpl_scale = motion_data_config.default_scale_factor or 1.0
        elif data_format == "smplh":  # smplh
            pt_path = data_path / f"{task_name}.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"InterMimic data file not found: {pt_path}")

            human_joints, object_poses = load_intermimic_data(str(pt_path))
            smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)
        elif data_format == "mocap":
            downsample = 4
            npy_file = data_path / f"{task_name}.npy"
            if not npy_file.exists():
                raise FileNotFoundError(f"MOCAP data file not found: {npy_file}")

            human_joints = np.load(str(npy_file))[::downsample]

            default_human_height = motion_data_config.default_human_height or 1.78
            smpl_scale = constants.ROBOT_HEIGHT / default_human_height
        elif data_format == "smplx":
            npz_file = data_path / f"{task_name}.npz"

            human_data = np.load(str(npz_file))
            human_joints = human_data["global_joint_positions"]
            human_height = human_data["height"]
            smpl_scale = constants.ROBOT_HEIGHT / human_height
        else:
            # For other custom data format, if it uses consistent .npz file like SMPLX,
            # you can use the same logic as SMPLX.
            npz_file = data_path / f"{task_name}.npz"

            human_data = np.load(str(npz_file))
            human_joints = human_data["global_joint_positions"]
            human_height = human_data["height"]
            smpl_scale = constants.ROBOT_HEIGHT / human_height

        # Create dummy object poses for robot_only
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))

    elif task_type == "object_interaction":
        npz_path = data_path / f"{task_name}.npz"
        pt_path = data_path / f"{task_name}.pt"
        if npz_path.exists():
            # OMOMO: the source npz carries the object pose track alongside the human joints, already in
            # the same frame (see hcrl/omomo_objects.py).
            human_data = np.load(str(npz_path), allow_pickle=True)
            human_joints = human_data["global_joint_positions"]
            object_poses = human_data["object_poses"]
            smpl_scale = constants.ROBOT_HEIGHT / float(human_data["height"])
            if "root_quat" in human_data.files:
                _root_quat_track = np.asarray(human_data["root_quat"], dtype=float)
        elif pt_path.exists():
            human_joints, object_poses = load_intermimic_data(str(pt_path))
            smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)
        else:
            raise FileNotFoundError(f"No object-interaction data for {task_name} at {npz_path} or {pt_path}")

    elif task_type == "climbing":
        task_dir = data_path / task_name
        exact = task_dir / f"{task_name}.npy"
        npy_files = [exact] if exact.exists() else [f for f in task_dir.glob("*.npy") if not f.stem.endswith("_q0")]
        if not npy_files:
            raise FileNotFoundError(f"No .npy file found in {task_dir}")

        npy_file = npy_files[0]
        # MOCAP-specific downsample factor
        downsample = 4
        human_joints = np.load(str(npy_file))[::downsample]
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))
        default_human_height = motion_data_config.default_human_height or 1.78
        smpl_scale = constants.ROBOT_HEIGHT / default_human_height

    logger.debug(
        "Loaded %d frames, scale factor: %.4f",
        human_joints.shape[0],
        smpl_scale,
    )
    return human_joints, object_poses, smpl_scale


def setup_object_data(
    task_type: TaskType,
    constants: SimpleNamespace,
    object_dir: Path | None,
    smpl_scale: float,
    task_config: TaskConfig,
    augmentation: bool,
    object_scale_augmented: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Setup object-specific data (ground, object mesh, climbing terrain).
    Args:
        task_type: Type of task
        constants: Task constants
        object_dir: Object directory path (for climbing)
        smpl_scale: SMPL scaling factor
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        object_scale_augmented: Scale factor for augmented objects (default: [1.0, 1.0, 1.2])
    Returns:
        Tuple of (object_local_pts, object_local_pts_demo, object_urdf_path)
    """
    object_scale_normal = np.array([1.0, 1.0, 1.0])
    if object_scale_augmented is None:
        object_scale_augmented = np.array([1.0, 1.0, 1.2])  # For climbing task augmentation
    logger.info("Setting up object data for task: %s", task_type)

    if task_type == "robot_only":
        # Create ground points meshgrid
        ground_pts = create_ground_points(task_config.ground_range, task_config.ground_range, task_config.ground_size)
        return ground_pts, ground_pts, None

    if task_type == "object_interaction":
        # Load object data
        if constants.OBJECT_MESH_FILE is None:
            raise ValueError("OBJECT_MESH_FILE not set for object_interaction task")

        object_local_pts, object_local_pts_demo = load_object_data(
            constants.OBJECT_MESH_FILE, smpl_scale=smpl_scale, sample_count=100
        )
        return object_local_pts, object_local_pts_demo, constants.OBJECT_URDF_FILE

    if task_type == "climbing":
        if object_dir is None:
            raise ValueError("object_dir must be provided for climbing task")

        # Setup climbing-specific object
        box_asset_xml = object_dir / "box_assets.xml"
        scene_xml_name = Path(constants.ROBOT_URDF_FILE).name.replace(".urdf", f"_w_{constants.OBJECT_NAME}.xml")
        scene_xml_file = object_dir / scene_xml_name
        # Set SCENE_XML_FILE in constants BEFORE creating retargeter (needed for temp_retargeter)
        constants.SCENE_XML_FILE = str(scene_xml_file)

        np.random.seed(0)
        print("object mesh file: ", constants.OBJECT_MESH_FILE)
        object_local_pts, object_local_pts_demo_original = load_object_data(
            constants.OBJECT_MESH_FILE,
            smpl_scale=smpl_scale,
            surface_weights=lambda p: (
                task_config.surface_weight_high
                if p[2] > task_config.surface_weight_threshold
                else task_config.surface_weight_low
            ),
            sample_count=100,
        )

        if augmentation:
            ground_pts = create_ground_points(
                task_config.climbing_ground_range, task_config.climbing_ground_range, task_config.climbing_ground_size
            )
            object_local_pts_demo = np.concatenate([object_local_pts_demo_original, ground_pts], axis=0)
            object_scale = object_scale_augmented
            object_local_pts = object_scale * object_local_pts_demo
        else:
            object_scale = object_scale_normal
            object_local_pts_demo = object_local_pts_demo_original
            object_local_pts = object_local_pts_demo

        # Create scaled URDF and XML files
        scale_factors = tuple(float(value) for value in (object_scale * smpl_scale))
        object_urdf_file = create_scaled_multi_boxes_urdf(constants.OBJECT_URDF_FILE, scale_factors)
        object_asset_xml_path = create_scaled_multi_boxes_xml(str(box_asset_xml), scale_factors)
        new_scene_xml_path = create_new_scene_xml_file(str(scene_xml_file), scale_factors, object_asset_xml_path)
        constants.SCENE_XML_FILE = new_scene_xml_path

        return object_local_pts, object_local_pts_demo, object_urdf_file

    raise ValueError(f"Unknown task type: {task_type}")


def _compute_q_init_base(
    task_type: TaskType,
    data_format: str,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter | None = None,
) -> np.ndarray:
    """Compute base robot pose initialization (q_init_base).
    This is a shared helper function used by both single and parallel processing.
    Args:
        task_type: Type of task
        data_format: Data format
        human_joints: Human joint positions
        object_poses: Object poses in format [qw, qx, qy, qz, x, y, z]
        constants: Task constants
        retargeter: Optional retargeter instance (needed for climbing)
    Returns:
        q_init_base in MuJoCo order: [0:3] position, [3:7] quaternion, [7:] joints
    """
    if task_type == "robot_only":
        if data_format == "lafan":
            spine_joint_idx = constants.DEMO_JOINTS.index(root_keypoint(constants.DEMO_JOINTS))
            human_quat_init = estimate_human_orientation(human_joints, constants.DEMO_JOINTS)
            # MuJoCo order: pos first, then quat
            q_init_base = np.concatenate(
                [human_joints[0, spine_joint_idx, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)]
            )
        else:  # smplh
            _, human_quat_init = transform_from_human_to_world(
                human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
            )
            # MuJoCo order: pos first, then quat
            q_init_base = np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])
    elif task_type == "object_interaction":
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        # MuJoCo order: pos first, then quat
        q_init_base = np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])
    elif task_type == "climbing":
        if retargeter is None:
            raise ValueError("retargeter is required for climbing task")
        if data_format == "g1fk":
            # robot-FK pseudo-source: the SOURCE csv carries the full initial pose -- init from it (the
            # identity-retarget start; joints-at-zero sinks feet into thin terrain like the edge beam)
            seq_dir = Path(constants.OBJECT_DIR)
            q0_path = seq_dir / f"{seq_dir.name}_q0.npy"
            if q0_path.exists():
                return np.load(q0_path)
            human_quat_init = estimate_human_orientation(human_joints, retargeter.demo_joints)
        else:
            _, human_quat_init = transform_from_human_to_world(
                human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
            )
        spine_joint_idx = retargeter.demo_joints.index(root_keypoint(retargeter.demo_joints))
        # MuJoCo order: pos first, then quat
        q_init_base = np.concatenate(
            [
                human_joints[0, spine_joint_idx],
                human_quat_init,
                np.zeros(constants.ROBOT_DOF),
            ]
        )
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    return q_init_base


def convert_object_poses_to_mujoco_order(object_poses: np.ndarray) -> np.ndarray:
    """Convert object poses from [qw, qx, qy, qz, x, y, z] to MuJoCo order [x, y, z, qw, qx, qy, qz].
    Args:
        object_poses: Object poses array of shape (T, 7) in format [qw, qx, qy, qz, x, y, z]
    Returns:
        Object poses array in MuJoCo order [x, y, z, qw, qx, qy, qz]
    """
    return object_poses[:, [4, 5, 6, 0, 1, 2, 3]]


def build_retargeter_kwargs_from_config(
    retargeter_config: RetargeterConfig,
    constants: SimpleNamespace,
    object_urdf_path: str | None,
    task_type: str,
) -> dict:
    """Build kwargs for InteractionMeshRetargeter from a RetargeterConfig.
    This is a convenience function that allows building kwargs directly from
    a RetargeterConfig without needing a full RetargetingConfig.
    Args:
        retargeter_config: Retargeter configuration
        constants: Task constants
        object_urdf_path: Path to object URDF file
        task_type: Type of task
    Returns:
        Dictionary of kwargs for InteractionMeshRetargeter
    """
    kwargs = {
        "task_constants": constants,
        "object_urdf_path": object_urdf_path,
        "q_a_init_idx": retargeter_config.q_a_init_idx,
        "activate_joint_limits": retargeter_config.activate_joint_limits,
        "activate_obj_non_penetration": retargeter_config.activate_obj_non_penetration,
        "activate_foot_sticking": retargeter_config.activate_foot_sticking,
        "foot_lock": retargeter_config.foot_lock,
        "penetration_tolerance": retargeter_config.penetration_tolerance,
        "foot_sticking_tolerance": retargeter_config.foot_sticking_tolerance,
        "self_collision": retargeter_config.self_collision,
        "step_size": retargeter_config.step_size,
        "visualize": retargeter_config.visualize,
        "debug": retargeter_config.debug,
        "w_nominal_tracking_init": retargeter_config.w_nominal_tracking_init,
    }
    if task_type == "climbing":
        kwargs["nominal_tracking_tau"] = retargeter_config.nominal_tracking_tau
    return kwargs


def initialize_robot_pose(
    task_type: TaskType,
    data_format: str,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter,
    task_config: TaskConfig,
    augmentation: bool,
    save_dir: Path,
    task_name: str,
    augmentation_translation: np.ndarray | None = None,
    augmentation_rotation: float | None = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize robot pose (q_init, q_nominal) based on task.
    Returns qpos in MuJoCo order: [0:3] position, [3:7] quaternion, [7:] joints.
    Object poses are returned in MuJoCo order: [0:3] position, [3:7] quaternion.
    Args:
        task_type: Type of task
        data_format: Data format
        human_joints: Human joint positions
        object_poses: Object poses (assumed to be in format: [quat, pos] or [pos, quat])
        constants: Task constants
        retargeter: Retargeter instance
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        save_dir: Save directory path
        task_name: Task name
        augmentation_translation: Translation vector for augmentation (default: [0.2, 0.0, 0.0])
    Returns:
        Tuple of (q_init, q_nominal, object_poses_augmented, human_joints_modified, object_poses_modified)
        where qpos is in MuJoCo order and object_poses are in MuJoCo order
    """
    # Use default if not provided
    if augmentation_translation is None:
        augmentation_translation = _AUGMENTATION_TRANSLATION
    logger.info("Initializing robot pose")

    if task_type == "robot_only":
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants)
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    if task_type == "object_interaction":
        if augmentation:
            object_moving_frame_idx = extract_object_first_moving_frame(object_poses)
            object_poses_augmented = augment_object_poses(
                object_poses,
                object_moving_frame_idx,
                human_joints[0, 0, :],
                augmentation_translation,
                augmentation_rotation,
            )
            # Convert object_poses to MuJoCo order
            object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
            object_poses = convert_object_poses_to_mujoco_order(object_poses)

            original_path = save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_path}. Run without --augmentation first.")

            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            return q_nominal[0], q_nominal, object_poses_augmented, human_joints, object_poses
        object_poses_augmented = object_poses.copy()
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants)
        # Convert object_poses to MuJoCo order
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
        return q_init, None, object_poses_augmented, human_joints, object_poses

    if task_type == "climbing":
        if augmentation:
            original_path = save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_path}. Run without --augmentation first.")

            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            # Convert object_poses to MuJoCo order
            object_poses = convert_object_poses_to_mujoco_order(object_poses)
            return q_nominal[0], q_nominal, object_poses, human_joints, object_poses
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants, retargeter)
        # Convert object_poses to MuJoCo order
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    raise ValueError(f"Unknown task type: {task_type}")


def determine_output_path(
    task_type: TaskType,
    save_dir: Path,
    task_name: str,
    augmentation: bool,
) -> str:
    """Determine output file path based on task and augmentation.
    Args:
        task_type: Type of task
        save_dir: Save directory path
        task_name: Task name
        augmentation: Whether this is an augmentation run
    Returns:
        Output file path
    """
    if task_type == "robot_only":
        return str(save_dir / f"{task_name}.npz")
    if task_type in ("object_interaction", "climbing"):
        suffix = "_augmented" if augmentation else "_original"
        return str(save_dir / f"{task_name}{suffix}.npz")
    raise ValueError(f"Unknown task type: {task_type}")


# ----------------------------- Main -----------------------------


def main(cfg: RetargetingConfig) -> None:
    """Main retargeting pipeline.
    Args:
        cfg: Configuration arguments
    """
    # Validate configuration
    validate_config(cfg)

    robot = cfg.robot
    task_name = cfg.task_name
    task_type = cfg.task_type

    # Set defaults based on task type
    data_format: str = cfg.data_format or DEFAULT_DATA_FORMATS[task_type]
    save_dir = cfg.save_dir if cfg.save_dir is not None else Path(DEFAULT_SAVE_DIRS[task_type].format(robot=robot))
    data_path = cfg.data_path

    os.makedirs(save_dir, exist_ok=True)
    logger.info("Task: %s, Type: %s, Format: %s", task_name, task_type, data_format)
    logger.info("Data path: %s, Save dir: %s", data_path, save_dir)

    # Ensure configs match top-level selections
    if cfg.robot_config.robot_type != robot:
        cfg.robot_config = RobotConfig(robot_type=robot)

    if cfg.motion_data_config.robot_type != robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = MotionDataConfig(data_format=data_format, robot_type=robot)

    # Task-specific object setup: set default object_dir for climbing if not provided
    if task_type == "climbing" and cfg.task_config.object_dir is None:
        from dataclasses import replace

        cfg.task_config = replace(cfg.task_config, object_dir=data_path / task_name)

    constants = create_task_constants(
        robot_config=cfg.robot_config,
        motion_data_config=cfg.motion_data_config,
        task_config=cfg.task_config,
        task_type=task_type,
    )

    # Load motion data
    human_joints, object_poses, smpl_scale = load_motion_data(
        task_type, data_format, data_path, task_name, constants, cfg.motion_data_config
    )

    # Get toe names from motion data config (depends only on data_format)
    toe_names = cfg.motion_data_config.toe_names

    # Setup object data
    object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
        task_type,
        constants,
        cfg.task_config.object_dir,
        smpl_scale,
        cfg.task_config,
        cfg.augmentation,
        object_scale_augmented=_OBJECT_SCALE_AUGMENTED,
    )

    # Create retargeter
    retargeter_kwargs = build_retargeter_kwargs_from_config(cfg.retargeter, constants, object_urdf_path, task_type)
    # hcrl: per-window foot z-lock from precomputed stance windows (see hcrl/stance_windows.py) --
    # xy sticking alone stops skate but nothing pulls a hovering source foot DOWN to the surface.
    _stick_path = data_path / task_name / f"{task_name}_foot_sticking.npz" if task_type == "climbing" else None
    if _stick_path is not None and _stick_path.exists():
        _sw = np.load(_stick_path, allow_pickle=True)
        if "windows_left" in _sw and "windows_right" in _sw:
            from holosoma_retargeting.config_types.retargeter import FootLockConfig

            # the anchored link must be a constrainable foot link; add the toe spheres (the stance
            # detector's keypoint) to the sticking-link set the Jacobians are built for
            _tc = retargeter_kwargs["task_constants"]
            for _toe in ("left_ankle_roll_sphere_5_link", "right_ankle_roll_sphere_5_link"):
                if _toe not in _tc.FOOT_STICKING_LINKS:
                    _tc.FOOT_STICKING_LINKS = [*_tc.FOOT_STICKING_LINKS, _toe]

            retargeter_kwargs["foot_lock"] = FootLockConfig(
                enable=True,
                windows={
                    "left": [tuple(w) for w in _sw["windows_left"]],
                    "right": [tuple(w) for w in _sw["windows_right"]],
                },
                tolerance=0.01,
                lock_links_substr="sphere_5",  # the toe sphere: one 3D-anchored point per foot, ankle roll stays free
            )
            logger.info("Foot z-lock windows: L %d / R %d", len(_sw["windows_left"]), len(_sw["windows_right"]))

    retargeter = InteractionMeshRetargeter(**retargeter_kwargs)
    # env var so the batch driver can toggle it without threading a flag through the cfg
    retargeter.limb_retarget = bool(getattr(cfg, "limb_retarget", False)) or os.environ.get(
        "HOLOSOMA_LIMB_RETARGET", ""
    ).lower() in ("1", "true", "yes")
    logger.info("Retargeter created")

    # hcrl: anti-oscillation damping when stance windows are active -- with the toe anchored and
    # sole-sphere XY stuck, the lateral-lean null space is near-tied and flips at 15 Hz (ankle-roll
    # rocking, mm-scale root wobble). Ankle roll gets velocity damping (stance roll velocity ~ 0);
    # everything else gets acceleration damping, which is free on smooth motion so it cannot drag.
    # hcrl: acceleration damping is gated on foot_lock above, which only ever fires for climbing clips,
    # so ordinary retargets ran with accel_damp_weight=0 and snapped frame to frame. The term costs
    # nothing at constant velocity, so applying it generally smooths the solve without dragging motion.
    # hcrl: the root's quaternion rows (qpos 3..6) get the same scalar smoothing as a knee, so the
    # torso can swing frame to frame while the joints look smooth. Boost just those rows.
    _rootw = _envf("HCRL_ROOT_SMOOTH_W", 8.0)
    if _rootw > 0 and retargeter.q_a_init_idx == -7 and np.isscalar(retargeter.smooth_weight):
        _sv = np.full(retargeter.nq_a, float(retargeter.smooth_weight))
        _sv[3:7] = _rootw
        retargeter.smooth_weight = _sv
        logger.info("Root-orientation smoothing weight: %.1f (joints %.2f)", _rootw, float(_sv[7]))

    _rr = _envf("HCRL_ROOT_RATE_W", 0.0)  # measured: no improvement, off by default
    if _rr > 0 and _root_quat_track is not None:
        retargeter.root_rate_weight = _rr
        retargeter.root_quat_track = _root_quat_track
        logger.info("Root angular-rate prior: w=%.1f over %d frames", _rr, len(_root_quat_track))

    _jaw = _envf("HCRL_JOINT_ANGLE_W", 5.0)
    if _jaw > 0:
        from holosoma_retargeting.hcrl.source_angles import t1_joint_angle_targets

        retargeter.joint_angle_weight = _jaw
        retargeter.joint_angle_targets = t1_joint_angle_targets(human_joints)
        _m = {k: float(np.abs(v).mean()) for k, v in retargeter.joint_angle_targets.items()}
        logger.info("Joint-angle tracking w=%.1f, source |angle| means: %s", _jaw,
                    {k: round(v, 2) for k, v in _m.items()})

    # debug: record the mapped source points the solver actually optimizes against, so an overlay
    # shows the real targets rather than a reconstruction of them
    _dump = os.environ.get("HCRL_DUMP_TARGETS", "")
    if _dump:
        retargeter._dump_targets = []
        logger.info("Dumping solver targets to %s", _dump)

    # convergence knobs: is the residual under-convergence (helped by more/larger steps) or the cost's
    # own optimum (unchanged by them, meaning the term weights are what to fix)?
    _ni = int(_envf("HCRL_N_ITER", 0))
    if _ni > 0:
        retargeter.solve_n_iter = _ni
    _ss = _envf("HCRL_STEP_SIZE", 0.0)
    if _ss > 0:
        retargeter.step_size = _ss
    logger.info("Solve: n_iter=%s step_size=%.2f", _ni or "default", retargeter.step_size)

    retargeter.keypoint_track_weight = _envf("HCRL_KP_W", 0.0)
    _lw = os.environ.get("HCRL_LAP_W", "")
    if _lw != "":
        retargeter.laplacian_weights = float(_lw)
    retargeter.debug_terms = os.environ.get("HCRL_DEBUG_TERMS", "") in ("1", "true")

    _ad = os.environ.get("HOLOSOMA_ACCEL_DAMP", "3.0")
    _sw_env = os.environ.get("HOLOSOMA_SMOOTH_WEIGHT", "")
    if not retargeter.foot_lock.enable and float(_ad) > 0:
        retargeter.accel_damp_weight = float(_ad)
        if _sw_env:
            retargeter.smooth_weight = float(_sw_env)
        logger.info(
            "Temporal smoothing: accel_damp=%.2f smooth=%s", retargeter.accel_damp_weight, retargeter.smooth_weight
        )

    # hcrl: the redundancy priors below are NOT foot-lock specific -- they were gated behind it, so only
    # climbing clips ever got them and every other retarget rode its joint stops with the arm and pelvis
    # parked wherever the null space landed. Applied generally now, still env-overridable.
    retargeter.joint_limit_barrier_weight = _envf("HCRL_JL_W", 50.0)
    retargeter.joint_limit_barrier_margin = _envf("HCRL_JL_MARGIN", 0.10)
    retargeter.joint_limit_barrier_margin_frac = _envf("HCRL_JL_MARGIN_FRAC", 0.15)
    _jl_joints = os.environ.get("HCRL_JL_JOINTS", "").strip()
    retargeter.joint_limit_barrier_joints = tuple(_jl_joints.split(",")) if _jl_joints else None
    retargeter.pelvis_track_weight = _envf("HCRL_PELVIS_W", 5.0)
    retargeter.arm_reg_weight = _envf("HCRL_ARM_W", 2.0)
    retargeter.swing_ankle_weight = _envf("HCRL_SWING_ANKLE_W", 0.5)
    logger.info(
        "hcrl priors: jl_w=%.1f margin=%.3f/%.2f pelvis=%.1f arm=%.1f swing_ankle=%.2f",
        retargeter.joint_limit_barrier_weight,
        retargeter.joint_limit_barrier_margin,
        retargeter.joint_limit_barrier_margin_frac,
        retargeter.pelvis_track_weight,
        retargeter.arm_reg_weight,
        retargeter.swing_ankle_weight,
    )

    if retargeter.foot_lock.enable and retargeter.q_a_init_idx == -7:
        _w = np.full(retargeter.nq_a, float(retargeter.smooth_weight))
        for _jn in ("left_ankle_roll_joint", "right_ankle_roll_joint"):
            _w[retargeter.robot_model.jnt_qposadr[retargeter.robot_model.joint(_jn).id]] = 3.0
        retargeter.smooth_weight = _w
        # 1.0 is the sweet spot: 4.0 carries momentum through landings (foot/root overshoot then correct)
        retargeter.accel_damp_weight = 1.0
        # stance foot-orientation engagement: yaw frozen at entry + terrain-flat pitch/roll, ramped +
        # slew-limited -- position pins alone snap the ankle attitude the frame a window binds
        retargeter.foot_orient_weight = 30.0
        # hcrl v3 (terrain-bfm §4.6): the G1's ROM is narrower than the human's and nothing penalized
        # riding a stop, so the solve parked joints on their limits (waist_pitch 54% of frames,
        # ankle_roll 35% in `edge`). Barrier keeps a margin; the pelvis/arm priors remove the null
        # spaces that made the solver WANT the stop in the first place. Env-overridable so the weight
        # ablation sweeps without editing code (HCRL_JL_W=0 HCRL_PELVIS_W=0 ... == the v1 corpus).
        logger.info(
            "hcrl v3 priors: jl_w=%.1f margin=%.3f/%.2f pelvis=%.1f arm=%.1f swing_ankle=%.2f",
            retargeter.joint_limit_barrier_weight,
            retargeter.joint_limit_barrier_margin,
            retargeter.joint_limit_barrier_margin_frac,
            retargeter.pelvis_track_weight,
            retargeter.arm_reg_weight,
            retargeter.swing_ankle_weight,
        )

    # Preprocess motion data
    if task_type == "robot_only":
        human_joints = preprocess_motion_data(human_joints, retargeter, toe_names, smpl_scale)

    # hcrl: ball-contact radius constraint (Soccer-X). Needs the splined ball sidecar next to the source
    # npz; transforms it into the SOLVE frame by recovering the z-shift empirically from the processed
    # joints, then detects contact segments on the toe keypoints with hysteresis.
    if os.environ.get("HCRL_BALL_CONSTRAINT", "") in ("1", "true"):
        _bf = data_path / f"{cfg.task_name}.ball_smooth.npz"
        if not _bf.exists():
            _bf = data_path / f"{cfg.task_name}.ball.npz"
        if _bf.exists():
            _raw = np.load(str(data_path / f"{cfg.task_name}.npz"))
            _raw_j = _raw["global_joint_positions"].astype(float)
            _zshift = float(_raw_j[0, 0, 2] - human_joints[0, 0, 2] / smpl_scale)
            _ball = np.load(str(_bf))["soccer_pos"].astype(float)
            _ball[:, 2] -= _zshift
            _ball = _ball * smpl_scale
            _n = min(len(human_joints), len(_ball))
            _segs = []
            # J_OC_dict in the solver is keyed by SOURCE keypoint names, not robot links
            for _side, _toe_j, _toe_link in (("L", 10, "L_Foot"), ("R", 11, "R_Foot")):
                _d = np.linalg.norm(human_joints[:_n, _toe_j] - _ball[:_n], axis=1)
                _enter, _exit = 0.16 * smpl_scale, 0.22 * smpl_scale
                _in, _s0, _r0 = False, 0, 0.0
                for _t in range(_n):
                    if not _in and _d[_t] < _enter:
                        _in, _s0, _r0 = True, _t, float(_d[_t])
                    elif _in and (_d[_t] > _exit or _t == _n - 1):
                        _segs.append((_toe_link, _s0, _t if _d[_t] > _exit else _t + 1, _r0))
                        _in = False
            retargeter.ball_track = _ball
            retargeter.ball_contacts = tuple(_segs)
            logger.info("Ball constraint: %d segment(s): %s", len(_segs),
                        [(l.split("_")[0], a, b, round(r, 3)) for l, a, b, r in _segs])
    elif task_type in {"object_interaction", "climbing"}:
        human_joints, object_poses, object_moving_frame_idx = preprocess_motion_data(
            human_joints,
            retargeter,
            toe_names,
            scale=smpl_scale,
            object_poses=object_poses,
        )

    # Initialize robot pose
    q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
        task_type,
        data_format,
        human_joints,
        object_poses,
        constants,
        retargeter,
        cfg.task_config,
        cfg.augmentation,
        save_dir,
        task_name,
        augmentation_translation=_AUGMENTATION_TRANSLATION,
    )

    # Extract foot sticking sequences. A precomputed override (hcrl/stance_windows.py: position+height
    # rule against the terrain) takes priority -- the velocity heuristic finds nothing on IK-retargeted
    # sources whose stance feet skate faster than 1 cm/s.
    # Per-clip toe-step cap from the SOURCE: motion that is genuinely fast (a kick, a jump) must not be
    # flattened by a fixed cap, while anything faster than the source itself is a teleport.
    _toe_idx = [retargeter.demo_joints.index(t) for t in toe_names]
    _steps = np.zeros((len(human_joints), 2))
    _steps[1:] = np.linalg.norm(np.diff(human_joints[:, _toe_idx], axis=0), axis=2)
    _steps = np.maximum(_steps, np.roll(_steps, 1, axis=0))  # 2-frame max: tolerate phase offsets
    toe_step_cap = np.maximum(DEFAULT_TOE_STEP_CAP, TOE_STEP_CAP_SOURCE_SCALE * _steps)

    sticking_file = data_path / task_name / f"{task_name}_foot_sticking.npz" if task_type == "climbing" else None
    if sticking_file is not None and sticking_file.exists():
        _stick = np.load(sticking_file, allow_pickle=True)
        _mask = _stick["sticking"]  # (T, 2) bool for [left, right] toes at solver rate
        _toes = [str(t) for t in _stick["toe_names"]]
        foot_sticking_sequences = [
            {_toes[0]: bool(_mask[min(t, len(_mask) - 1), 0]), _toes[1]: bool(_mask[min(t, len(_mask) - 1), 1])}
            for t in range(len(human_joints))
        ]
        logger.info("Loaded foot sticking override: %s (L %.0f%% / R %.0f%%)",
                    sticking_file.name, 100 * _mask[:, 0].mean(), 100 * _mask[:, 1].mean())

        # Stance frames get the tight default cap: a planted foot has no licence to move fast.
        for _side, _k in (("windows_left", 0), ("windows_right", 1)):
            for _win in _stick[_side]:
                _s0, _s1 = max(int(_win[0]) - 1, 0), min(int(_win[1]) + 1, len(toe_step_cap) - 1)
                toe_step_cap[_s0 : _s1 + 1, _k] = DEFAULT_TOE_STEP_CAP
    else:
        foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
            human_joints, retargeter.demo_joints, toe_names
        )

    retargeter.foot_step_max_seq = toe_step_cap
    logger.info("Toe-step cap: default %.3f, per-clip max L %.3f / R %.3f m/frame",
                DEFAULT_TOE_STEP_CAP, toe_step_cap[:, 0].max(), toe_step_cap[:, 1].max())

    # Source sole planes, when the format supplies them: the joint mapping pins only an ankle and a
    # toe per foot, which leaves pitch/roll free and lets the sole settle toe-down. Scaling and
    # translation upstream preserve normal directions, so these need no transform.
    sole_normal_file = data_path / f"{task_name}.npz"
    if sole_normal_file.exists():
        with np.load(str(sole_normal_file)) as source_npz:
            sole_normal = source_npz.get("sole_normal")
            source_npz_height = source_npz.get("sole_height")
        if sole_normal is not None:
            retargeter.sole_normal_seq = sole_normal
            retargeter.sole_normal_weight = _envf("HCRL_SOLE_W", 5.0)
            # heights are source-scale like the keypoints, so they need the same smpl_scale; the
            # clamp keeps a noisy source frame from ever commanding the sole below the floor
            retargeter.sole_height_seq = np.maximum(source_npz_height * smpl_scale, 0.0)
            retargeter.sole_height_weight = _envf("HCRL_SOLE_Z_W", 2000.0)
            tilt = np.degrees(np.arccos(np.clip(sole_normal[..., 2], -1.0, 1.0)))
            logger.info("Sole-orientation matching: weight %.1f, source tilt median %.1f deg",
                        retargeter.sole_normal_weight, float(np.median(tilt)))

    # Task-specific foot sticking adjustments
    if task_type == "object_interaction":
        # Disable initial sticking
        foot_sticking_sequences[0][toe_names[0]] = False
        foot_sticking_sequences[0][toe_names[1]] = False

    # Determine output path
    dest_res_path = determine_output_path(task_type, save_dir, task_name, cfg.augmentation)

    # Retarget motion
    logger.info("Starting retargeting...")
    retargeter.retarget_motion(
        human_joint_motions=human_joints,
        object_poses=object_poses,
        object_poses_augmented=object_poses_augmented,
        object_points_local_demo=object_local_pts_demo,
        object_points_local=object_local_pts,
        foot_sticking_sequences=foot_sticking_sequences,
        q_a_init=q_init,
        q_nominal_list=q_nominal,
        original=not cfg.augmentation,
        dest_res_path=dest_res_path,
    )
    if os.environ.get("HCRL_DUMP_TARGETS", ""):
        np.save(os.environ["HCRL_DUMP_TARGETS"], np.asarray(retargeter._dump_targets))
        logger.info("Wrote %d target frames", len(retargeter._dump_targets))
    logger.info("Retargeting complete. Results saved to: %s", dest_res_path)

    if cfg.retargeter.debug:
        input("Press Enter to exit ...")


if __name__ == "__main__":
    cfg = tyro.cli(RetargetingConfig)
    main(cfg)
