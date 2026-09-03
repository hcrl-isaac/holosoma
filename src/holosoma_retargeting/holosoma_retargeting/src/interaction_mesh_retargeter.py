from __future__ import annotations

import sys
import time
from pathlib import Path
from types import ModuleType

import cvxpy as cp  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from scipy import sparse as sp  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from tqdm import tqdm
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from holosoma_retargeting.config_types.data_type import root_keypoint
from holosoma_retargeting.config_types.retargeter import FootLockConfig, SelfCollisionConfig

# Substrings identifying arm keypoints in a joint mapping, across source formats.
_ARM_KEYPOINT_PARTS = ("shoulder", "elbow", "wrist", "hand")

# Foot points pushed out of the ball per side per frame. The deepest one drives the correction; the
# rest pin the rotation the rigid foot would otherwise use to dodge it.
BALL_CONTACT_POINTS = 6

# Add src to path for direct execution
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import with type ignore for mypy compatibility
from mujoco_utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    _world_mesh_from_geom,
)
from utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_interaction_mesh,
    get_adjacency_list,
    transform_points_local_to_world,
    transform_points_world_to_local,
)
from viser_utils import create_motion_control_sliders  # type: ignore[import-not-found,no-redef]  # noqa: E402


class InteractionMeshRetargeter:
    """
    A class to perform kinematic retargeting from human motion to a robot,
    preserving spatial relationships using an interaction mesh.
    """

    def __init__(
        self,
        task_constants: ModuleType,
        object_urdf_path: str,
        q_a_init_idx: int = -7,
        activate_foot_sticking: bool = True,
        activate_obj_non_penetration: bool = True,
        activate_joint_limits: bool = True,
        step_size: float = 0.2,
        collision_detection_threshold: float = 0.1,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        foot_lock: FootLockConfig | None = None,
        self_collision: SelfCollisionConfig | None = None,
        visualize: bool = False,
        debug: bool = False,
        w_nominal_tracking_init: float = 5.0,
        nominal_tracking_tau: float = 10.0,
    ):
        """This kinematic retargeter solves the diffIK problem with hard constraints in SQP style.
        During each SQP iteration, the problem is solved with the following constraints and costs:
            1. [Cost] Minimize the Laplacian deformation in the object frame.
            2. [Constraint] Enforce the non-penetration constraints w/ the ground and (if activated) the object.
            3. [Constraint] Enforce the foot sticking constraints if activated.
            4. [Constraint] Enforce the joint limits if activated.
            5. [Constraint] Enforce trust region of dq.
        The constraints are linearized and the costs are quadratic with a trust region.

        Args:
            q_a_init_idx: the index in robot's configuration where the optimization variables start. -7: starts from the
            floating base, -3: starts from the translation of the floating base, 0: starts from the actuated DOF,
            12: starts from waist, 15: starts from left shoulder
            step_size: trust region for each SQP iteration.
            collision_detection_threshold: only start to detect collision
            when the distance is smaller than this threshold.
            penetration_tolerance: tolerance for penetration when enforcing non-penetration constraints.
            foot_sticking_tolerance: tolerance for foot sticking constraints in x, y.
            foot_lock: configuration for explicit frame-range based foot locking constraints.
            nominal_tracking_tau: the time constant for the nominal tracking cost.
        """

        self.robot_model_path = task_constants.ROBOT_URDF_FILE
        self.object_model_path = object_urdf_path
        self.object_name = task_constants.OBJECT_NAME
        self.collision_detection_threshold = collision_detection_threshold
        self.activate_foot_sticking = activate_foot_sticking
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.activate_joint_limits = activate_joint_limits
        self.foot_links = dict(zip(task_constants.FOOT_STICKING_LINKS, task_constants.FOOT_STICKING_LINKS))
        self.penetration_tolerance = penetration_tolerance
        self.step_size = step_size
        self.visualize = visualize
        self.debug = debug
        self.demo_joints = task_constants.DEMO_JOINTS
        self.laplacian_match_links = task_constants.JOINTS_MAPPING
        self.task_constants = task_constants

        self.smplh_mapped_joint_indices = [self.demo_joints.index(name) for name in self.laplacian_match_links]

        # Setup weights and parameters
        self.laplacian_weights = 10
        self.smooth_weight = 0.2
        self.accel_damp_weight = 0.0  # acceleration damping: zero-cost at constant velocity (anti-oscillation)
        self.foot_step_max_seq = None  # optional (T, 2) [left, right] per-frame toe-step caps (flight phases)
        # Teleporting feet are excluded by bounding Cartesian toe speed, which depends on neither
        # terrain nor contact detection -- so it is not tied to foot_lock.
        self.teleport_guard = True
        # Source sole-plane normals (T, 2, 3) for [left, right]; without them nothing pins foot
        # pitch/roll, because the joint mapping gives each foot only an ankle and a toe point.
        self.sole_normal_seq = None
        self.sole_normal_weight = 0.0
        # Source sole ground heights (T, 2); flattening alone lifts the sole, so stance also needs a height.
        self.sole_height_seq = None
        self.sole_height_weight = 0.0
        self.sole_planted_height = 0.03
        self._sole_body_id_cache: dict[str, list[int]] = {}
        # Solver-frame centres (T, 3) of an object that does NOT scale with the human, with the foot
        # surface points to keep out of it; NaN rows mean "no object this frame".
        self.ball_seq = None
        self.ball_clearance_seq = None  # (T, 2) [left, right] clearance each foot must keep from it
        self.ball_foot_points = None
        self.ball_radius = 0.0
        self.ball_weight = 0.0
        self.foot_orient_weight = 0.0  # stance-engagement foot angular-rate damping (0 = off; _foot_orient_damp)
        self.joint_limit_barrier_weight = 0.0  # one-sided hinge inside `margin` of an actuated stop
        self.joint_limit_barrier_margin = 0.0  # rad, absolute cap; 0 disables the barrier
        self.joint_limit_barrier_margin_frac = 0.15  # and never more than this fraction of the range
        self.joint_limit_barrier_min_range = 0.15  # rad; below this the joint is a deliberate clamp, skip
        self.joint_limit_barrier_joints = None  # optional (name, ...): barrier ONLY these joints
        self.pelvis_track_weight = 0.0  # source-pelvis position prior (kills the pelvis<->waist null space)
        self.arm_reg_weight = 0.0  # source-arm position prior (stops the solver parking a redundant arm)
        self.root_rate_weight = 0.0  # match the SOURCE root angular rate (kills inferred-torso jitter)
        self.joint_angle_weight = 0.0  # track SOURCE anatomical joint angles, not just keypoint positions
        self.keypoint_track_weight = 0.0  # absolute position prior on EVERY mapped keypoint
        self.ball_track = None  # (T, 3) ball positions in the SOLVE frame (scaled + shifted)
        self.ball_contacts = ()  # tuples (toe_link_name, start, end, r0) in solve scale
        self.ball_tolerance = 0.005  # m, slack either side of r0
        self.joint_angle_targets = None  # {joint name: (T,) target angle in rad}
        self.root_quat_track = None  # (T, 4) wxyz source root orientation, or None
        self.swing_ankle_weight = 0.0  # neutral-ankle prior while a foot is in free swing
        # Source foot heading (T, 2) [left, right], rad about +z: an ankle and a toe point leave the
        # sole's yaw to a 0.13 m lever, so the robot foot's own forward axis is steered to it.
        self.foot_yaw_seq = None
        self.foot_yaw_weight = 0.0
        self.toe_kp_indices = None  # positions of the toe keypoints in the joint mapping (ground anchoring)
        self.hip_kp_indices = None  # (left, right) hip keypoint positions in the mapping (lateral axis)
        self.ankle_kp_indices = None
        self.foot_min_sep = 0.0  # m; minimum lateral toe separation the targets are widened to
        self.self_collision_escape = 0.02  # m per SQP iteration a violated pair may separate
        self.self_collision_margin = 0.0  # m; soft repulsion starts here (0 = off)
        self.self_collision_margin_weight = 0.0
        self.ground_margin = 0.0  # m; soft cushion above the ground for non-foot bodies (0 = off)
        self.body_contact_gain = 0.0  # temporal smoothing x (1 + gain * non-foot bodies on the ground)
        self.body_contact_root = 0.0  # root-orientation smoothing added per non-foot body on the ground
        self.ground_margin_weight = 200.0
        self.foot_stack_clearance = 0.0  # m of extra vertical gap a crossing foot keeps over the stance foot
        self.foot_stack_thickness = 0.035  # m; foot body origin to its top surface
        self.foot_stack_half_width = 0.05  # m; footprint half extents, for the plan-overlap test
        self.foot_stack_half_length = 0.11
        self.foot_stack_weight = 100.0
        # Per-frame posture cost on the upper-arm twist rows: (T, 2) weights, used when the source elbow is
        # nearly straight and the swivel is undefined, so the twist does not wander into a branch.
        self.twist_prior_seq = None
        self.twist_rows = None
        # Arm-plane matching: (shoulder, elbow, wrist) keypoint-name triples whose plane normal is
        # steered to the source's, which fixes the elbow swivel branch without a joint target.
        self.arm_plane_triples = ()
        self.arm_plane_weight = 0.0
        # Tolerance for foot sticking constraints in x, y.
        self.foot_sticking_tolerance = foot_sticking_tolerance
        self.stick_tol_seq = None  # optional (T, 2) per-frame [left, right] sticking band, metres
        self._init_foot_lock(foot_lock)
        self._self_collision_config = self_collision

        # Setup visualization if requested
        if self.visualize:
            self._setup_visualization()

        # Load Mujoco model
        if self.object_name == "ground":
            robot_xml_path = self.robot_model_path.replace(".urdf", ".xml")
        elif self.object_name == "multi_boxes":
            robot_xml_path = self.task_constants.SCENE_XML_FILE
        else:
            robot_xml_path = self.robot_model_path.replace(".urdf", "_w_" + self.object_name + ".xml")

        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        print("Loading robot model from: ", robot_xml_path)

        self.robot_data = mujoco.MjData(self.robot_model)
        self._init_self_collision(self._self_collision_config)

        if self.robot_data.qpos.shape[0] > 7 + self.task_constants.ROBOT_DOF:
            self.has_dynamic_object = True
        else:
            self.has_dynamic_object = False

        self.nq = self.robot_model.nq

        self.q_a_init_idx = q_a_init_idx
        self.q_a_indices = np.arange(7 + self.q_a_init_idx, 7 + self.task_constants.ROBOT_DOF)

        self.nq_a = len(self.q_a_indices)

        # Create complete limits with floating base (-inf, inf) and actuated joint limits
        n_floating_base = 7
        joint_names = [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)]
        actuated_joints = [(i, name) for i, name in enumerate(joint_names) if name]  # Filter out None names

        large_number = 1e6
        complete_lower_limits = np.concatenate(
            [-large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 0]]
        )
        complete_upper_limits = np.concatenate(
            [large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 1]]
        )

        self.q_a_lb = complete_lower_limits[self.q_a_indices]
        self.q_a_ub = complete_upper_limits[self.q_a_indices]

        self.q_a_lb[np.array(list(self.task_constants.MANUAL_LB.keys())).astype(int)] = list(
            self.task_constants.MANUAL_LB.values()
        )
        self.q_a_ub[np.array(list(self.task_constants.MANUAL_UB.keys())).astype(int)] = list(
            self.task_constants.MANUAL_UB.values()
        )

        # dqa rows that are actuated joints: qpos < 7 is the floating base (free translation + a
        # quaternion whose MANUAL_LB/UB box is +-1), which must never see the joint-limit barrier.
        self._actuated_rows = np.flatnonzero(self.q_a_indices >= 7)
        self._ankle_rows = {
            side: self._resolve_joint_rows(tuple(joints))
            for side, joints in self.task_constants.ANKLE_JOINTS.items()
        }
        # Keypoint priors index the joint mapping, whose order and names vary by source format.
        match_names = list(self.laplacian_match_links.keys())
        self._pelvis_kp = match_names.index(root_keypoint(match_names))
        self._arm_kps = tuple(
            i for i, name in enumerate(match_names) if any(part in name.lower() for part in _ARM_KEYPOINT_PARTS)
        )

        # Prevent too much waist twist
        self.Q_diag = np.zeros(self.nq_a) * 1e-3
        self.Q_diag[np.array(list(self.task_constants.MANUAL_COST.keys())).astype(int)] = list(
            self.task_constants.MANUAL_COST.values()
        )

        self.w_nominal_tracking_init = w_nominal_tracking_init
        self.nominal_tracking_tau = nominal_tracking_tau
        self.track_nominal_indices = task_constants.NOMINAL_TRACKING_INDICES

    def _barrier_rows_and_margins(self) -> tuple[np.ndarray, np.ndarray]:
        """Actuated dqa rows that get a joint-limit barrier, and each one's margin (rad).

        The margin is RELATIVE (a fraction of the joint's own range, capped by the absolute value):
        a flat margin would spend 38% of ankle_roll's tiny +-0.262 range, and `edge` clips genuinely
        need that range. Joints deliberately clamped to a sliver -- wrist_yaw is held at +-0.05 rad on
        purpose, and is the most "saturated" joint in the corpus BY DESIGN -- are skipped outright.

        ``joint_limit_barrier_joints`` narrows it further to a named set. A BLANKET barrier measurably
        fights stairs descent (lowering the body wants knee/ankle near their stops, and `settle`
        regresses); restricting it to the joints where saturation is an actual measured defect keeps
        the rest of the leg free.
        """
        rows = (
            self._actuated_rows
            if self.joint_limit_barrier_joints is None
            else self._resolve_joint_rows(tuple(self.joint_limit_barrier_joints))
        )
        rng = self.q_a_ub[rows] - self.q_a_lb[rows]
        keep = rng > self.joint_limit_barrier_min_range
        rows, rng = rows[keep], rng[keep]
        margins = np.minimum(float(self.joint_limit_barrier_margin), self.joint_limit_barrier_margin_frac * rng)
        return rows, margins

    def _resolve_joint_rows(self, joint_names: tuple[str, ...]) -> np.ndarray:
        """dqa rows for named joints (qpos address -> position within q_a_indices); absent names are skipped."""
        rows = []
        for name in joint_names:
            try:
                joint = self.robot_model.joint(name)
            except KeyError:
                continue
            adr = self.robot_model.jnt_qposadr[joint.id]
            hit = np.flatnonzero(self.q_a_indices == adr)
            if hit.size:
                rows.append(int(hit[0]))
        return np.array(rows, dtype=int)

    def _init_foot_lock(self, foot_lock: FootLockConfig | None) -> None:
        """Initialize foot lock configuration and normalize window mappings."""
        self.foot_lock = foot_lock or FootLockConfig()
        self._foot_lock_windows: dict[str, tuple[tuple[int, int], ...]] = {"left": (), "right": ()}
        if self.foot_lock.windows is None:
            return
        for key, windows in self.foot_lock.windows.items():
            key_lower = key.lower()
            side = None
            if key_lower.startswith("l") or ("left" in key_lower):
                side = "left"
            elif key_lower.startswith("r") or ("right" in key_lower):
                side = "right"
            if side is None:
                continue
            # windows may be (start, end), (start, end, z), (start, end, x, y, z), or additionally carry
            # a stance attitude (start, end, x, y, z, yaw, pitch); normalize to
            # (start, end, x|None, y|None, z|None, yaw|None, pitch|None)
            normalized_windows: list[tuple] = []
            for window in windows:
                if len(window) not in (2, 3, 5, 7):
                    raise ValueError(f"Invalid foot lock window for {key}: {window}")
                start, end = int(window[0]), int(window[1])
                if end < start:
                    raise ValueError(f"Invalid foot lock window with end < start for {key}: {window}")
                if len(window) == 7:
                    normalized_windows.append((start, end, *(float(v) for v in window[2:7])))
                elif len(window) == 5:
                    normalized_windows.append(
                        (start, end, float(window[2]), float(window[3]), float(window[4]), None, None)
                    )
                elif len(window) == 3:
                    normalized_windows.append((start, end, None, None, float(window[2]), None, None))
                else:
                    normalized_windows.append((start, end, None, None, None, None, None))
            self._foot_lock_windows[side] = tuple(normalized_windows)

    def _init_self_collision(self, self_collision: SelfCollisionConfig | None) -> None:
        """Initialize self-collision configuration and precompute geom pairs."""
        sc = self_collision or SelfCollisionConfig()
        self._self_collision_enabled = sc.enable and len(sc.pairs) > 0
        self._self_collision_tolerance = sc.tolerance
        self._self_collision_windows: list[tuple[int, int]] | None = sc.windows
        self._self_collision_geom_pairs: list[tuple[int, int]] = []

        self._sc_last_vis_frame = -1

        if not self._self_collision_enabled:
            return

        m = self.robot_model

        # Build body_name → [geom_ids] mapping (only geoms with collision enabled)
        body_to_geoms: dict[str, list[int]] = {}
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            body_id = m.geom_bodyid[g]
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            body_to_geoms.setdefault(body_name, []).append(g)

        # Build geom pairs from body name pairs
        for body_a, body_b in sc.pairs:
            geoms_a = body_to_geoms.get(body_a, [])
            geoms_b = body_to_geoms.get(body_b, [])
            if not geoms_a:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_a}'")
            if not geoms_b:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_b}'")
            for ga in geoms_a:
                for gb in geoms_b:
                    self._self_collision_geom_pairs.append((ga, gb))

        print(
            f"[SelfCollision] Initialized with {len(self._self_collision_geom_pairs)} geom pairs "
            f"from {len(sc.pairs)} body pairs, tolerance={sc.tolerance}m"
        )

    def _setup_visualization(self):
        """Setup Viser visualization components."""
        self.server = viser.ViserServer()

        # 1) Ensure a world frame exists (absolute path!)
        try:
            self.server.scene.add_frame("/world", show_axes=False)
        except Exception:
            print("Starting viser")

        # Create parent frames for robot and object
        self.robot_base = self.server.scene.add_frame("/world/robot", show_axes=False)

        print("robot_model_path: ", self.robot_model_path)

        # Load robot URDF
        self.robot_urdf = yourdfpy.URDF.load(
            self.robot_model_path,
            load_meshes=True,
            build_scene_graph=True,
        )

        print("Viser using robot URDF: ", self.robot_model_path)

        # Create ViserUrdf instance for robot, attaching it to the robot_base frame
        self.viser_robot = ViserUrdf(
            self.server,
            urdf_or_path=self.robot_urdf,
            root_node_name="/world/robot",  # This links to the robot_base frame we created
        )

        # Similarly for object
        if self.object_model_path:
            self.object_base = self.server.scene.add_frame("/world/object", show_axes=False)

            self.object_urdf = yourdfpy.URDF.load(
                self.object_model_path,
                load_meshes=True,
                build_scene_graph=True,
            )

            # Create ViserUrdf instance for object, attaching it to the object_base frame
            self.viser_object = ViserUrdf(
                self.server,
                urdf_or_path=self.object_urdf,
                root_node_name="/world/object",  # This links to the object_base frame we created
            )
            print("Viser using object URDF: ", self.object_model_path)

        else:
            self.viser_object = None

        # Check the number of actuated joints and their names
        robot_joint_limits = self.viser_robot.get_actuated_joint_limits()
        print("\nRobot joints:")
        print("Number of actuated joints:", len(robot_joint_limits))
        print("Joint names:", list(robot_joint_limits.keys()))

        # Initialize robot with this configuration
        robot_initial_config = np.zeros(len(robot_joint_limits))
        self.viser_robot.update_cfg(robot_initial_config)

        # Add grid
        self.server.scene.add_grid(
            "/world/grid",
            width=8,
            height=8,
            position=(0.0, 0.0, 0.0),
        )

    def draw_mesh_from_geom(self, model, data, geom_id, geom_name, name="/mesh", color=(50, 150, 255), opacity=0.5):
        """
        Draw a single MuJoCo mesh geom (already baked to world coords) in viser.
        color is [0, 255] RGB ints; opacity is [0,1].
        """
        if not hasattr(self, "server"):
            return
        V, F = _world_mesh_from_geom(model, data, geom_id, geom_name)
        self.server.scene.add_mesh_simple(
            name,
            vertices=V.astype(np.float32),
            faces=F.astype(np.int32),
            position=(0.0, 0.0, 0.0),  # already world-frame
            color=tuple(int(c) for c in color),
            opacity=float(opacity),
        )

    def draw_mesh_pair_with_contact(
        self,
        model,
        data,
        geom_id1,
        geom_id2,
        geom1_name,
        geom2_name,
        fromto=None,
        group_name="pair",
        color1=(50, 150, 255),
        color2=(255, 120, 60),
        opacity=0.45,
        show_segment=True,
    ):
        """
        Draw two meshes and (optionally) a contact/query segment.
        Uses the existing self.draw_keypoints(...) to visualize points.
        """
        # Note: sometime geom does not have mesh, mesh_id will be -1
        if int(model.geom_dataid[geom_id1]) == -1 or int(model.geom_dataid[geom_id2]) == -1:
            return

        base = f"/{group_name}"
        # meshes
        self.draw_mesh_from_geom(model, data, geom_id1, geom1_name, name=f"{base}/mesh1", color=color1, opacity=opacity)
        self.draw_mesh_from_geom(model, data, geom_id2, geom2_name, name=f"{base}/mesh2", color=color2, opacity=opacity)

        # contact points (q: green, c: red) via your draw_keypoints
        if fromto is not None:
            q = np.asarray(fromto[:3], dtype=float)
            c = np.asarray(fromto[3:], dtype=float)

            # your existing helper (rgba expects floats 0..1)
            self.draw_keypoints(q, name=f"{group_name}_q", rgba=(0.0, 1.0, 0.0, 1.0))
            self.draw_keypoints(c, name=f"{group_name}_c", rgba=(1.0, 0.0, 0.0, 1.0))


    def _apply_limb_retarget(self, human_joint_motions):
        """Rescale the mapped source keypoints to the robot's own segment lengths.

        A uniform height scale keeps human proportions, so on a robot with different proportions the
        mapped targets are unreachable and the IK saturates joints spanning them. Rescaling per segment
        keeps bone directions and gives the solver reachable targets. Off by default; enable with
        ``limb_retarget=True``.

        Args:
            human_joint_motions: ``(T, J, 3)`` source joints.

        Returns:
            ``(T, J, 3)`` with the mapped keypoints rescaled; other joints untouched.
        """
        offset = float(getattr(self, "ground_kp_offset", 0.0))  # planted toe target height above the robot toe
        if not getattr(self, "limb_retarget", False):
            return human_joint_motions - np.array([0.0, 0.0, offset])
        import mujoco as mj

        from holosoma_retargeting.hcrl.limb_retarget import rescale_to_robot_limbs, robot_segment_lengths

        if getattr(self, "_limb_cache", None) is None:
            parent, length, rigid = robot_segment_lengths(
                self.robot_model, self.robot_data, self.laplacian_match_links, mj
            )
            self._limb_cache = (parent, length, rigid)
            span = ", ".join(
                f"{k}:{v:.3f}{'' if rigid.get(k) else '(cap)'}" for k, v in sorted(length.items()) if v > 0
            )
            print(f"[limb-retarget] robot segments: {span}", flush=True)
        parent, length, rigid = self._limb_cache
        names = list(self.laplacian_match_links.keys())
        out = human_joint_motions.copy()
        kp = out[:, self.smplh_mapped_joint_indices]
        toe_names = [names[i] for i in (self.toe_kp_indices or ())]
        new_kp = rescale_to_robot_limbs(kp, names, parent, length, rigid, horizontal=toe_names)
        # The rescale grows from the root, so a shorter robot leg lifts the feet off the floor by the
        # length difference; keep the lower toe's height instead (less the calibrated toe offset),
        # which is the ground contact. One keypoint per frame, so the body cannot hop when the
        # lowest point would switch between feet.
        if self.toe_kp_indices:
            toes = np.asarray(self.toe_kp_indices)
            lo = toes[np.argmin(kp[:, toes, 2], axis=1)]
            rows = np.arange(len(kp))
            dz = kp[rows, lo, 2] - offset - new_kp[rows, lo, 2]
        else:
            dz = kp[:, :, 2].min(axis=1) - offset - new_kp[:, :, 2].min(axis=1)
        out += dz[:, None, None] * np.array([0.0, 0.0, 1.0])
        out[:, self.smplh_mapped_joint_indices] = new_kp + (out[:, self.smplh_mapped_joint_indices] - kp)
        # Human feet pass closer than the robot's feet are wide; instead of letting self-collision resolve
        # that at contact, widen each foot's targets laterally (about the pelvis heading) by half the
        # shortfall, so the solve anticipates the robot's foot width.
        min_sep = float(getattr(self, "foot_min_sep", 0.0))
        if min_sep > 0 and self.toe_kp_indices and len(self.toe_kp_indices) == 2 and getattr(self, "hip_kp_indices", None):
            mapped = np.asarray(self.smplh_mapped_joint_indices)
            hl, hr = (mapped[i] for i in self.hip_kp_indices)
            lat = out[:, hl, :2] - out[:, hr, :2]  # left-pointing lateral axis
            lat /= np.linalg.norm(lat, axis=1, keepdims=True) + 1e-9
            tl, tr = (mapped[i] for i in self.toe_kp_indices)
            sep = ((out[:, tl, :2] - out[:, tr, :2]) * lat).sum(1)  # left toe minus right toe, along lateral
            shortfall = np.clip(min_sep - sep, 0.0, None)  # >0 when the feet are closer than allowed
            shift = (0.5 * shortfall)[:, None] * lat
            cols_l = [tl] + ([mapped[self.ankle_kp_indices[0]]] if getattr(self, "ankle_kp_indices", None) else [])
            cols_r = [tr] + ([mapped[self.ankle_kp_indices[1]]] if getattr(self, "ankle_kp_indices", None) else [])
            for c in cols_l:
                out[:, c, :2] += shift
            for c in cols_r:
                out[:, c, :2] -= shift
        # On flat ground a toe target below the sole is source noise the non-penetration constraint
        # will refuse anyway; asking for it only drags the body down at foot strike.
        if self.toe_kp_indices and self.object_name == "ground" and offset != 0.0 and getattr(self, "toe_floor_clamp", True):
            toe_cols = np.asarray(self.smplh_mapped_joint_indices)[np.asarray(self.toe_kp_indices)]
            lift = np.maximum(0.005 - out[:, toe_cols, 2], 0.0)  # (T, 2)
            out[:, toe_cols, 2] += lift
            # lift the ankle with its toe, or the foot target pitches toe-up whenever the toe is clamped
            ankle_cols = getattr(self, "ankle_kp_cols", None)
            if ankle_cols is not None:
                out[:, ankle_cols, 2] += lift
        return out

    def retarget_motion(
        self,
        human_joint_motions,
        object_poses,
        object_poses_augmented,
        object_points_local_demo,
        object_points_local,
        foot_sticking_sequences,
        q_a_init=None,
        q_nominal_list=None,
        original=True,
        dest_res_path=None,
    ):
        """
        The main function to retarget an entire motion sequence frame by frame.

        Args:
            human_joint_motions (np.ndarray): (num_frames, num_joints, 3) array.
            object_poses (np.ndarray): (num_frames, 7) array of demo object poses (quat, trans).
            object_poses_augmented (np.ndarray): (num_frames, 7) array of augmented object poses (quat, trans).
            object_points_local_demo (np.ndarray): Demo object points in local frame (rest pose).
            object_points_local (np.ndarray): Current object points in local frame (rest pose).
            foot_sticking_sequences (list): List of foot sticking sequences for each frame.
            q_a_init (np.ndarray, optional): Initial robot configuration.
            q_a_nominal (np.ndarray, optional): Nominal robot configuration.

        Returns:
            tuple: (retargeted_motions, obj_pts_demo_list, obj_pts_list, tetrahedra)
        """
        human_joint_motions = self._apply_limb_retarget(human_joint_motions)

        num_frames = human_joint_motions.shape[0]
        if q_nominal_list is not None:
            q_locked_list = q_nominal_list
        else:
            q_locked_list = np.zeros((num_frames, self.nq))
            q_locked_list[0, self.q_a_indices] = q_a_init

        # Only a dynamic object owns the last 7 qpos slots; on a ground scene they are the right leg,
        # and the identity object pose would seed Right_Hip_Yaw at 1.0 (its stop) on every clip.
        if self.has_dynamic_object:
            q_locked_list[:, -7:] = object_poses_augmented
        q = np.copy(q_locked_list[0])
        retargeted_motions = [q]

        tetrahedra = []
        obj_pts_demo_list = []  # scaled object pts
        obj_pts_list = []  # original size object pts

        print(f"\nStarting motion retargeting for {num_frames} frames...")

        with tqdm(range(num_frames)) as pbar:
            for i in pbar:
                # Get object poses and transform points
                object_quat_demo = object_poses[i, 3:]
                object_trans_demo = object_poses[i, :3]

                # Get human joint positions and create interaction mesh in object frame
                human_mapped_joints = human_joint_motions[i, self.smplh_mapped_joint_indices]
                if getattr(self, "_dump_targets", None) is not None:
                    self._dump_targets.append(np.asarray(human_mapped_joints, dtype=np.float32).copy())

                if self.object_name == "ground":
                    human_mapped_joints_in_object = human_mapped_joints
                else:
                    human_mapped_joints_in_object = transform_points_world_to_local(
                        object_quat_demo, object_trans_demo, human_mapped_joints
                    )

                source_vertices, source_tetrahedra = create_interaction_mesh(
                    np.vstack([human_mapped_joints_in_object, object_points_local_demo])
                )
                tetrahedra.append(source_tetrahedra)

                if self.debug:
                    # Only for visualization
                    object_quat = object_poses_augmented[i, 3:]
                    object_trans = object_poses_augmented[i, :3]
                    obj_pts_demo = transform_points_local_to_world(
                        object_quat_demo, object_trans_demo, object_points_local_demo
                    )
                    obj_pts = transform_points_local_to_world(object_quat, object_trans, object_points_local)

                    obj_pts_demo_list.append(obj_pts_demo)
                    obj_pts_list.append(obj_pts)
                    human_kpts_handle_list = self.draw_keypoints(human_mapped_joints, name="human_kpts")  # 15 X 3
                    obj_kpts_demo_handle_list = self.draw_keypoints(
                        obj_pts_demo, name="object_demo_kpts", rgba=(1, 0, 0, 1)
                    )  # 100 X 3
                    obj_kpts_handle_list = self.draw_keypoints(
                        obj_pts, name="object_kpts", rgba=(0, 1, 1, 1)
                    )  # 100 X 3

                # Create adjacency list and calculate target Laplacian coordinates
                adj_list = get_adjacency_list(source_tetrahedra, len(source_vertices))
                target_laplacian = calculate_laplacian_coordinates(source_vertices, adj_list)

                # Run optimization
                if original:
                    w_nominal_tracking = self.w_nominal_tracking_init
                else:
                    w_nominal_tracking = self.w_nominal_tracking_init * np.exp(-i / self.nominal_tracking_tau)

                q, cost = self.iterate(
                    q_locked=q_locked_list[i],
                    q_n=q,
                    q_t_last=retargeted_motions[-1],
                    target_laplacian=target_laplacian,
                    adj_list=adj_list,
                    obj_pts_local=object_points_local,
                    foot_sticking=foot_sticking_sequences[i],
                    w_nominal_tracking=w_nominal_tracking,
                    q_a_nominal=(q_nominal_list[i, self.q_a_indices] if q_nominal_list is not None else None),
                    init_t=i == 0,
                    n_iter=50 if i == 0 else 10,
                    frame_idx=i,
                    q_t_last2=retargeted_motions[-2] if len(retargeted_motions) >= 2 else None,
                    human_src_pts=human_mapped_joints_in_object,
                )
                if self.debug:
                    robot_link_positions = self._get_robot_link_positions(
                        q, self.laplacian_match_links.values()
                    )  # 15 X 3
                    robot_kpts_handle_list = self.draw_keypoints(
                        robot_link_positions, name="robot_kpts", rgba=(0, 1, 0, 1)
                    )

                retargeted_motions.append(q)
                if self.visualize and self.debug:
                    self.draw_q(q)

                pbar.set_postfix(cost=cost)

        # Remove previous debug visualization
        if self.debug:
            for handle in human_kpts_handle_list:
                handle.remove()
            human_kpts_handle_list.clear()

            for handle in obj_kpts_demo_handle_list:
                handle.remove()
            obj_kpts_demo_handle_list.clear()

            for handle in obj_kpts_handle_list:
                handle.remove()
            obj_kpts_handle_list.clear()

            for handle in robot_kpts_handle_list:
                handle.remove()
            robot_kpts_handle_list.clear()

        # Save results. The ball rides along: downstream contact is only right against the same
        # centres the clearance term solved against.
        extras = {} if self.ball_seq is None else {"ball": np.asarray(self.ball_seq, dtype=np.float32)}
        np.savez(
            dest_res_path,
            qpos=np.array(retargeted_motions)[1:],
            human_joints=human_joint_motions,
            fps=30,
            cost=cost,
            **extras,
        )
        print("Saving results to path:", dest_res_path)

        if self.visualize:
            robot_dof = len(self.viser_robot.get_actuated_joint_limits())

            create_motion_control_sliders(
                server=self.server,
                viser_robot=self.viser_robot,
                robot_base_frame=self.robot_base,
                motion_sequence=np.asarray(retargeted_motions)[1:],
                robot_dof=robot_dof,
                viser_object=self.viser_object,
                object_base_frame=getattr(self, "object_base", None) if self.viser_object else None,
                contains_object_in_qpos=bool(self.viser_object) and bool(self.has_dynamic_object),
                initial_fps=30,
                initial_interp_mult=2,
                loop=False,
            )

            # 4) optional: visibility toggle
            with self.server.gui.add_folder("Visibility"):
                show_meshes_cb = self.server.gui.add_checkbox("Show meshes", self.viser_robot.show_visual)

                @show_meshes_cb.on_update
                def _(_):
                    self.viser_robot.show_visual = show_meshes_cb.value
                    if self.viser_object is not None:
                        self.viser_object.show_visual = show_meshes_cb.value

        return (
            np.array(retargeted_motions)[1:],
            obj_pts_demo_list,
            obj_pts_list,
            tetrahedra,
        )

    def solve_single_iteration(
        self,
        q_locked: np.ndarray,
        q_a_n_last: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        verbose=False,
        init_t=False,
        frame_idx: int = 0,
        q_t_last2: np.ndarray | None = None,
        human_src_pts: np.ndarray | None = None,
    ):
        """The main function to solve a single iteration of the DiffIK problem.
        Args:
            q_locked: the locked robot and object configuration.
            q_a_n_last: the last optimized robot configuration at current time step.
            q_t_last: the robot and object configuration at the last time step.
            foot_sticking: a sequence of booleans indicating whether the foot [left, right] is sticking to the ground.
            smpl_joints: the (possibly scaled) SMPL joint positions to match for IK.
            q_ref: the reference robot configuration.
            smpl_joints_original: the original SMPL joint positions (used for contact matching).
            obj_original: the original object pose (used for contact matching).
            init_t: the current time step is the first time step.
            frame_idx: frame index used by explicit foot lock window constraints.
            human_src_pts: (15, 3) scaled source keypoints for this frame, in the SAME frame as
                ``p_OC_dict`` (object frame), i.e. what the Laplacian target was built from.
        """
        assert len(q_a_n_last) == self.nq_a

        # Lock the object pose and set the current robot slice to last accepted solution
        q = np.copy(q_locked)
        q[self.q_a_indices] = q_a_n_last

        # Compute Laplacian pieces
        J_OC_dict, p_OC_dict, _ = self._calc_manipulator_jacobians(
            q, links=self.laplacian_match_links, obj_frame=(self.object_name != "ground")
        )
        robot_link_keys = list(self.laplacian_match_links.keys())
        V_r = len(robot_link_keys)
        V_o = len(obj_pts_local)
        V = V_r + V_o

        # Stack Jacobians for robot points
        J_V = np.zeros((3 * V, self.nq_a))
        for i, key in enumerate(robot_link_keys):
            J_V[3 * i : 3 * (i + 1), :] = J_OC_dict[key]

        robot_pts_local = np.array([p_OC_dict[k] for k in robot_link_keys])
        vertices = np.vstack([robot_pts_local, obj_pts_local])  # (V x 3)

        use_lap = float(np.max(np.atleast_1d(self.laplacian_weights))) > 0.0

        # Decision variables
        dqa = cp.Variable(len(self.q_a_indices), name="dqa")

        # Constraints list
        constraints = []

        if use_lap:
            L = calculate_laplacian_matrix(vertices, adj_list)  # (V x V), EXPECT SPARSE OR SMALL
            if not sp.issparse(L):
                L = sp.csr_matrix(L)

            Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
            J_L = Kron @ J_V

            lap0 = L @ vertices
            lap0_vec = lap0.reshape(-1)  # (3V,)
            target_lap_vec = target_laplacian.reshape(-1)  # (3V,)

            w_v = (self.laplacian_weights * np.ones(V)).astype(float)  # (V,)
            sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

            lap_var = cp.Variable(3 * V, name="laplacian")

            # Linear equality
            constraints += [cp.Constant(J_L[:, self.q_a_indices]) @ dqa - lap_var == -lap0_vec]

        # Foot constraints (sticking + foot lock window Z pinning)
        apply_foot_sticking = (self.q_a_init_idx < 12) and self.activate_foot_sticking
        apply_foot_lock = (self.q_a_init_idx < 12) and self.foot_lock.enable
        foot_anchor_terms = []
        foot_orient_terms = []
        if apply_foot_sticking or apply_foot_lock:
            J_WF_dict, p_WF_dict, _ = self._calc_manipulator_jacobians(q, links=self.foot_links, obj_frame=False)

            # HARD Cartesian foot velocity cap (every frame, both feet): constraint releases and mesh
            # pulls must never teleport a foot -- physically impossible foot motion is excluded in the
            # QP itself rather than depending on contact detection being right. Cap per 30 fps frame.
            if self.teleport_guard and not init_t:
                _, p_last_all, _ = self._calc_manipulator_jacobians(q_t_last, links=self.foot_links, obj_frame=False)
                for key, J_WF in J_WF_dict.items():
                    if self.foot_lock.lock_links_substr and self.foot_lock.lock_links_substr not in key:
                        continue
                    step_max = self._foot_step_cap(frame_idx, key) - 0.005  # in-QP slightly under backtrack
                    already = p_WF_dict[key] - p_last_all[key]  # displacement accrued at linearization pt
                    Jc = J_WF[:3, self.q_a_indices]
                    for axis in range(3):
                        constraints += [
                            Jc[axis] @ dqa >= -step_max - already[axis],
                            Jc[axis] @ dqa <= step_max - already[axis],
                        ]

            # Foot sticking: constrain XY to stay near previous frame position
            if apply_foot_sticking:
                _, p_WF_t_last_dict, _ = self._calc_manipulator_jacobians(
                    q_t_last, links=self.foot_links, obj_frame=False
                )
                left_key = right_key = None
                for key in foot_sticking:
                    if key.lower().startswith("l"):
                        left_key = key
                    elif key.lower().startswith("r"):
                        right_key = key
                if left_key is None or right_key is None:
                    raise ValueError("foot_sticking must include one left* and one right* key")

                for key, J_WF in J_WF_dict.items():
                    anchor_active = self.foot_lock.enable and self._foot_lock_anchor(key, frame_idx) is not None
                    apply_left = ("left" in key) and foot_sticking[left_key] and not anchor_active
                    apply_right = ("right" in key) and foot_sticking[right_key] and not anchor_active
                    if apply_left or apply_right:
                        # A fixed 1 mm band stops a settling foot dead and releases it with a jump;
                        # let the stance foot move as far as the SOURCE foot moved this frame.
                        tol = self.foot_sticking_tolerance
                        if self.stick_tol_seq is not None:
                            tol = float(self.stick_tol_seq[min(frame_idx, len(self.stick_tol_seq) - 1), 0 if apply_left else 1])
                        p_lb = p_WF_t_last_dict[key] - p_WF_dict[key] - tol
                        p_ub = p_lb + 2 * tol  # symmetric window

                        Jxy = J_WF[:2, self.q_a_indices]  # (2 x nq_act)
                        constraints += [
                            Jxy @ dqa >= p_lb[:2],
                            Jxy @ dqa <= p_ub[:2],
                        ]

            # Foot lock windows: pin Z to floor within configured frame ranges
            if apply_foot_lock:
                foot_anchor_terms = []
                for key, J_WF in J_WF_dict.items():
                    if self.foot_lock.lock_links_substr and self.foot_lock.lock_links_substr not in key:
                        continue
                    anchor = self._foot_lock_anchor(key, frame_idx)
                    if anchor is None:
                        # anticipatory approach shaping: blend the last 4 swing frames toward the
                        # UPCOMING anchor (cosine ramp into t0) so the foot arrives with ~0 settle
                        a_ramp, approach = self._foot_approach_anchor(key, frame_idx)
                        if approach is not None and not init_t:
                            p_now = p_WF_dict[key]
                            for axis, a_val in enumerate(approach):
                                delta = float(np.clip(a_val - p_now[axis], -0.04, 0.04))
                                Ja = J_WF[axis, self.q_a_indices]
                                foot_anchor_terms.append(a_ramp * cp.square(Ja @ dqa - delta))
                        continue

                    # strong SOFT pull toward the source plant position (<= 4 cm/frame/axis). A hard
                    # band is jointly infeasible whenever the lagging foot's straight-line path to the
                    # anchor grazes terrain (non-penetration blocks the required step); as an objective,
                    # non-penetration steers the foot around the corner over a few frames instead.
                    p_now = p_WF_dict[key]
                    for axis, a_val in enumerate(anchor):
                        if a_val is None:
                            continue
                        delta = float(np.clip(a_val - p_now[axis], -0.04, 0.04))
                        Ja = J_WF[axis, self.q_a_indices]
                        foot_anchor_terms.append(cp.square(Ja @ dqa - delta))

            # Soft foot-orientation ENGAGEMENT DAMPING: position pins alone let the ankle snap to a
            # constraint-consistent attitude the frame a window binds. Penalize the foot's per-frame
            # rotation (relative to the previous FRAME, so SQP iterations cannot compound it) through
            # the first ~0.2 s of each window -- the foot still reaches whatever attitude its own
            # geometry settles into, just smoothly. See _foot_orient_damp; swing/mid-stance untouched.
            if apply_foot_lock and self.foot_orient_weight > 0 and not init_t:
                for side in ("left", "right"):
                    w_damp = self._foot_orient_damp(side, frame_idx)
                    if w_damp <= 0:
                        continue
                    bid = mujoco.mj_name2id(
                        self.robot_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link"
                    )
                    R_last = self._body_rot(q_t_last, bid)
                    R_now = self._body_rot(q, bid)  # also restores FK(q) for the Jacobian below
                    Jr = self._calc_rot_jacobian(bid)[:, self.q_a_indices]
                    accrued = Rotation.from_matrix(R_now @ R_last.T).as_rotvec()
                    foot_orient_terms.append(w_damp * cp.sum_squares(Jr @ dqa + accrued))

        # Non-penetration constraints. Sources can START inside geometry (that is the defect being
        # cleaned); demanding full escape in one linearized step is jointly infeasible, so cap the
        # per-iteration escape rate -- existing penetration decays over a few iterations instead.
        Js, phis = self._update_jacobians_and_phis_from_q(q)
        ground_soft = []
        # bodies other than the feet touching the ground: a body lying on the floor is held by many
        # contacts against targets it cannot reach, and the root rocks as the active set switches
        n_body_ground = sum(
            1 for key in phis
            if any("ground" in self._geom_names[g] for g in key) and not any("foot" in self._geom_names[g] for g in key)
        )
        contact_gain = 1.0 + self.body_contact_gain * n_body_ground
        for key, phi in phis.items():
            Ja_n_full = Js[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            rhs = min(-phi - self.penetration_tolerance, 0.02)
            constraints += [Ja_n @ dqa >= rhs]
            # cushion: a body lying on the ground otherwise rides the hard boundary while the targets
            # pull it into the floor, and rocks as the active contact set switches
            if self.ground_margin > 0:
                names = (self._geom_names[key[0]], self._geom_names[key[1]])
                if any("ground" in n for n in names) and not any("foot" in n for n in names):
                    ground_soft.append(cp.square(cp.pos(self.ground_margin - (phi + Ja_n @ dqa))))

        # Self-collision constraints: new_distance >= tolerance  =>  phi + J @ dqa >= tol. Exact-penalty
        # slack (hard whenever reachable) since a pair can start overlapping while foot sticking or the
        # trust region forbids the separating step, which makes the pure inequality infeasible.
        Js_sc, phis_sc = self._compute_self_collision_constraints(frame_idx)
        sc_slacks, sc_soft = [], []
        for key, phi in phis_sc.items():
            Ja_n_full = Js_sc[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            rhs = min(self._self_collision_tolerance - phi, self.self_collision_escape)
            sl = cp.Variable(nonneg=True)
            sc_slacks.append(sl)
            constraints += [Ja_n @ dqa >= rhs - sl]
            # smooth repulsion inside `margin`: the pair is pushed apart as it approaches, so the
            # separation is anticipated over frames rather than paid at contact
            if self.self_collision_margin > 0:
                sc_soft.append(cp.square(cp.pos(self.self_collision_margin - (phi + Ja_n @ dqa))))

        # Joint limits constraints (actuated)
        if self.activate_joint_limits:
            constraints += [
                dqa >= (self.q_a_lb - q_a_n_last),
                dqa <= (self.q_a_ub - q_a_n_last),
            ]

        # Step size constraints (Lorentz cone)
        constraints += [cp.SOC(self.step_size, dqa)]

        # Objective
        obj_terms = []
        term_labels = []

        def _add_term(label, expr):  # noqa: ANN001, ANN202
            obj_terms.append(expr)
            term_labels.append(label)


        if use_lap:
            _add_term("laplacian", cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec)))

        if sc_slacks:
            _add_term("self_collision_slack", 500.0 * cp.sum(cp.hstack(sc_slacks)))
        if sc_soft:
            _add_term("self_collision_margin", self.self_collision_margin_weight * cp.sum(cp.hstack(sc_soft)))
        if ground_soft:
            _add_term("ground_margin", self.ground_margin_weight * cp.sum(cp.hstack(ground_soft)))

        # hcrl: FOOT STACKING CLEARANCE. When the feet overlap in plan the crossing foot must clear the
        # stance foot vertically; the required gap ramps with the overlap, so the foot descends as it
        # slides off instead of dropping when the contact constraint releases.
        if self.foot_stack_clearance > 0 and not init_t:
            ids = [mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, self.task_constants.FOOT_LINKS[s]) for s in ("left", "right")]
            if min(ids) >= 0:
                self.robot_data.qpos[:] = q
                mujoco.mj_forward(self.robot_model, self.robot_data)
                pl, pr = (self.robot_data.xpos[b].copy() for b in ids)
                hi, lo_ = (0, 1) if pl[2] >= pr[2] else (1, 0)
                # overlap in the LOWER foot's own frame: side-by-side feet (lateral offset beyond the
                # foot's half-width) do not overlap however close their centres are
                R_lo = self.robot_data.xmat[ids[lo_]].reshape(3, 3)
                rel = R_lo.T @ ((pl if hi == 0 else pr) - (pr if hi == 0 else pl))
                lat = float(np.clip((self.foot_stack_half_width + 0.02 - abs(rel[1])) / 0.02, 0.0, 1.0))
                lon = float(np.clip((self.foot_stack_half_length + 0.03 - abs(rel[0])) / 0.03, 0.0, 1.0))
                overlap = lat * lon
                if overlap > 0.0:
                    J_hi = self._calc_pos_jacobian(ids[hi])[:, self.q_a_indices]
                    J_lo = self._calc_pos_jacobian(ids[lo_])[:, self.q_a_indices]
                    dz_now = float((pl if hi == 0 else pr)[2] - (pr if hi == 0 else pl)[2])
                    need = self.foot_stack_thickness + overlap * self.foot_stack_clearance
                    _add_term("foot_stack", self.foot_stack_weight * cp.square(cp.pos(need - (dz_now + (J_hi[2] - J_lo[2]) @ dqa))))

        # foot anchor pull (see foot-lock block): heavily weighted so stance feet land and stay planted
        if apply_foot_lock and foot_anchor_terms:
            _add_term("foot_anchor", 200.0 * cp.sum(cp.hstack(foot_anchor_terms)))

        # stance foot-orientation engagement (see foot-lock block): ramped, slew-limited, soft
        if foot_orient_terms:
            _add_term("foot_orient", self.foot_orient_weight * cp.sum(cp.hstack(foot_orient_terms)))

        # nominal tracking for selected indices
        if (w_nominal_tracking > 0) and (q_a_nominal is not None):
            idx = np.array(self.track_nominal_indices, dtype=int)
            if idx.size > 0:
                z = dqa[idx] - (q_a_nominal[idx] - q_a_n_last[idx])
                _add_term("nominal", w_nominal_tracking * cp.sum_squares(z))

        # Q_diag cost
        Qd = np.asarray(self.Q_diag, dtype=float).reshape(-1)
        _add_term("posture_Qd", cp.sum_squares(cp.multiply(np.sqrt(Qd), dqa + q_a_n_last)))
        if self.twist_prior_seq is not None and self.twist_rows is not None:
            _t = min(frame_idx, len(self.twist_prior_seq) - 1)
            for k, row in enumerate(self.twist_rows):
                w_t = float(self.twist_prior_seq[_t, k])
                if w_t > 0:
                    _add_term("twist_prior", w_t * cp.square(dqa[row] + q_a_n_last[row]))

        # Acceleration damping: penalize deviation from the constant-velocity extrapolation. Unlike the
        # velocity term it costs nothing on smooth motion, so it suppresses 2-frame QP-tie flips only.
        if np.any(self.accel_damp_weight) and (q_t_last2 is not None) and not init_t:
            dqa_cv = 2 * q_t_last[self.q_a_indices] - q_t_last2[self.q_a_indices] - q_a_n_last
            w_ad = contact_gain * np.broadcast_to(np.asarray(self.accel_damp_weight, dtype=float), (self.nq_a,))
            _add_term("accel_damp", cp.sum_squares(cp.multiply(np.sqrt(w_ad), dqa - dqa_cv)))

        # Smoothness cost. Frame 0 has no previous frame -- q_t_last is the initial guess (a T-pose),
        # and pulling toward it makes the first solved frame a swing away from it.
        dqa_smooth = q_t_last[self.q_a_indices] - q_a_n_last
        # additive root-orientation damping per body-ground contact: the lying-body rock is in the
        # root, and a route without absolute position terms cannot afford to slow every joint
        root_extra = self.body_contact_root * n_body_ground
        if root_extra > 0 and not init_t:
            _add_term("root_contact_damp", root_extra * cp.sum_squares(dqa[3:7] - dqa_smooth[3:7]))
        if init_t:
            pass
        elif np.isscalar(self.smooth_weight):
            _add_term("smooth_scalar", contact_gain * self.smooth_weight * cp.sum_squares(dqa - dqa_smooth))
        else:
            Wsmooth = contact_gain * np.asarray(self.smooth_weight, dtype=float)
            if Wsmooth.ndim == 1:
                _add_term("smooth_vec", cp.sum_squares(cp.multiply(np.sqrt(Wsmooth), dqa - dqa_smooth)))
            else:
                # if a full matrix was supplied, fall back to quad_form
                _add_term("smooth_quad", cp.quad_form(dqa - dqa_smooth, Wsmooth))

        # hcrl: JOINT-LIMIT BARRIER. The hard box above admits solutions pinned exactly AT a stop, and
        # nothing costs that -- waist_pitch sits on its +0.52 stop in 54% of corpus frames. A pinned joint
        # has zero control headroom downstream. This is a one-sided quadratic hinge that is exactly zero
        # outside `margin` of a stop, so the interior of the range is undistorted.
        if self.joint_limit_barrier_weight > 0 and self.joint_limit_barrier_margin > 0:
            rows, m = self._barrier_rows_and_margins()
            if rows.size:
                q_new = dqa[rows] + q_a_n_last[rows]
                over = cp.pos(q_new - (self.q_a_ub[rows] - m))
                under = cp.pos((self.q_a_lb[rows] + m) - q_new)
                _add_term("joint_limit_barrier", 
                    self.joint_limit_barrier_weight * (cp.sum_squares(over) + cp.sum_squares(under))
                )

        # hcrl: SOLE-ORIENTATION MATCHING. Two mapped points per foot (an ankle and a toe) define a
        # line, so the sole's pitch and roll cost the mesh objective nothing and it settles toe-down.
        # Rotate the robot sole toward the source's own sole plane; the residual is projected off the
        # normal so foot YAW stays free (the human's yaw is not a target).
        if self.sole_normal_weight > 0 and self.sole_normal_seq is not None and not init_t:
            t = min(max(frame_idx, 0), len(self.sole_normal_seq) - 1)
            for k, side in enumerate(("left", "right")):
                bid = mujoco.mj_name2id(
                    self.robot_model, mujoco.mjtObj.mjOBJ_BODY, self.task_constants.FOOT_LINKS[side]
                )
                if bid < 0:
                    continue
                normal_now = self._body_rot(q, bid)[:, 2]  # sole plane normal is the foot body's local +z
                target = np.asarray(self.sole_normal_seq[t, k], dtype=np.float64)
                axis = np.cross(normal_now, target)
                sin_a = float(np.linalg.norm(axis))
                if sin_a < 1e-8:
                    continue
                error = axis / sin_a * float(np.arctan2(sin_a, float(np.dot(normal_now, target))))
                keep_tilt = np.eye(3) - np.outer(normal_now, normal_now)
                Jr = self._calc_rot_jacobian(bid)[:, self.q_a_indices]
                _add_term("sole_normal", self.sole_normal_weight * cp.sum_squares(keep_tilt @ (Jr @ dqa - error)))

                # While the source sole is on the ground, put the robot's sole there too. The sole
                # plane sits SOLE_OFFSET below the foot body, so the body's target height follows it.
                # While the source sole is on the ground, put the robot's sole there too. Orientation
                # alone LIFTS the foot: rotating a toe-down sole flat raises its lowest contact point.
                if self.sole_height_weight > 0 and self.sole_height_seq is not None:
                    target_height = float(self.sole_height_seq[t, k])
                    if target_height < self.sole_planted_height:
                        sole_now = min(self.robot_data.xpos[b][2] for b in self._sole_body_ids(side))
                        Jp = self._calc_pos_jacobian(bid)[:, self.q_a_indices]
                        # One-sided: penalize HOVERING above the source's sole height, never being
                        # below it. A symmetric pull settles at a compromise well above the floor,
                        # and driving through the target relies on the non-penetration constraint
                        # catching it -- which is what put feet through the ground. Here that
                        # constraint is the floor and this term only ever pushes down onto it.
                        _add_term("foot_approach", 
                            self.sole_height_weight
                            * cp.square(cp.pos(Jp[2] @ dqa + (sole_now - target_height)))
                        )

        # hcrl: FOOT HEADING. Steer the foot body's forward axis (toward its toe sphere) to the source
        # ankle->toe heading; yaw only, so it never fights the sole-normal term above.
        if self.foot_yaw_weight > 0 and self.foot_yaw_seq is not None and not init_t:
            t = min(max(frame_idx, 0), len(self.foot_yaw_seq) - 1)
            for k, side in enumerate(("left", "right")):
                bid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, self.task_constants.FOOT_LINKS[side])
                toe = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot_sphere_5_link")
                if bid < 0 or toe < 0:
                    continue
                self._body_rot(q, bid)  # restores FK(q) for the Jacobian
                fwd = self.robot_data.xpos[toe] - self.robot_data.xpos[bid]
                err = float(self.foot_yaw_seq[t, k]) - float(np.arctan2(fwd[1], fwd[0]))
                err = float(np.arctan2(np.sin(err), np.cos(err)))
                Jr = self._calc_rot_jacobian(bid)[:, self.q_a_indices]
                _add_term("foot_yaw", self.foot_yaw_weight * cp.square(Jr[2] @ dqa - err))

        # hcrl: ARM PLANE. The upper-arm twist is unobserved by positions alone (the elbow apex can point
        # either way for one hand position); match the normal of the shoulder-elbow-wrist plane to the
        # source's, linearized through the three keypoint Jacobians. Relative shape only, no joint target.
        if self.arm_plane_weight > 0 and human_src_pts is not None and self.arm_plane_triples:
            for names in self.arm_plane_triples:
                if not all(n in J_OC_dict for n in names):
                    continue
                s_, e_, w_ = (p_OC_dict[n] for n in names)
                Js_, Je_, Jw_ = (J_OC_dict[n] for n in names)
                u, vv = e_ - s_, w_ - e_
                n_now = np.cross(u, vv)
                mag = float(np.linalg.norm(n_now))
                if mag < 1e-4:
                    continue
                i_s, i_e, i_w = (robot_link_keys.index(n) for n in names)
                u_s, v_s = human_src_pts[i_e] - human_src_pts[i_s], human_src_pts[i_w] - human_src_pts[i_e]
                n_src = np.cross(u_s, v_s)
                sin_bend = float(np.linalg.norm(n_src) / (np.linalg.norm(u_s) * np.linalg.norm(v_s) + 1e-9))
                # the plane is undefined for a straight arm; fade the term in between 15 and 35 deg of bend
                gate = float(np.clip((sin_bend - np.sin(np.radians(15))) / (np.sin(np.radians(35)) - np.sin(np.radians(15))), 0.0, 1.0))
                if gate <= 0.0:
                    continue
                n_tgt = n_src / np.linalg.norm(n_src) * mag  # same bend magnitude, source direction
                # d(u x v) = [u]x dv - [v]x du, with du = Je - Js, dv = Jw - Je (dqa)
                def skew(a):
                    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                dn = skew(u) @ (Jw_ - Je_) - skew(vv) @ (Je_ - Js_)
                # normalize by the segment lengths, not |n|: dividing by a near-zero normal made the
                # term's curvature explode at small bends and the elbow flickered straight/bent
                scale = float(np.linalg.norm(u) * np.linalg.norm(vv) + 1e-9)
                _add_term("arm_plane", gate * self.arm_plane_weight * cp.sum_squares((dn @ dqa + (n_now - n_tgt)) / scale))

        # hcrl: BALL CLEARANCE. The human is scaled to robot size but the ball is not, so a contact
        # that was tangent for the human lands (1 - scale) * radius inside it -- 33 mm for the T1.
        # Each foot is held at the clearance the HUMAN's own foot had, which is an absolute distance
        # to an object that never shrank. One-sided: the ball track is registered to the mocap only
        # to within a few cm, so pushing a foot out is safe, pulling it in would chase that noise.
        if self.ball_weight > 0 and self.ball_seq is not None and not init_t:
            t = min(max(frame_idx, 0), len(self.ball_seq) - 1)
            centre = np.asarray(self.ball_seq[t], dtype=np.float64)
            sides = ("left", "right") if np.isfinite(centre).all() else ()
            for k, side in enumerate(sides):
                bid = mujoco.mj_name2id(
                    self.robot_model, mujoco.mjtObj.mjOBJ_BODY, self.task_constants.FOOT_LINKS[side]
                )
                if bid < 0:
                    continue
                rot = self._body_rot(q, bid)
                origin = self.robot_data.xpos[bid].astype(np.float64)
                points = origin + self.ball_foot_points[side] @ rot.T
                offset = points - centre
                dist = np.maximum(np.linalg.norm(offset, axis=1), 1e-9)
                keep_out = self.ball_radius + (
                    0.0 if self.ball_clearance_seq is None else float(self.ball_clearance_seq[t, k])
                )
                deepest = np.argsort(dist)[:BALL_CONTACT_POINTS]
                deepest = deepest[dist[deepest] < keep_out]
                if not deepest.size:
                    continue
                Jp = self._calc_pos_jacobian(bid)[:, self.q_a_indices]
                Jr = self._calc_rot_jacobian(bid)[:, self.q_a_indices]
                rows = np.stack(
                    [offset[i] / dist[i] @ self._point_jacobian(Jp, Jr, points[i] - origin) for i in deepest]
                )
                obj_terms.append(
                    self.ball_weight * cp.sum_squares(cp.pos(-(rows @ dqa + (dist[deepest] - keep_out))))
                )

        # hcrl: PELVIS-TRACKING PRIOR. The interaction mesh is a DIFFERENTIAL (Laplacian) objective, so the
        # absolute root pose is in its null space: the solver is free to lean the pelvis back and cancel it
        # with waist pitch (measured corr -0.58, torso net upright). Anchoring the pelvis to the source
        # removes that null space at its origin -- it is the primary fix for the waist saturation, and the
        # barrier above is the guard.
        if self.pelvis_track_weight > 0 and human_src_pts is not None:
            k = robot_link_keys[self._pelvis_kp]
            _add_term("pelvis_track", 
                self.pelvis_track_weight
                * cp.sum_squares(J_OC_dict[k] @ dqa - (human_src_pts[self._pelvis_kp] - p_OC_dict[k]))
            )

        # hcrl: ROOT ANGULAR-RATE PRIOR. The source gives only joint POSITIONS, so torso orientation is
        # inferred from a handful of points and comes out noisier than the motion it came from (measured
        # 1.45x the source's per-frame rotation). Matching the source's rotation RATE -- not its absolute
        # orientation, whose rest pose differs from the robot's -- removes that jitter.
        if self.root_rate_weight > 0 and self.root_quat_track is not None and not init_t:
            _t = min(max(frame_idx, 1), len(self.root_quat_track) - 1)
            _q0, _q1 = self.root_quat_track[_t - 1], self.root_quat_track[_t]
            if float(np.dot(_q0, _q1)) < 0.0:  # quaternion double cover
                _q1 = -_q1
            _dq_src = _q1 - _q0
            _add_term("root_rate", self.root_rate_weight * cp.sum_squares(dqa[3:7] - _dq_src))

        # hcrl: JOINT-ANGLE TRACKING. Everything else in this objective matches keypoint POSITIONS. On a
        # robot whose proportions differ from the human's, position matching necessarily distorts the joint
        # ANGLES -- which for expressive motion (dance) carry the content, while an end-effector position
        # does not. The 1-DOF hinges have an unambiguous anatomical angle in the source, so track it.
        if self.joint_angle_weight > 0 and self.joint_angle_targets:
            ja_terms = []
            for jname, track in self.joint_angle_targets.items():
                rows = self._resolve_joint_rows((jname,))
                if len(rows) == 0:
                    continue
                _t = min(frame_idx, len(track) - 1)
                ja_terms.append(cp.square(dqa[rows[0]] + q_a_n_last[rows[0]] - float(track[_t])))
            if ja_terms:
                _add_term("joint_angle", self.joint_angle_weight * cp.sum(cp.hstack(ja_terms)))

        # hcrl: BALL-CONTACT RADIUS CONSTRAINT. During a detected contact segment the toe is held at a
        # constant distance r0 from the (splined) ball center: a target-level hold cannot beat the ~50 mm
        # tracking residual, so this is a HARD constraint like foot sticking, linearized along the current
        # radial direction u -- |u . (p + J dqa - ball)| within r0 +- tol. Radial only, so the foot may
        # roll around the ball surface, which is what dribbling contact does.
        if self.ball_track is not None and not init_t:
            _bt = min(frame_idx, len(self.ball_track) - 1)
            _finite = bool(np.isfinite(self.ball_track[_bt]).all())  # NaN centre = track dropout
            for _link, _s0, _e0, _r0 in self.ball_contacts if _finite else ():
                if not (_s0 <= frame_idx < _e0) or _link not in J_OC_dict:
                    continue
                _u = p_OC_dict[_link] - self.ball_track[_bt]
                _nu = float(np.linalg.norm(_u))
                if _nu < 1e-6:
                    continue
                _u = _u / _nu
                _radial = _u @ (p_OC_dict[_link] + J_OC_dict[_link] @ dqa - self.ball_track[_bt])
                # exact-penalty slack: hard whenever reachable, never infeasible on entry frames where
                # the toe starts outside r0 +- tol (one linearized step may not close the gap)
                _sl = cp.Variable(nonneg=True)
                constraints += [
                    _radial <= _r0 + self.ball_tolerance + _sl,
                    _radial >= _r0 - self.ball_tolerance - _sl,
                ]
                _add_term("ball_contact_slack", 500.0 * _sl)

        # hcrl: KEYPOINT POSITION TRACKING. The mesh term matches the Laplacian -- relative shape -- and
        # the only ABSOLUTE position priors were the pelvis and the arm, so hips/knees/ankles/feet had
        # nothing pinning them to their targets and settled 50-85 mm away. Sweeping solver iterations and
        # step size changed the residual by 0.1 mm, so this is the cost's optimum, not under-convergence.
        if self.keypoint_track_weight > 0 and human_src_pts is not None:
            kp_terms = [
                cp.sum_squares(J_OC_dict[robot_link_keys[i]] @ dqa - (human_src_pts[i] - p_OC_dict[robot_link_keys[i]]))
                for i in range(len(robot_link_keys))
            ]
            _add_term("keypoint_track", self.keypoint_track_weight * cp.sum(cp.hstack(kp_terms)))

        # hcrl: ARM REGULARIZER. Arms carry no contact in most clips, so the mesh leaves them
        # under-determined and the solver parks them in whatever pose the null space lands on (the
        # "awkward arm" defect -- same redundancy class as the waist, not a joint-limit problem: the
        # elbow saturates in only 0.02% of frames). Pull all 6 arm keypoints toward the source.
        if self.arm_reg_weight > 0 and human_src_pts is not None:
            arm_terms = [
                cp.sum_squares(J_OC_dict[robot_link_keys[i]] @ dqa - (human_src_pts[i] - p_OC_dict[robot_link_keys[i]]))
                for i in self._arm_kps
            ]
            _add_term("arm_reg", self.arm_reg_weight * cp.sum(cp.hstack(arm_terms)))

        # hcrl: NEUTRAL-ANKLE PRIOR IN FREE SWING. Ankle pitch/roll are unconstrained mid-swing and drift
        # onto their stops (roll is pinned in 35% of `edge` frames), so the foot lands pointed/inverted.
        # Gated strictly OFF through the approach ramp and the engagement window, so it can never fight the
        # attitude the foot settles into on contact -- that target was tried and measured to hurt
        # (see _foot_orient_damp), and this prior deliberately does not reintroduce one.
        if self.swing_ankle_weight > 0 and apply_foot_lock and not init_t:
            for side in ("left", "right"):
                key = f"{side}_ankle_roll_link"
                free_swing = (
                    not self._is_foot_locked_in_window(key, frame_idx)
                    and self._foot_orient_damp(side, frame_idx) <= 0.0
                    and self._foot_approach_anchor(key, frame_idx)[1] is None
                )
                rows_a = self._ankle_rows[side]
                if free_swing and rows_a.size:
                    _add_term("swing_ankle", 
                        self.swing_ankle_weight * cp.sum_squares(dqa[rows_a] + q_a_n_last[rows_a])
                    )

        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

        # -------- Solve with Clarabel --------
        solver_kwargs = {"verbose": verbose}
        problem.solve(solver=cp.CLARABEL, **solver_kwargs)
        if (problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) and init_t:
            constraints = [c for c in constraints if not isinstance(c, cp.constraints.second_order.SOC)]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
            problem.solve(solver=cp.CLARABEL, **solver_kwargs)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            # hcrl debug: report which hard constraints are already violated at dqa = 0
            print(f"[infeasible-debug] frame {frame_idx}: step_size={self.step_size}")
            for key, phi in phis.items():
                if phi < 0.05:
                    print(f"  pen pair {self._geom_names[key[0]]} vs {self._geom_names[key[1]]}: phi={phi:.4f}")
            if self.activate_joint_limits:
                lb_gap = (self.q_a_lb - q_a_n_last).max()
                ub_gap = (q_a_n_last - self.q_a_ub).max()
                print(f"  joint-limit worst gaps: lb {lb_gap:.4f}, ub {ub_gap:.4f} (>0 == already out of limits)")
            raise RuntimeError(f"CVXPY solve failed: {problem.status}")

        dqa_star = dqa.value
        cost = problem.value
        if getattr(self, "debug_terms", False) and frame_idx % int(getattr(self, "debug_terms_every", 25)) == 0:
            vals = [(lab, float(t.value)) for lab, t in zip(term_labels, obj_terms)]
            tot = sum(v for _, v in vals) or 1.0
            top = sorted(vals, key=lambda kv: -kv[1])
            print(f"[terms] frame {frame_idx} total={tot:.3f}: "
                  + "  ".join(f"{k}={v:.3f}({100 * v / tot:.0f}%)" for k, v in top if v > 1e-4), flush=True)

        q_star = np.copy(q)
        q_star[self.q_a_indices] = dqa_star + q_a_n_last
        q_star[3:7] /= np.linalg.norm(q_star[3:7]) + 1e-12

        return q_star, cost

    def _foot_orient_damp(self, side: str, frame_idx: int) -> float:
        """Engagement-phase angular-rate damping weight in [0, 1]: cosine ramp-in over the last 3
        swing frames BEFORE a window (descent toe-down rotation finishes before contact, not after),
        full for the first 4 stance frames (the constraint-activation snap), cosine fade to 0 by
        frame 9. No attitude TARGET exists -- flat and source-implied targets both measurably fight
        the attitude the G1's own geometry settles into (longer foot than the scaled human); damping
        only limits the RATE of reaching it. Mid/late stance and the rest of swing are untouched."""
        for w in self._foot_lock_windows.get(side, ()):
            start, end = w[0], w[1]
            if start - 3 <= frame_idx < start:
                return 0.5 * (1 - np.cos(np.pi * (frame_idx - start + 4) / 3))  # 0.25 / 0.75 / 1.0 into t0
            if start <= frame_idx <= min(end, start + 9):
                k = frame_idx - start
                return 1.0 if k <= 3 else 0.5 * (1 + np.cos(np.pi * (k - 3) / 6))
        return 0.0

    def _foot_approach_anchor(self, foot_link_key: str, frame_idx: int) -> tuple[float, tuple | None]:
        """(cosine ramp in [0, 1], anchor xyz) over the last 4 swing frames before a stance window:
        anticipatory approach shaping so the foot ARRIVES at its known anchor and the post-contact
        settle ~ 0 (descents land toe-first with 3-4 cm arrival error otherwise). The ramp is tiny 4
        frames out and ~0.9 the frame before contact -- it bends only the final approach, not the arc."""
        key_lower = foot_link_key.lower()
        side = "left" if "left" in key_lower else ("right" if "right" in key_lower else None)
        if side is None:
            return 0.0, None
        for w in self._foot_lock_windows.get(side, ()):
            start = w[0]
            if start - 4 <= frame_idx < start and w[2] is not None:
                p = (frame_idx - (start - 5)) / 4.0  # reaches full anchor weight the frame before contact
                return 0.5 * (1 - np.cos(np.pi * p)), (w[2], w[3], w[4])
        return 0.0, None

    def _body_rot(self, q: np.ndarray, body_id: int) -> np.ndarray:
        """World rotation matrix of a body at configuration q (leaves robot_data at q's FK)."""
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        return self.robot_data.xmat[body_id].reshape(3, 3).copy()

    def _sole_body_ids(self, side: str) -> list[int]:
        """Body ids of one foot's sole spheres, resolved once the robot model exists."""
        if side not in self._sole_body_id_cache:
            self._sole_body_id_cache[side] = [
                mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in self.task_constants.SOLE_LINKS[side]
            ]
        return self._sole_body_id_cache[side]

    @staticmethod
    def _point_jacobian(jac_pos: np.ndarray, jac_rot: np.ndarray, arm: np.ndarray) -> np.ndarray:
        """Jacobian of a point rigidly offset from a body origin: v_p = v_body + w x arm.

        Args:
            jac_pos: Positional Jacobian of the body origin, shape (3, n).
            jac_rot: Rotational Jacobian of the body, shape (3, n).
            arm: World-frame offset from the body origin to the point, shape (3,).

        Returns:
            Positional Jacobian of the point, shape (3, n).
        """
        skew = np.array([[0.0, -arm[2], arm[1]], [arm[2], 0.0, -arm[0]], [-arm[1], arm[0], 0.0]])
        return jac_pos - skew @ jac_rot

    def _calc_pos_jacobian(self, body_id: int) -> np.ndarray:
        """Positional Jacobian (3 x nq) at the CURRENT robot_data FK state: v_world = Jp @ qdot."""
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        p = self.robot_data.xpos[body_id].astype(np.float64).reshape(3, 1)
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p, int(body_id))
        return Jp @ self._build_transform_qdot_to_qvel_fast()

    def _calc_rot_jacobian(self, body_id: int) -> np.ndarray:
        """Rotational Jacobian (3 x nq) at the CURRENT robot_data FK state: w_world = Jr @ qdot."""
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        p = self.robot_data.xpos[body_id].astype(np.float64).reshape(3, 1)
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p, int(body_id))
        return Jr @ self._build_transform_qdot_to_qvel_fast()

    def _foot_step_cap(self, frame_idx: int, foot_link_key: str) -> float:
        """Backtrack-level toe step cap for this frame/foot: 0.075 default ~= natural swing peak;
        a per-clip (T, 2) override raises it through real flight phases (jumps) so the cap cannot
        flatten the motion, while stance-adjacent frames keep the tight default."""
        if self.foot_step_max_seq is None:
            return 0.075
        k = 1 if "right" in foot_link_key.lower() else 0
        t = min(max(frame_idx, 0), len(self.foot_step_max_seq) - 1)
        return float(self.foot_step_max_seq[t, k])

    def _is_foot_locked_in_window(self, foot_link_key: str, frame_idx: int) -> bool:
        """Check whether a foot link is locked by configured frame windows."""
        key_lower = foot_link_key.lower()
        side = None
        if "left" in key_lower:
            side = "left"
        elif "right" in key_lower:
            side = "right"
        if side is None:
            return False

        return any(w[0] <= frame_idx <= w[1] for w in self._foot_lock_windows.get(side, ()))

    def _foot_lock_anchor(self, foot_link_key: str, frame_idx: int) -> tuple | None:
        """(x|None, y|None, z) anchor for a locked foot at this frame (z falls back to the global floor)."""
        key_lower = foot_link_key.lower()
        side = "left" if "left" in key_lower else ("right" if "right" in key_lower else None)
        if side is None:
            return None
        for start, end, x, y, z, *_ in self._foot_lock_windows.get(side, ()):
            if start <= frame_idx <= end:
                return (x, y, z if z is not None else float(self.foot_lock.z_floor))
        return None

    def _compute_self_collision_constraints(self, frame_idx: int):
        """Compute Jacobians and distances for self-collision body pairs.

        Assumes ``mj_forward`` has already been called with the current q
        (done by ``_update_jacobians_and_phis_from_q`` which runs first).

        Returns:
            Js: dict mapping (geom_a, geom_b) -> relative Jacobian (1 x nq)
            phis: dict mapping (geom_a, geom_b) -> signed distance
        """
        if not self._self_collision_enabled:
            return {}, {}

        # Check frame windows
        if self._self_collision_windows is not None:
            if not any(start <= frame_idx <= end for start, end in self._self_collision_windows):
                return {}, {}

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        if not hasattr(self, "_geom_names"):
            raise RuntimeError(
                "[SelfCollision] _geom_names not initialized. Please run _prefilter_pairs_with_mj_collision first."
            )

        _first_iter = self._sc_last_vis_frame != frame_idx
        if _first_iter:
            self._sc_last_vis_frame = frame_idx

        for geom_a, geom_b in self._self_collision_geom_pairs:
            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, geom_a, geom_b, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(geom_a),
                    m.geom(geom_b),
                    self._geom_names[geom_a],
                    self._geom_names[geom_b],
                    fromto,
                    dist,
                )
                key = ("self", geom_a, geom_b)
                Js[key] = J_rel
                phis[key] = float(dist)

        if _first_iter and self.visualize:
            self._draw_self_collision_geoms()

        return Js, phis

    def iterate(
        self,
        q_locked: np.ndarray,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        init_t: bool = False,
        n_iter: int = 10,  # overridden by self.solve_n_iter when set
        frame_idx: int = 0,
        q_t_last2: np.ndarray | None = None,
        human_src_pts: np.ndarray | None = None,
    ):
        """Iterate the solver for multiple iterations."""
        last_cost = np.inf
        n_iter = int(getattr(self, "solve_n_iter", 0) or n_iter)
        for _ in range(n_iter):
            q_a_n_last = q_n[self.q_a_indices]
            q_n, cost = self.solve_single_iteration(
                q_locked=q_locked,
                q_a_n_last=q_a_n_last,
                q_t_last=q_t_last,
                target_laplacian=target_laplacian,
                adj_list=adj_list,
                obj_pts_local=obj_pts_local,
                foot_sticking=foot_sticking,
                q_a_nominal=q_a_nominal,
                w_nominal_tracking=w_nominal_tracking,
                init_t=init_t,
                frame_idx=frame_idx,
                q_t_last2=q_t_last2,
                human_src_pts=human_src_pts,
            )
            if np.isclose(cost, last_cost):
                break
            last_cost = cost

        # HARD teleport guarantee on the FINAL pose: the in-QP velocity cap bounds the LINEARIZED step,
        # and linearization error across SQP iterations lets the true step overshoot. Backtrack the
        # whole frame update (nonlinear FK check) until no toe moves more than FOOT_STEP_MAX from the
        # previous frame -- physically impossible foot motion cannot leave this function.
        if self.teleport_guard and not init_t:

            def _step_excess(q_test: np.ndarray) -> float:
                """Max (toe step - per-foot cap); > 0 means some toe exceeds its cap."""
                _, p_now, _ = self._calc_manipulator_jacobians(q_test, links=self.foot_links, obj_frame=False)
                _, p_prev, _ = self._calc_manipulator_jacobians(q_t_last, links=self.foot_links, obj_frame=False)
                ex = [
                    float(np.linalg.norm(p_now[k] - p_prev[k])) - self._foot_step_cap(frame_idx, k)
                    for k in p_now
                    if (not self.foot_lock.lock_links_substr) or self.foot_lock.lock_links_substr in k
                ]
                return max(ex) if ex else 0.0

            if _step_excess(q_n) > 0:
                lo, hi = 0.0, 1.0
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    q_mid = q_t_last + mid * (q_n - q_t_last)
                    q_mid[3:7] /= np.linalg.norm(q_mid[3:7])  # keep the base quat unit under blending
                    if _step_excess(q_mid) > 0:
                        hi = mid
                    else:
                        lo = mid
                q_n = q_t_last + lo * (q_n - q_t_last)
                q_n[3:7] /= np.linalg.norm(q_n[3:7])
        return q_n, cost

    def _draw_self_collision_geoms(self):
        """Draw collision cylinders for self-collision geom pairs in viser."""
        if not hasattr(self, "server") or not self._self_collision_enabled:
            return
        m, d = self.robot_model, self.robot_data
        seen_geoms: set[int] = set()
        colors = [(255, 80, 80), (80, 80, 255)]  # red for first body, blue for second
        for geom_a, geom_b in self._self_collision_geom_pairs:
            for idx, gid in enumerate([geom_a, geom_b]):
                if gid in seen_geoms:
                    continue
                seen_geoms.add(gid)
                gtype = int(m.geom_type[gid])
                if gtype not in (3, 5):  # 3 = capsule, 5 = cylinder
                    continue
                radius = float(m.geom_size[gid][0])
                half_len = float(m.geom_size[gid][1])
                cyl = trimesh.creation.capsule(radius=radius, height=2 * half_len, count=[16, 16])
                # World transform from MuJoCo data
                pos = d.geom_xpos[gid]
                rot_mat = d.geom_xmat[gid].reshape(3, 3)
                transform = np.eye(4)
                transform[:3, :3] = rot_mat
                transform[:3, 3] = pos
                cyl.apply_transform(transform)
                body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or ""
                self.server.scene.add_mesh_simple(
                    f"/world/sc_geom/{body_name}_g{gid}",
                    vertices=cyl.vertices.astype(np.float32),
                    faces=cyl.faces.astype(np.int32),
                    color=colors[idx % 2],
                    opacity=0.35,
                )

    def draw_q(self, q: np.ndarray):
        """Draw a single robot configuration."""
        # Update robot joint configurations
        robot_joint_positions = q[7 : 7 + self.task_constants.ROBOT_DOF]
        self.viser_robot.update_cfg(robot_joint_positions)

        # Update robot base pose using set_transform
        robot_quat = q[3:7]  # Base orientation
        robot_pos = q[:3]  # Base position

        # Update robot base frame
        self.robot_base.position = robot_pos
        self.robot_base.wxyz = robot_quat  # Assuming quaternion is in wxyz order

        # Update object pose if it exists
        if hasattr(self, "viser_object") and self.viser_object is not None:
            if self.has_dynamic_object:
                object_quat = q[-4:]
                object_pos = q[-7:-4]
            else:
                object_quat = np.asarray([1, 0, 0, 0])
                object_pos = np.zeros(3)

            # Update object base frame
            self.object_base.position = object_pos
            self.object_base.wxyz = object_quat  # Assuming quaternion is in wxyz order

    def draw_keypoints(self, p, name="keypoint", rgba=(0, 0, 1, 1)):
        """Draw keypoints in visualization."""
        if not hasattr(self, "server"):
            return None

        # Create a sphere mesh using trimesh
        sphere = trimesh.primitives.Sphere(radius=0.02)
        vertices = sphere.vertices
        faces = sphere.faces

        color = tuple(int(c * 255) for c in rgba[:3])
        opacity = float(rgba[3])

        kpts_handle_list = []

        # Draw keypoints
        if len(p.shape) == 1:
            # Single point
            kpts_handle = self.server.scene.add_mesh_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                position=p,
                color=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)
        elif len(p.shape) == 2:
            # Multiple points
            kpts_handle = self.server.scene.add_batched_meshes_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                batched_positions=p,
                batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (p.shape[0], 1)),
                batched_colors=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)

        return kpts_handle_list

    def visualize_motion(
        self,
        human_joint_motions,
        obj_pts_demo,
        obj_pts,
        retargeted_motions,
        tetrahedra,
        dt=1 / 30,
        visualize_tetrahedra=False,
    ):
        for i in range(len(human_joint_motions)):
            object_pts_demo = obj_pts_demo[i]
            object_pts = obj_pts[i]
            self.draw_keypoints(human_joint_motions[i, self.smplh_mapped_joint_indices], name="human")
            self.draw_keypoints(object_pts_demo, name="object_demo", rgba=(1, 0, 0, 1))
            self.draw_keypoints(object_pts, name="object", rgba=(0, 1, 0, 1))
            self.draw_q(retargeted_motions[i])
            robot_link_positions = self._get_robot_link_positions(
                retargeted_motions[i], self.laplacian_match_links.values()
            )
            self.draw_keypoints(robot_link_positions, name="robot", rgba=(0, 1, 0, 1))
            input()
            if visualize_tetrahedra:
                self.visualize_tetrahedra(
                    np.vstack(
                        [
                            human_joint_motions[i, self.smplh_mapped_joint_indices],
                            object_pts_demo,
                        ]
                    ),
                    tetrahedra[i],
                    name="human_tetrahedra",
                )
                self.visualize_tetrahedra(
                    np.vstack([robot_link_positions, object_pts]),
                    tetrahedra[i],
                    name="robot_tetrahedra",
                    rgba=(0, 1, 1, 1),
                )
            else:
                time.sleep(dt)

    def visualize_tetrahedra(self, vertices, tetrahedra, name="tetrahedra", color=(0, 0, 0, 1)):
        # Convert color to 0-255 range
        color_255 = np.array(color[:3]) * 255

        # Prepare points and colors for all edges
        points = []
        colors = []

        for tet in tetrahedra:
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = tet[i], tet[j]
                    points.extend([vertices[u], vertices[v]])
                    colors.extend([color_255, color_255])

        # Convert to numpy arrays
        points = np.array(points)
        colors = np.array(colors)

        # Add line segments for all edges at once
        self.server.scene.add_line_segments(
            f"/{name}",
            points=points,
            colors=colors,
            line_width=0.01,
        )

    def _compute_jacobian_for_contact_relative(self, geom1, geom2, geom1_name, geom2_name, fromto, dist):
        # Get closest points from fromto buffer
        pos1 = fromto[:3]  # closest point on geom1
        pos2 = fromto[3:]  # closest point on geom2

        v = pos1 - pos2
        norm_v = np.linalg.norm(v)

        if norm_v > 1e-12:
            nhat_BA_W = np.sign(dist) * (v / norm_v)
        # Degenerate: points coincide. Heuristics fallback.
        # If one side is a plane/ground, use its known normal.
        elif "ground" in geom2_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, 1.0]) * (1.0 if dist >= 0 else -1.0)
        elif "ground" in geom1_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, -1.0]) * (1.0 if dist >= 0 else -1.0)
        else:
            nhat_BA_W = np.array([0.0, 0.0, 0.0])

        J_bodyA = self._calc_contact_jacobian_from_point(geom1.bodyid, pos1, input_world=True)
        J_bodyB = self._calc_contact_jacobian_from_point(geom2.bodyid, pos2, input_world=True)

        # Compute relative Jacobian
        Jc = J_bodyA - J_bodyB

        return nhat_BA_W @ Jc

    def _prefilter_pairs_with_mj_collision(self, threshold: float):
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]

        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        m.geom_margin[:] = threshold

        # Run collision. This runs broad→narrow and fills d.contact.
        mujoco.mj_collision(m, d)

        # Collect unique candidate pairs that involve at least one masked geom
        candidates = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        # Restore margins to keep physics untouched
        m.geom_margin[:] = self._saved_margins

        return candidates

    def _update_jacobians_and_phis_from_q(self, q: np.ndarray):
        self.robot_data.qpos[:] = q

        mujoco.mj_forward(self.robot_model, self.robot_data)  # kinematics & AABBs valid

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        # 1) Fast prefilter via mj_collision with temporary margins
        candidates = self._prefilter_pairs_with_mj_collision(threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        # 2) Precise distance only on candidates (early-exit at threshold)
        contype, conaff = m.geom_contype, m.geom_conaffinity

        def masks_ok(g1, g2):
            if contype[g1] == 0 and conaff[g1] == 0:
                return False
            if contype[g2] == 0 and conaff[g2] == 0:
                return False
            if self.object_name in self._geom_names[g1] and "ground" in self._geom_names[g2]:
                return False
            if "ground" in self._geom_names[g1] and self.object_name in self._geom_names[g2]:
                return False
            return (
                self.object_name in self._geom_names[g1]
                or self.object_name in self._geom_names[g2]
                or "ground" in self._geom_names[g1]
                or "ground" in self._geom_names[g2]
            )

        for g1, g2 in candidates:
            # Optional: keep your own filters here (e.g., skip object-ground, only keep interaction with object/ground)
            if not masks_ok(g1, g2):
                continue

            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, g1, g2, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(g1), m.geom(g2), self._geom_names[g1], self._geom_names[g2], fromto, dist
                )
                Js[(g1, g2)] = J_rel
                phis[(g1, g2)] = float(dist)

                # For debug
                # self.draw_mesh_pair_with_contact(self.robot_model, self.robot_data, g1, g2,   \
                #     self._geom_names[g1], self._geom_names[g2], fromto=fromto)

        return Js, phis

    def _world_to_body_frame(self, p_w: np.ndarray, body_idx: int) -> np.ndarray:
        """Transform point from world frame to body frame."""
        p_w = np.asarray(p_w).reshape(3)
        body_pos = self.robot_data.xpos[body_idx].reshape(3)
        body_mat = self.robot_data.xmat[body_idx].reshape(3, 3)
        return body_mat.T @ (p_w - body_pos)

    def _get_geometry_name(self, geom_id: int) -> str:
        """Get geometry name from ID."""
        return mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

    def _build_transform_qdot_to_qvel_fast(self, use_world_omega=True):
        """
        Return T(q) (nv x nq) such that v = T(q) @ qdot.
        - Free root: qpos=[x,y,z, qw,qx,qy,qz], qvel=[vx,vy,vz, ωx,ωy,ωz]
        where ω and v are WORLD-expressed in MuJoCo.
        - 23 hinge joints: v = qdot.

        If use_world_omega=False, uses BODY-omega mapping (for debugging).
        """
        nq, nv = self.robot_model.nq, self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        # ---- root free joint (assumed joint 0) ----
        j0 = 0
        assert self.robot_model.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE
        qadr = self.robot_model.jnt_qposadr[j0]  # 0
        dadr = self.robot_model.jnt_dofadr[j0]  # 0

        # Linear block: v_lin = xyz_dot
        T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)

        # Angular block: ω_* = 2 * E_*(q) * quat_dot
        w, x, y, z = self.robot_data.qpos[qadr + 3 : qadr + 7]

        def get_e_world(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, qz, -qy],
                    [-qy, -qz, qw, qx],
                    [-qz, qy, -qx, qw],
                ]
            )

        def get_e_body(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, -qz, qy],
                    [-qy, qz, qw, -qx],
                    [-qz, -qy, qx, qw],
                ]
            )

        E_fn = get_e_world if use_world_omega else get_e_body

        # ---- FREE joint #1 (human/root): use model addresses, but this should be the first joint ----
        j_free1 = 0
        assert self.robot_model.jnt_type[j_free1] == mujoco.mjtJoint.mjJNT_FREE
        qadr1 = int(self.robot_model.jnt_qposadr[j_free1])  # expect 0
        dadr1 = int(self.robot_model.jnt_dofadr[j_free1])  # start of its 6 qvel dofs

        qw, qx, qy, qz = self.robot_data.qpos[qadr1 + 3 : qadr1 + 7]
        E1 = 2.0 * E_fn(qw, qx, qy, qz)
        # linear-first: v_W = rdot, ω_W = 2E(q) * quat_dot
        T[dadr1 + 0 : dadr1 + 3, qadr1 + 0 : qadr1 + 3] = np.eye(3)  # v block
        T[dadr1 + 3 : dadr1 + 6, qadr1 + 3 : qadr1 + 7] = E1  # ω block

        if self.has_dynamic_object:
            # ---- FREE joint #2 (object): assume it's the last FREE joint; fill its 6x7 block ----
            # Find it by type (safer than hardcoding tail indices)
            free_joints = [
                j for j in range(self.robot_model.njnt) if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            ]
            assert len(free_joints) >= 2, "Expected two FREE joints (human + object)."
            j_free2 = free_joints[1]  # second FREE joint
            qadr2 = int(self.robot_model.jnt_qposadr[j_free2])  # expect nq-7
            dadr2 = int(self.robot_model.jnt_dofadr[j_free2])  # its 6 qvel dofs (often at nv-6)

            qw, qx, qy, qz = self.robot_data.qpos[qadr2 + 3 : qadr2 + 7]
            E2 = 2.0 * E_fn(qw, qx, qy, qz)
            T[dadr2 + 0 : dadr2 + 3, qadr2 + 0 : qadr2 + 3] = np.eye(3)  # v block
            T[dadr2 + 3 : dadr2 + 6, qadr2 + 3 : qadr2 + 7] = E2  # ω block

        # ---- remaining hinge/slide joints: v = qdot ----
        for j in range(1, self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qa = self.robot_model.jnt_qposadr[j]
                da = self.robot_model.jnt_dofadr[j]
                T[da, qa] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
                raise NotImplementedError("BALL joint block not implemented.")

        return T

    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray, input_world=False):
        """
        Translational Jacobian J(q) (3 x nq) such that
        v_point_world = J(q) @ qdot.

        Fast analytic version: J_qdot = J_v @ T(q)
        """

        p_body = np.asarray(p_body, dtype=float).reshape(3)

        # 1) Make sure kinematics are current once
        mujoco.mj_forward(self.robot_model, self.robot_data)

        # 2) World point (3,1) for mj_jac
        R_WB = self.robot_data.xmat[body_idx].reshape(3, 3)
        p_WB = self.robot_data.xpos[body_idx]

        if input_world:
            p_W = p_body.astype(np.float64).reshape(3, 1)
        else:
            p_W = (p_WB + R_WB @ p_body).astype(np.float64).reshape(3, 1)

        # 3) J_v: translational Jacobian wrt generalized velocities (3 x nv)
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_idx))  # Jp = J_v

        T = self._build_transform_qdot_to_qvel_fast()

        return Jp @ T

    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str],
        obj_frame: bool = False,
        point_offsets: np.ndarray | None = None,
    ):
        """Compute position-based Jacobians using MuJoCo."""
        J_XC_dict = {}
        p_XC_dict = {}

        if obj_frame:
            if self.has_dynamic_object:
                obj_quat = q[-4:]
                obj_pos = q[-7:-4]
                obj_rot = Rotation.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]).as_matrix()
                obj_rot_inv = obj_rot.T
            else:
                obj_rot = Rotation.from_quat([0, 0, 0, 1]).as_matrix()
                obj_rot_inv = obj_rot.T
                obj_pos = np.zeros(3)

        q_mujoco = q.copy()
        self.robot_data.qpos[:] = q_mujoco

        mujoco.mj_forward(self.robot_model, self.robot_data)

        for name, link_name in links.items():
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)

            if point_offsets is not None:
                pC_B = point_offsets
            else:
                pC_B = np.zeros(3)

            J = self._calc_contact_jacobian_from_point(body_id, pC_B)
            pos_world = self.robot_data.xpos[body_id]

            if obj_frame:
                p_XC = obj_rot_inv @ (pos_world - obj_pos)
                J_XC = obj_rot_inv @ J
            else:
                p_XC = pos_world
                J_XC = J

            # Store reduced Jacobian and position with hard copies to avoid aliasing
            J_XC_dict[name] = np.array(J_XC[:, self.q_a_indices], dtype=float, copy=True)  # FIX (copy)
            p_XC_dict[name] = np.array(p_XC, dtype=float, copy=True)

        P_WO = {"position": obj_pos, "rotation": obj_rot} if obj_frame else None

        return J_XC_dict, p_XC_dict, P_WO

    def _get_robot_link_positions(self, q, link_names):
        """Get robot link positions for given configuration using Mujoco."""
        mujoco_q = q.copy()

        # Set the configuration
        if mujoco_q.shape != self.robot_data.qpos.shape:
            self.robot_data.qpos = mujoco_q[:-7]  # Exclude object information from q
        else:
            self.robot_data.qpos = mujoco_q
        # Forward kinematics to update all positions
        mujoco.mj_forward(self.robot_model, self.robot_data)

        robot_link_positions = []

        for link_name in link_names:
            # Get body ID from name
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id == -1:
                raise ValueError(f"Body {link_name} not found in Mujoco model")

            # Get position in world frame
            # xpos gives us the position of the body's center of mass in world coordinates
            pos = self.robot_data.xpos[body_id].copy()
            robot_link_positions.append(pos)

        return np.array(robot_link_positions)
