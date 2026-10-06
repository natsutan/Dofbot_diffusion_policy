from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

# =============================================================================
# ROS 2 / Isaac Sim startup settings
# =============================================================================
# These values must be set before the ROS 2 bridge extension is loaded.
os.environ.setdefault("ROS_DOMAIN_ID", "0")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_DISTRO", "jazzy")

from isaacsim import SimulationApp

# GUI enabled.
simulation_app = SimulationApp({"headless": False})

# Isaac Sim modules must be imported after SimulationApp is created.
try:
    import omni.graph.core as og
    import omni.kit.app
    import omni.timeline
    import omni.usd

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    # The URDF importer extension must be enabled before importing its Python API.
    _extension_manager = omni.kit.app.get_app().get_extension_manager()
    _extension_manager.set_extension_enabled_immediate(
        "isaacsim.asset.importer.urdf",
        True,
    )
    for _ in range(10):
        simulation_app.update()

    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    import isaacsim.core.experimental.utils.stage as stage_utils
except Exception:
    simulation_app.close()
    raise


# =============================================================================
# User settings
# =============================================================================

URDF_PATH = Path("/home/natu/dofbot/urdf/dofbot.urdf")
USD_OUTPUT_ROOT = Path("/home/natu/dofbot/generated")

# False: reuse the generated USD when present.
# True : delete the generated package and recreate it from the URDF.
FORCE_URDF_REIMPORT = False

ROBOT_REFERENCE_PATH = "/World/DOFBOT"
ROBOT_REFERENCE_POSITION = (0.0, 0.0, 0.0)

TOPIC_NAME = "/joint_command"
ROS_DOMAIN_ID = 0
GRAPH_PATH = "/ActionGraph"

STAGE_UNITS_IN_METERS = 1.0
STAGE_UP_AXIS = "Z"

# Floor top is z=0.0 m.
FLOOR_CENTER_Z = -0.005
FLOOR_HALF_THICKNESS = 0.005
FLOOR_TOP_Z = FLOOR_CENTER_Z + FLOOR_HALF_THICKNESS

# 30 mm dynamic cube.
PICK_CUBE_PATH = "/World/PickCube"
PICK_CUBE_SIZE = 0.030
PICK_CUBE_POSITION_XY = (0.20, -0.05)
PICK_CUBE_MASS = 0.0027
PICK_CUBE_STATIC_FRICTION = 1.0
PICK_CUBE_DYNAMIC_FRICTION = 0.8
PICK_CUBE_RESTITUTION = 0.0

GRIPPER_JOINT_NAMES = [
    "Llink1_Joint",
    "Llink2_Joint",
    "Llink3_Joint",
    "Rlink1_Joint",
    "Rlink2_Joint",
    "Rlink3_Joint",
]

ARM_JOINT_NAMES = [
    "arm1_Joint",
    "arm2_Joint",
    "arm3_Joint",
    "arm4_Joint",
    "arm5_Joint",
]

# All arm and gripper joints are actively position-driven.
DRIVEN_JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

JOINT_DRIVE_STIFFNESS = 10000.0
JOINT_DRIVE_DAMPING = 1000.0
JOINT_DRIVE_MAX_FORCE = 100000.0


# =============================================================================
# Isaac Sim helpers
# =============================================================================

def enable_extension(extension_name: str) -> None:
    """Enable an Isaac Sim extension and fail clearly if it could not start."""
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate(extension_name, True)

    if not ext_manager.is_extension_enabled(extension_name):
        raise RuntimeError(f"Failed to enable extension: {extension_name}")

    print(f"{extension_name} = enabled")


def enable_required_extensions() -> None:
    """Enable only the extensions required for this simulation."""
    for extension_name in (
        "isaacsim.asset.importer.urdf",
        "isaacsim.core.nodes",
        "isaacsim.ros2.bridge",
        "omni.kit.window.movie_capture",
    ):
        enable_extension(extension_name)


def update_app(frame_count: int) -> None:
    """Advance Kit for extension and stage initialization."""
    for _ in range(frame_count):
        simulation_app.update()


# =============================================================================
# Stage setup
# =============================================================================

def create_new_stage():
    """Create a new metre-based, Z-up USD stage."""
    context = omni.usd.get_context()
    context.new_stage()
    update_app(5)

    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Failed to create a new USD stage")

    UsdGeom.SetStageMetersPerUnit(stage, STAGE_UNITS_IN_METERS)
    UsdGeom.SetStageUpAxis(stage, STAGE_UP_AXIS)
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def add_physics_scene(stage) -> None:
    """Add the PhysX scene and Earth gravity."""
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)


def add_floor(stage) -> None:
    """Add a 4 m x 4 m static collision floor whose top is at z=0."""
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, FLOOR_CENTER_Z))
    floor.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, FLOOR_HALF_THICKNESS))

    collision = UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    collision.CreateCollisionEnabledAttr(True)

    material_path = "/World/Materials/FloorMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.2, 0.2, 0.2)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )
    UsdShade.MaterialBindingAPI(floor.GetPrim()).Bind(material)


def add_pick_cube(stage) -> None:
    """Add a dynamic 30 mm cube at the requested XY position."""
    cube_center_z = FLOOR_TOP_Z + PICK_CUBE_SIZE / 2.0

    cube = UsdGeom.Cube.Define(stage, PICK_CUBE_PATH)
    cube.CreateSizeAttr(PICK_CUBE_SIZE)
    cube.CreateExtentAttr(
        [
            Gf.Vec3f(-PICK_CUBE_SIZE / 2.0),
            Gf.Vec3f(PICK_CUBE_SIZE / 2.0),
        ]
    )
    cube.AddTranslateOp().Set(
        Gf.Vec3d(
            PICK_CUBE_POSITION_XY[0],
            PICK_CUBE_POSITION_XY[1],
            cube_center_z,
        )
    )
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.85, 0.25, 0.10)])

    cube_prim = cube.GetPrim()

    collision = UsdPhysics.CollisionAPI.Apply(cube_prim)
    collision.CreateCollisionEnabledAttr(True)

    rigid_body = UsdPhysics.RigidBodyAPI.Apply(cube_prim)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(False)
    rigid_body.CreateStartsAsleepAttr(False)

    mass = UsdPhysics.MassAPI.Apply(cube_prim)
    mass.CreateMassAttr(PICK_CUBE_MASS)

    physics_material_path = "/World/Materials/PickCubePhysicsMaterial"
    physics_material = UsdShade.Material.Define(stage, physics_material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    material_api.CreateStaticFrictionAttr(PICK_CUBE_STATIC_FRICTION)
    material_api.CreateDynamicFrictionAttr(PICK_CUBE_DYNAMIC_FRICTION)
    material_api.CreateRestitutionAttr(PICK_CUBE_RESTITUTION)

    UsdShade.MaterialBindingAPI.Apply(cube_prim).Bind(
        physics_material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )

    print("\n========== Pick cube created ==========")
    print(f"prim_path = {PICK_CUBE_PATH}")
    print(
        "position  = "
        f"({PICK_CUBE_POSITION_XY[0]:.3f}, "
        f"{PICK_CUBE_POSITION_XY[1]:.3f}, {cube_center_z:.3f}) m"
    )
    print(f"size      = {PICK_CUBE_SIZE:.3f} m")
    print(f"mass      = {PICK_CUBE_MASS:.3f} kg")


def add_light(stage) -> None:
    """Add a dome light."""
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(3000.0)
    dome.CreateExposureAttr(0.0)


# =============================================================================
# URDF import and robot setup
# =============================================================================

def expected_generated_usd_path(urdf_path: Path) -> Path:
    robot_name = urdf_path.stem
    return USD_OUTPUT_ROOT / robot_name / f"{robot_name}.usda"


def create_urdf_import_config(urdf_path: Path) -> URDFImporterConfig:
    return URDFImporterConfig(
        urdf_path=str(urdf_path),
        usd_path=str(USD_OUTPUT_ROOT),
        merge_fixed_joints=False,
        merge_mesh=False,
        debug_mode=False,
        collision_from_visuals=False,
        allow_self_collision=False,
        fix_base=True,
        link_density=None,
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
    )


def generate_or_reuse_robot_usd(urdf_path: Path) -> Path:
    """Convert the URDF to USD, or reuse the cached generated package."""
    urdf_path = urdf_path.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    USD_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cached_usd = expected_generated_usd_path(urdf_path)
    package_dir = cached_usd.parent

    if FORCE_URDF_REIMPORT and package_dir.exists():
        print(f"Removing old generated USD package: {package_dir}")
        shutil.rmtree(package_dir)

    if cached_usd.is_file():
        print(f"Reusing generated robot USD: {cached_usd}")
        return cached_usd

    if package_dir.exists():
        print(f"Removing incomplete generated USD package: {package_dir}")
        shutil.rmtree(package_dir)

    print(f"Importing URDF: {urdf_path}")
    importer = URDFImporter(create_urdf_import_config(urdf_path))
    generated_path = Path(importer.import_urdf()).expanduser().resolve()

    if not generated_path.is_file():
        raise RuntimeError(
            "URDF importer returned a path that does not exist: "
            f"{generated_path}"
        )

    print(f"Generated robot USD: {generated_path}")
    return generated_path


def print_prim_world_position(stage, prim, label: str) -> Gf.Vec3d:
    """Print and return a prim's world-space translation."""
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_matrix = xform_cache.GetLocalToWorldTransform(prim)
    position = world_matrix.ExtractTranslation()
    print(
        f"{label} world position = "
        f"({position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}) m"
    )
    return position


def get_joint_external_name(prim) -> str:
    """Return isaac:nameOverride when authored, otherwise the prim name."""
    override_attr = prim.GetAttribute("isaac:nameOverride")
    if override_attr and override_attr.HasAuthoredValueOpinion():
        value = override_attr.Get()
        if value:
            return str(value)
    return prim.GetName()


def find_descendant_by_name(stage, root_path: str, target_name: str):
    """Find a descendant by prim name or isaac:nameOverride."""
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None

    for prim in Usd.PrimRange(root_prim):
        if (
            prim.GetName() == target_name
            or get_joint_external_name(prim) == target_name
        ):
            return prim

    return None


def add_robot_reference(robot_usd_path: Path) -> str:
    """Reference the robot USD and explicitly place it at world origin."""
    stage_utils.add_reference_to_stage(
        usd_path=str(robot_usd_path),
        path=ROBOT_REFERENCE_PATH,
    )
    update_app(20)

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(ROBOT_REFERENCE_PATH)
    if not prim.IsValid():
        raise RuntimeError(
            f"Robot reference was not created at {ROBOT_REFERENCE_PATH}"
        )

    xform_api = UsdGeom.XformCommonAPI(prim)
    if not xform_api.SetTranslate(Gf.Vec3d(*ROBOT_REFERENCE_POSITION)):
        raise RuntimeError(
            f"Failed to set robot reference position: {ROBOT_REFERENCE_POSITION}"
        )

    update_app(5)

    print(f"Robot USD referenced at: {ROBOT_REFERENCE_PATH}")
    print_prim_world_position(stage, prim, "robot reference")

    base_link_prim = find_descendant_by_name(
        stage,
        ROBOT_REFERENCE_PATH,
        "base_link",
    )
    if base_link_prim is None:
        print("WARNING: base_link prim was not found below the robot reference.")
    else:
        print(f"base_link prim path     = {base_link_prim.GetPath()}")
        print_prim_world_position(stage, base_link_prim, "base_link")

    return ROBOT_REFERENCE_PATH


def find_articulation_root_path(stage, search_root_path: str) -> str:
    """Find the articulation root below the robot reference."""
    search_root = stage.GetPrimAtPath(search_root_path)
    if not search_root.IsValid():
        raise RuntimeError(f"Invalid robot reference path: {search_root_path}")

    candidates: list[str] = []
    for prim in Usd.PrimRange(search_root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            candidates.append(str(prim.GetPath()))

    if not candidates:
        raise RuntimeError(
            "No UsdPhysics.ArticulationRootAPI prim was found below "
            f"{search_root_path}"
        )

    candidates.sort(key=lambda path: (path.count("/"), path))
    articulation_root_path = candidates[0]
    print(f"Articulation root path: {articulation_root_path}")
    return articulation_root_path


def iter_physics_joints(root_prim) -> Iterable:
    """Yield supported USD physics joint prims below root_prim."""
    joint_types = {
        "PhysicsRevoluteJoint",
        "PhysicsPrismaticJoint",
        "PhysicsFixedJoint",
        "PhysicsSphericalJoint",
    }
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() in joint_types:
            yield prim


def set_usd_joint_drives(
    stage,
    robot_reference_path: str,
    joint_names: Iterable[str],
    stiffness: float = JOINT_DRIVE_STIFFNESS,
    damping: float = JOINT_DRIVE_DAMPING,
    max_force: float = JOINT_DRIVE_MAX_FORCE,
) -> None:
    """Set angular position drives on the selected revolute joints."""
    robot_prim = stage.GetPrimAtPath(robot_reference_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Invalid robot path: {robot_reference_path}")

    requested_names = set(joint_names)
    found_names: set[str] = set()

    print("\n========== Set USD Joint Drives ==========")

    for prim in iter_physics_joints(robot_prim):
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue

        external_name = get_joint_external_name(prim)
        prim_name = prim.GetName()

        if (
            external_name not in requested_names
            and prim_name not in requested_names
        ):
            continue

        matched_name = (
            external_name if external_name in requested_names else prim_name
        )
        found_names.add(matched_name)

        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(stiffness))
        drive.CreateDampingAttr(float(damping))
        drive.CreateMaxForceAttr(float(max_force))

        print(
            f"{matched_name:20s} path={prim.GetPath()} "
            f"stiffness={stiffness} damping={damping} "
            f"maxForce={max_force}"
        )

    missing_names = sorted(requested_names - found_names)
    if missing_names:
        print("WARNING: The following driven joints were not found:")
        for name in missing_names:
            print(f"  {name}")

    if not found_names:
        raise RuntimeError(
            "None of the requested driven joints were found in the imported USD"
        )


# =============================================================================
# ROS 2 Action Graph
# =============================================================================

def create_ros2_action_graph(articulation_root_path: str):
    """Subscribe to JointState position commands and control the articulation."""
    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                (
                    "SubscribeJointState",
                    "isaacsim.ros2.bridge.ROS2SubscribeJointState",
                ),
                (
                    "JointNameResolver",
                    "isaacsim.core.nodes.IsaacJointNameResolver",
                ),
                (
                    "ArticulationController",
                    "isaacsim.core.nodes.IsaacArticulationController",
                ),
                (
                    "ReadSimTime",
                    "isaacsim.core.nodes.IsaacReadSimulationTime",
                ),
                (
                    "PublishClock",
                    "isaacsim.ros2.bridge.ROS2PublishClock",
                ),
            ],
            og.Controller.Keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "SubscribeJointState.inputs:execIn",
                ),
                (
                    "SubscribeJointState.outputs:execOut",
                    "JointNameResolver.inputs:execIn",
                ),
                (
                    "JointNameResolver.outputs:execOut",
                    "ArticulationController.inputs:execIn",
                ),
                (
                    "Context.outputs:context",
                    "SubscribeJointState.inputs:context",
                ),
                (
                    "Context.outputs:context",
                    "PublishClock.inputs:context",
                ),
                (
                    "SubscribeJointState.outputs:jointNames",
                    "JointNameResolver.inputs:jointNames",
                ),
                (
                    "JointNameResolver.outputs:jointNames",
                    "ArticulationController.inputs:jointNames",
                ),
                (
                    "JointNameResolver.outputs:robotPath",
                    "ArticulationController.inputs:robotPath",
                ),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
                (
                    "OnPlaybackTick.outputs:tick",
                    "PublishClock.inputs:execIn",
                ),
                (
                    "ReadSimTime.outputs:simulationTime",
                    "PublishClock.inputs:timeStamp",
                ),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("Context.inputs:domain_id", ROS_DOMAIN_ID),
                ("Context.inputs:useDomainIDEnvVar", False),
                ("SubscribeJointState.inputs:topicName", TOPIC_NAME),
                (
                    "JointNameResolver.inputs:robotPath",
                    articulation_root_path,
                ),
                ("PublishClock.inputs:topicName", "/clock"),
            ],
        },
    )

    print("\n========== ROS 2 Action Graph created ==========")
    print(f"graph_path             = {GRAPH_PATH}")
    print(f"articulation_root_path = {articulation_root_path}")
    print(f"topic_name             = {TOPIC_NAME}")
    print(f"domain_id              = {ROS_DOMAIN_ID}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    enable_required_extensions()
    update_app(20)

    stage = create_new_stage()
    add_physics_scene(stage)
    add_floor(stage)
    add_pick_cube(stage)
    add_light(stage)

    robot_usd_path = generate_or_reuse_robot_usd(URDF_PATH)
    robot_reference_path = add_robot_reference(robot_usd_path)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Current USD stage is unavailable")

    articulation_root_path = find_articulation_root_path(
        stage,
        robot_reference_path,
    )

    # Restore strong position drives on every arm and gripper joint.
    set_usd_joint_drives(
        stage=stage,
        robot_reference_path=robot_reference_path,
        joint_names=DRIVEN_JOINT_NAMES,
    )

    create_ros2_action_graph(articulation_root_path)
    update_app(30)

    print("\nStarting simulation automatically...")
    print(f"Waiting for JointState position commands on {TOPIC_NAME}.")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    try:
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        timeline.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()