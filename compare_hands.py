import sys
import math
import numpy as np
from isaacgym import gymapi, gymutil

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# -----------------------------------------------------------------------------
# Use two URDF files: one for each agent.
# -----------------------------------------------------------------------------
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments

asset_descriptors = [
    AssetDesc("allegro/allegro.urdf", False),
    AssetDesc("allegro/allegro_internal.urdf", False),
]

# -----------------------------------------------------------------------------
# Remove the asset_id argument since we are loading both assets.
# -----------------------------------------------------------------------------
args = gymutil.parse_arguments(
    description="Compare different URDFs (internal vs public)",
    custom_parameters=[
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"},
    ])

# -----------------------------------------------------------------------------
# Define a function to extract and compute DOF information from an asset.
# -----------------------------------------------------------------------------
def get_dof_info(asset, speed_scale, asset_name):
    num_dofs = gym.get_asset_dof_count(asset)
    dof_names = gym.get_asset_dof_names(asset)
    dof_props = gym.get_asset_dof_properties(asset)
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]
    dof_positions = dof_states['pos']

    stiffnesses = dof_props['stiffness']
    dampings = dof_props['damping']
    armatures = dof_props['armature']
    has_limits = dof_props['hasLimits']
    # Copy limits so we can modify them
    lower_limits = dof_props['lower'].copy()
    upper_limits = dof_props['upper'].copy()

    defaults = np.zeros(num_dofs)
    speeds = np.zeros(num_dofs)
    for i in range(num_dofs):
        if has_limits[i]:
            if dof_types[i] == gymapi.DOF_ROTATION:
                lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
                upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
            if lower_limits[i] > 0.0:
                defaults[i] = lower_limits[i]
            elif upper_limits[i] < 0.0:
                defaults[i] = upper_limits[i]
        else:
            if dof_types[i] == gymapi.DOF_ROTATION:
                lower_limits[i] = -math.pi
                upper_limits[i] = math.pi
            elif dof_types[i] == gymapi.DOF_TRANSLATION:
                lower_limits[i] = -1.0
                upper_limits[i] = 1.0
        dof_positions[i] = defaults[i]
        if dof_types[i] == gymapi.DOF_ROTATION:
            speeds[i] = speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]),
                                            0.25 * math.pi, 3.0 * math.pi)
        else:
            speeds[i] = speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]),
                                            0.1, 7.0)

    # Print DOF properties
    for i in range(num_dofs):
        print(f"Asset {asset_name} DOF %d" % i)
        print("  Name:     '%s'" % dof_names[i])
        print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
        print("  Stiffness:  %r" % stiffnesses[i])
        print("  Damping:    %r" % dampings[i])
        print("  Armature:   %r" % armatures[i])
        print("  Limited?    %r" % has_limits[i])
        if has_limits[i]:
            print("    Lower:   %f" % lower_limits[i])
            print("    Upper:   %f" % upper_limits[i])

    return {
        'num_dofs': num_dofs,
        'dof_names': dof_names,
        'dof_props': dof_props,
        'dof_states': dof_states,
        'dof_types': dof_types,
        'dof_positions': dof_positions,
        'stiffnesses': stiffnesses,
        'dampings': dampings,
        'armatures': armatures,
        'has_limits': has_limits,
        'lower_limits': lower_limits,
        'upper_limits': upper_limits,
        'defaults': defaults,
        'speeds': speeds
    }

# -----------------------------------------------------------------------------
# Initialize gym and simulation.
# -----------------------------------------------------------------------------
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# Add a ground plane.
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
gym.add_ground(sim, plane_params)

# Create viewer.
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# -----------------------------------------------------------------------------
# Load the two assets separately.
# -----------------------------------------------------------------------------
asset_root = "./assets/"

# Load asset for agent 1 (using the first URDF)
asset_options0 = gymapi.AssetOptions()
asset_options0.fix_base_link = True
asset_options0.flip_visual_attachments = asset_descriptors[0].flip_visual_attachments
asset_options0.use_mesh_materials = True
print("Loading asset '%s' from '%s'" % (asset_descriptors[0].file_name, asset_root))
asset0 = gym.load_asset(sim, asset_root, asset_descriptors[0].file_name, asset_options0)

# Load asset for agent 2 (using the second URDF)
asset_options1 = gymapi.AssetOptions()
asset_options1.fix_base_link = True
asset_options1.flip_visual_attachments = asset_descriptors[1].flip_visual_attachments
asset_options1.use_mesh_materials = True
print("Loading asset '%s' from '%s'" % (asset_descriptors[1].file_name, asset_root))
asset1 = gym.load_asset(sim, asset_root, asset_descriptors[1].file_name, asset_options1)

# -----------------------------------------------------------------------------
# Obtain DOF information for each asset using our function.
# -----------------------------------------------------------------------------
dof_info0 = get_dof_info(asset0, args.speed_scale, asset_name = asset_descriptors[0].file_name)
dof_info1 = get_dof_info(asset1, args.speed_scale, asset_name = asset_descriptors[1].file_name)

# If the two assets have different DOF counts, stop the program.
if dof_info0['num_dofs'] != dof_info1['num_dofs']:
    print("Error: The two assets have different DOF counts: %d vs %d" % (dof_info0['num_dofs'], dof_info1['num_dofs']))
    sys.exit(1)

# -----------------------------------------------------------------------------
# Create one environment and add exactly 2 actors—each using a different asset.
# Place them 2 meters apart (here one at (-1,0,0.5) and the other at (1,0,0.5)).
# -----------------------------------------------------------------------------
spacing   = 1
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
envs = []
actor_handles = []

print("Creating 1 environment with 2 actors (2 meters apart)")
env = gym.create_env(sim, env_lower, env_upper, 1)
envs.append(env)

# Create the first actor (agent1) using asset0.
pose1 = gymapi.Transform()
pose1.p = gymapi.Vec3(0.0, 0.0, 0.5)
actor_handle1 = gym.create_actor(env, asset0, pose1, "agent1", 0, 1)
actor_handles.append(actor_handle1)

# Set color for agent1 (blue)
num_bodies1 = gym.get_actor_rigid_body_count(env, actor_handle1)
for i in range(num_bodies1):
    gym.set_rigid_body_color(env, actor_handle1, i, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.5, 1.0))

# Create the second actor (agent2) using asset1.
pose2 = gymapi.Transform()
pose2.p = gymapi.Vec3(0.0, 0.0, 0.5)
actor_handle2 = gym.create_actor(env, asset1, pose2, "agent2", 0, 1)
actor_handles.append(actor_handle2)

# Set color for agent2 (orange)
num_bodies2 = gym.get_actor_rigid_body_count(env, actor_handle2)
for i in range(num_bodies2):
    gym.set_rigid_body_color(env, actor_handle2, i, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.5, 0.2))

# Set default DOF states for both actors.
gym.set_actor_dof_states(env, actor_handle1, dof_info0['dof_states'], gymapi.STATE_ALL)
gym.set_actor_dof_states(env, actor_handle2, dof_info1['dof_states'], gymapi.STATE_ALL)

# Position the camera.
cam_pos    = gymapi.Vec3(0.5, -0.5, 0.8)
cam_target = gymapi.Vec3(0, 0, 0.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# -----------------------------------------------------------------------------
# Joint animation state constants and initialization.
# -----------------------------------------------------------------------------
ANIM_SEEK_LOWER   = 1
ANIM_SEEK_UPPER   = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED     = 4

# For agent1 (asset0)
anim_state0 = ANIM_SEEK_LOWER
current_dof0 = 0
print("Animating asset0 DOF %d ('%s')" % (current_dof0, dof_info0['dof_names'][current_dof0]))

# For agent2 (asset1)
anim_state1 = ANIM_SEEK_LOWER
current_dof1 = 0
print("Animating asset1 DOF %d ('%s')" % (current_dof1, dof_info1['dof_names'][current_dof1]))

# -----------------------------------------------------------------------------
# Simulation loop.
# -----------------------------------------------------------------------------
while not gym.query_viewer_has_closed(viewer):

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Update agent1 (asset0)
    speed0 = dof_info0['speeds'][current_dof0]
    if anim_state0 == ANIM_SEEK_LOWER:
        dof_info0['dof_positions'][current_dof0] -= speed0 * dt
        if dof_info0['dof_positions'][current_dof0] <= dof_info0['lower_limits'][current_dof0]:
            dof_info0['dof_positions'][current_dof0] = dof_info0['lower_limits'][current_dof0]
            anim_state0 = ANIM_SEEK_UPPER
    elif anim_state0 == ANIM_SEEK_UPPER:
        dof_info0['dof_positions'][current_dof0] += speed0 * dt
        if dof_info0['dof_positions'][current_dof0] >= dof_info0['upper_limits'][current_dof0]:
            dof_info0['dof_positions'][current_dof0] = dof_info0['upper_limits'][current_dof0]
            anim_state0 = ANIM_SEEK_DEFAULT
    elif anim_state0 == ANIM_SEEK_DEFAULT:
        dof_info0['dof_positions'][current_dof0] -= speed0 * dt
        if dof_info0['dof_positions'][current_dof0] <= dof_info0['defaults'][current_dof0]:
            dof_info0['dof_positions'][current_dof0] = dof_info0['defaults'][current_dof0]
            anim_state0 = ANIM_FINISHED
    elif anim_state0 == ANIM_FINISHED:
        dof_info0['dof_positions'][current_dof0] = dof_info0['defaults'][current_dof0]
        current_dof0 = (current_dof0 + 1) % dof_info0['num_dofs']
        anim_state0 = ANIM_SEEK_LOWER
        print("Animating asset0 DOF %d ('%s')" % (current_dof0, dof_info0['dof_names'][current_dof0]))

    # Update agent2 (asset1)
    speed1 = dof_info1['speeds'][current_dof1]
    if anim_state1 == ANIM_SEEK_LOWER:
        dof_info1['dof_positions'][current_dof1] -= speed1 * dt
        if dof_info1['dof_positions'][current_dof1] <= dof_info1['lower_limits'][current_dof1]:
            dof_info1['dof_positions'][current_dof1] = dof_info1['lower_limits'][current_dof1]
            anim_state1 = ANIM_SEEK_UPPER
    elif anim_state1 == ANIM_SEEK_UPPER:
        dof_info1['dof_positions'][current_dof1] += speed1 * dt
        if dof_info1['dof_positions'][current_dof1] >= dof_info1['upper_limits'][current_dof1]:
            dof_info1['dof_positions'][current_dof1] = dof_info1['upper_limits'][current_dof1]
            anim_state1 = ANIM_SEEK_DEFAULT
    elif anim_state1 == ANIM_SEEK_DEFAULT:
        dof_info1['dof_positions'][current_dof1] -= speed1 * dt
        if dof_info1['dof_positions'][current_dof1] <= dof_info1['defaults'][current_dof1]:
            dof_info1['dof_positions'][current_dof1] = dof_info1['defaults'][current_dof1]
            anim_state1 = ANIM_FINISHED
    elif anim_state1 == ANIM_FINISHED:
        dof_info1['dof_positions'][current_dof1] = dof_info1['defaults'][current_dof1]
        current_dof1 = (current_dof1 + 1) % dof_info1['num_dofs']
        anim_state1 = ANIM_SEEK_LOWER
        print("Animating asset1 DOF %d ('%s')" % (current_dof1, dof_info1['dof_names'][current_dof1]))

    if args.show_axis:
        gym.clear_lines(viewer)

    # Update the DOF states for both actors.
    gym.set_actor_dof_states(env, actor_handle1, dof_info0['dof_states'], gymapi.STATE_POS)
    gym.set_actor_dof_states(env, actor_handle2, dof_info1['dof_states'], gymapi.STATE_POS)

    if args.show_axis:
        # Draw DOF axes for agent1.
        dof_handle0 = gym.get_actor_dof_handle(env, actor_handle1, current_dof0)
        frame0 = gym.get_dof_frame(env, dof_handle0)
        p1 = frame0.origin
        p2 = frame0.origin + frame0.axis * 0.7
        color = gymapi.Vec3(1.0, 0.0, 0.0)
        gymutil.draw_line(p1, p2, color, gym, viewer, env)
        # Draw DOF axes for agent2.
        dof_handle1 = gym.get_actor_dof_handle(env, actor_handle2, current_dof1)
        frame1 = gym.get_dof_frame(env, dof_handle1)
        p1 = frame1.origin
        p2 = frame1.origin + frame1.axis * 0.7
        gymutil.draw_line(p1, p2, color, gym, viewer, env)

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
