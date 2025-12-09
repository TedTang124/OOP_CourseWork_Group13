import pybullet as p
import pybullet_data
import pandas as pd
import time
import math
import random
from abc import ABC, abstractmethod

# ---- Abstract Class ----
class SceneObject(ABC):
    def __init__(self, urdf_file, position, orientation=(0, 0, 0)):
        self.urdf_file = urdf_file
        self.position = position
        self.orientation = p.getQuaternionFromEuler(orientation)
        self.id = None
    @abstractmethod
    def load(self):
        pass

class Grippers(ABC):
    def __init__(self, urdf_file, position, orientation=(0, 0, 0)):
        self.urdf_file = urdf_file
        self.position = position
        self.orientation = p.getQuaternionFromEuler(orientation)
        self.id = None
    @abstractmethod
    def load(self):
        pass
    @abstractmethod
    def move_pickup(self):
        pass
    @abstractmethod
    def open_gripper(self):
        pass
    @abstractmethod
    def close_gripper(self):
        pass

# ---- Class ----
class Cube(SceneObject):
    def __init__(self, position, orientation=(0, 0, 0)):
        super().__init__("cube_small.urdf", position, orientation)

    def load(self):
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation)
        return self.id
    
class Cylinder(SceneObject):
    def __init__(self, position, orientation=(0, 0, 0)):
        super().__init__("cylinder.urdf", position, orientation)

    def load(self):
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation)
        return self.id

class TwoFingersGripper(Grippers):
    def __init__(self, position, orientation=(0, 0, 0)):
        super().__init__("pr2_gripper.urdf", position, orientation)

    def load(self):
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation)
        return self.id
    
    def attach_fixed(self, offset):
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId = self.id,
            parentLinkIndex = -1,
            childBodyUniqueId = -1,
            childLinkIndex = -1,
            jointType = p.JOINT_FIXED,
            jointAxis = [0, 0, 0],
            parentFramePosition = offset,
            childFramePosition = self.position,
            parentFrameOrientation = [0,0,0,1],
            childFrameOrientation = self.orientation
        )

    def move_pickup(self, obj_position):
        dx = obj_position[0] - self.position[0]
        dy = obj_position[1] - self.position[1]
        dz = obj_position[2] - self.position[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        steps = int(dist/ 0.02)

        for step in range(steps):
            alpha = (step + 1) / steps
            new_x = self.position[0] + dx * alpha
            new_y = self.position[1] + dy * alpha
            new_z = self.position[2] + dz * alpha
            p.changeConstraint(self.constraint_id, 
                           jointChildPivot=[new_x, new_y, new_z],
                           maxForce=100)
            p.stepSimulation()
            time.sleep(1./240.)

            # check distance from new end-effector to object
            cur_dx = obj_position[0] - new_x
            cur_dy = obj_position[1] - new_y
            cur_dz = obj_position[2] - new_z
            cur_dist = math.sqrt(cur_dx*cur_dx + cur_dy*cur_dy + cur_dz*cur_dz)
            if cur_dist < 0.31:
                break

        self.position = [new_x, new_y, new_z]
        self.close_gripper()
        for _ in range(30):
            p.stepSimulation()
            time.sleep(1./240.)

    def move_up(self, obj_id, new_z=1, duration=2.0, no_contact_timeout=0.5,
                contact_ratio_threshold=0.99):
        sim_hz = 50
        total_steps = max(1, int(duration * sim_hz))
        timeout_steps = max(1, int(no_contact_timeout * sim_hz))

        contact_steps = 0
        total_contact_checks = 0
        consecutive_no_contact_steps = 0

        start_z = self.position[2]
        dz = (new_z - start_z) / float(total_steps)

        for step_idx in range(total_steps):
            self.position[2] += dz
            p.changeConstraint(self.constraint_id,
                            jointChildPivot=self.position,
                            maxForce=100)

            # check contact
            contacts = p.getContactPoints(bodyA=self.id, bodyB=obj_id) or []
            has_contact = len(contacts) > 0

            if has_contact:
                contact_steps += 1
                consecutive_no_contact_steps = 0
            else:
                consecutive_no_contact_steps += 1
            total_contact_checks += 1

            # --- EARLY ABORT RULE ---
            if consecutive_no_contact_steps >= timeout_steps:
                print(f"[move_up] Lost contact for {no_contact_timeout} seconds (consecutive_no_contact_steps={consecutive_no_contact_steps}) → aborting and returning False")
                return False

            # step simulation
            p.stepSimulation()
            time.sleep(1./240.)

        # finish lift, compute ratio for final decision
        contact_ratio = contact_steps / float(max(1, total_contact_checks))
        print(f"[move_up] contact_ratio={contact_ratio:.2f}")
        return contact_ratio >= contact_ratio_threshold

    def open_gripper(self):
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.55, maxVelocity=10, force=40)

    def close_gripper(self):
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0, maxVelocity=2.7, force=200)

class Sample():
    current_sample = 0

    def __init__(self, obj_center, radius, samples=5):
        self.samples = samples
        self.obj_center = obj_center
        self.radius = radius
        self.sample_name = None
        Sample.current_sample += 1
    
    def sample_grasp_pose(self):
        positions = []
        orientations = []

        for _ in range(self.samples):
            # Sample on sphere surface around the object
            theta = random.uniform(0, 2 * math.pi)  # Azimuth
            phi = random.uniform(0, math.pi/2)        # Polar
            
            x = self.radius * math.sin(phi) * math.cos(theta) + self.obj_center[0]
            y = self.radius * math.sin(phi) * math.sin(theta) + self.obj_center[1]
            z = self.radius * math.cos(phi) + self.obj_center[2]
            position = [x, y, z]
            positions.append(position)

            dx = self.obj_center[0] - x
            dy = self.obj_center[1] - y
            dz = self.obj_center[2] - z
            dist_xy = math.sqrt(dx*dx + dy*dy)
            yaw = math.atan2(dy, dx)
            pitch = -math.atan2(dz, dist_xy)
            roll  = random.uniform(-math.pi, math.pi)

            orientation = [roll, pitch, yaw]
            orientations.append(orientation)

        samples_po_orein = pd.DataFrame({
            "positions": positions,
            "orientations": orientations})
        
        return samples_po_orein  # 6D pose

# ---- Functions ----
def relative_position(gripper_id, obj_id):
    g_pos, g_orn = p.getBasePositionAndOrientation(gripper_id)
    o_pos, o_orn = p.getBasePositionAndOrientation(obj_id)

    # Convert to relative
    inv_o_pos, inv_o_orn = p.invertTransform(o_pos, o_orn)
    rel_pos, rel_orn = p.multiplyTransforms(inv_o_pos, inv_o_orn, g_pos, g_orn)

    # Convert to Euler
    rel_roll, rel_pitch, rel_yaw = p.getEulerFromQuaternion(rel_orn)
    return rel_pos, (rel_roll, rel_pitch, rel_yaw)

def setup_environment():
    cid = p.connect(p.GUI)
    print("Connection ID:", cid)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)

    plane_id = p.loadURDF("plane.urdf")

    p.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=40,
        cameraPitch=-30,
        cameraTargetPosition=[0.6, 0.3, 0.2]
    )
    return plane_id

# ---- MAIN ----
setup_environment()

# Generate Cube
cube_pos = [0.6, 0.3, 0.025]
graspable = []
rel_x = []
rel_y = []
rel_z = []
rel_roll = []
rel_pitch = []
rel_yaw = []
obj1 = Cube(cube_pos)
obj1_id = obj1.load()

# Sample grasp poses around the object
sampler = Sample(obj_center=[0.6, 0.3, 0.025], radius=1, samples=50)
df_samples = sampler.sample_grasp_pose()

# --- Test a single grasp pose ---
#first_pos = df_samples['positions'][0]
#first_orn_euler = df_samples['orientations'][0]
#gripper = TwoFingersGripper(position=first_pos, orientation=first_orn_euler)
#gripper.load()
#gripper.attach_fixed(offset=[0,0,0])
#gripper.open_gripper()
#for _ in range(25):
#        p.stepSimulation()
#        time.sleep(1./240.)
#gripper.move_pickup(obj_position=cube_pos)
#condition = gripper.move_up(obj1_id)
#graspable.append(condition)

for i in range(50):
    pos = df_samples["positions"][i]
    orn = df_samples["orientations"][i]

    # Create gripper instance
    gripper = TwoFingersGripper(position=pos, orientation=orn)
    current_gid = gripper.load()
    gripper.attach_fixed(offset=[0,0,0])
    gripper.open_gripper()
    
    # wait for fingers to open
    for _ in range(25):
        p.stepSimulation()
        time.sleep(1./240.)

    # record relative pose
    rel_pos, rel_orn = relative_position(current_gid, obj1_id)
    rel_x.append(rel_pos[0])
    rel_y.append(rel_pos[1])
    rel_z.append(rel_pos[2])
    rel_roll.append(rel_orn[0])
    rel_pitch.append(rel_orn[1])
    rel_yaw.append(rel_orn[2])

    # attempt grasp
    gripper.move_pickup(obj_position=cube_pos)
    result = gripper.move_up(obj1_id)

    print(result)
    graspable.append(result)
    p.removeBody(current_gid)

    # reset object position before next grasp attempt
    p.resetBasePositionAndOrientation(obj1_id, cube_pos, [0,0,0,1])
    for _ in range(40):
        p.stepSimulation()
        time.sleep(1./240.)

df_real = pd.DataFrame({
    "rel_x": rel_x,
    "rel_y": rel_y,
    "rel_z": rel_z,
    "rel_roll": rel_roll,
    "rel_pitch": rel_pitch,
    "rel_yaw": rel_yaw,
    "success": graspable})
print(df_real)
df_real.to_csv("grasp_dataset.csv", index=False)