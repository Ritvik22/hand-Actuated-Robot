from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class SimObservation:
    ee_position: np.ndarray
    joint_positions: np.ndarray


class RobotHandSimulator:
    def __init__(self, timestep: float = 1.0 / 120.0, gui: bool = True):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(timestep)
        self.timestep = timestep

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        self.end_effector_link = 6

        self.objects = []
        self._spawn_objects()

        self.arm_joint_ids = [i for i in range(p.getNumJoints(self.robot))]
        self.gripper_proxy = np.zeros(10, dtype=np.float32)

    def _spawn_objects(self) -> None:
        self.objects.clear()
        positions = [[0.6, 0.0, 0.05], [0.7, 0.1, 0.05], [0.7, -0.1, 0.05]]
        for pos in positions:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], rgbaColor=[0.2, 0.6, 0.8, 1.0])
            obj = p.createMultiBody(baseMass=0.2, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
            self.objects.append(obj)

    def reset(self) -> None:
        for j in self.arm_joint_ids:
            p.resetJointState(self.robot, j, 0.0)
        for obj in self.objects:
            p.removeBody(obj)
        self._spawn_objects()

    def apply_action(self, translation: np.ndarray, finger_angles: np.ndarray) -> None:
        x = 0.4 + (translation[0] - 0.5) * 0.6
        y = (translation[1] - 0.5) * 0.8
        z = 0.2 + np.clip(-translation[2], -0.2, 0.2)

        target_pos = np.array([x, y, z], dtype=np.float32)
        target_orn = p.getQuaternionFromEuler([0, np.pi, 0])
        joint_targets = p.calculateInverseKinematics(self.robot, self.end_effector_link, target_pos, target_orn)

        for j, tgt in zip(self.arm_joint_ids, joint_targets):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=tgt, force=300)

        self.gripper_proxy = np.asarray(finger_angles, dtype=np.float32)

    def step(self, n: int = 4) -> None:
        for _ in range(n):
            p.stepSimulation()

    def get_observation(self) -> SimObservation:
        ee_state = p.getLinkState(self.robot, self.end_effector_link)
        joints = [p.getJointState(self.robot, j)[0] for j in self.arm_joint_ids]
        return SimObservation(ee_position=np.array(ee_state[0], dtype=np.float32), joint_positions=np.array(joints, dtype=np.float32))

    def close(self) -> None:
        p.disconnect(self.client)
