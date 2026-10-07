import math
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.envs.mujoco.ant_v4 import AntEnv

class AntBackflipEnv(AntEnv):
    def __init__(self, **kwargs):
        super().__init__(terminate_when_unhealthy=False, **kwargs)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        
        self.start_z = float(observation[0])
        self.episode_initial_z = self.start_z
        self.episode_max_height_gain = 0.0
        self.passed_inverted = False
        self.state =  "Takeoff"
        self.lb_counter = 0
        self.hold_counter = 0
        self.entered_flip = False
        self.entered_land = False
        self.entered_hold = False

        return observation, info

    def _foot_contact_count(self) -> int:
        floor_id = self.model.geom("floor").id
        foot_ids = {
            self.model.geom(name).id
            for name in (
                "left_ankle_geom",
                "right_ankle_geom",
                "third_ankle_geom",
                "fourth_ankle_geom",
            )
        }

        touching_feet = set()
        for contact in self.data.contact[:self.data.ncon]:
            if contact.geom1 == floor_id and contact.geom2 in foot_ids:
                touching_feet.add(contact.geom2)
            elif contact.geom2 == floor_id and contact.geom1 in foot_ids:
                touching_feet.add(contact.geom1)

        return len(touching_feet)

    def step(self, action):
        observation, base_reward, terminated, truncated, info = super().step(action)

        # replace ant default reward
        base_reward = 0.0

        # reward rates are multiplied by self.dt so their scale is less dependent
        # on the environment's control frequency.
        weight_hold_height = 2.0
        weight_z = 3.0
        weight_flip = 1.0
        weight_y = 0.05
        weight_pitch = 0.05
        weight_roll = 0.02
        weight_yaw = 0.02
        weight_takeoff_orientation = 0.5
        weight_stable = 0.5
        weight_hold_velocity = 0.05

        # phase transition bonuses
        launch_bonus = 5.0
        flip_bonus = 6.0
        landing_bonus = 5.0
        launch_height_gain = 0.3
        stable_landing_steps = 50
        hold_steps = 100

        # phase one
        z_height = float(observation[0])
        height_gain = z_height - self.start_z
        self.episode_max_height_gain = max(
            self.episode_max_height_gain, z_height - self.episode_initial_z
        )
        info["episode_max_height_gain"] = self.episode_max_height_gain
        vertical_velocity = float(observation[15])
        penalize_off_axis_motion = self.state in ("Flip", "Land")

         # phase two and three
        w, qx, qy, qz = map(float, observation[1:5])
        torso_up = 1.0 - 2.0 * (qx * qx + qy * qy)
        pitch_velocity = float(observation[17])

        stable = False
        if self.state in ("Land", "Hold"):
            stable = (
                torso_up > 0.9
                and abs(height_gain) < 0.1
                and abs(vertical_velocity) < 0.1
                and abs(float(observation[16])) < 0.2
                and abs(pitch_velocity) < 0.2
                and abs(float(observation[18])) < 0.2
                and self._foot_contact_count() >= 3
            )

        match self.state:
            case "Takeoff":
                r_takeoff = weight_z * max(vertical_velocity, 0.0) * self.dt
                r_takeoff_orientation = weight_takeoff_orientation * torso_up * self.dt
                base_reward += r_takeoff + r_takeoff_orientation

                if height_gain >= launch_height_gain and vertical_velocity > 0.0:
                    base_reward += launch_bonus 
                    self.state = "Flip"
                    self.entered_flip = True
            case "Flip":
                r_flip = weight_flip * max(-pitch_velocity, 0.0) * self.dt
                base_reward += r_flip
            
                if torso_up < 0.0:
                    self.passed_inverted = True 
                if self.passed_inverted and torso_up > 0.8:
                    base_reward += flip_bonus
                    self.state = "Land"
                    self.entered_land = True
            case "Land":
                r_pitch = -weight_pitch * pitch_velocity ** 2 * self.dt
                r_orientation = torso_up * self.dt
                base_reward += r_pitch + r_orientation

                if stable:
                    self.lb_counter += 1
                    base_reward += weight_stable * self.dt
                    if self.lb_counter == stable_landing_steps:
                        base_reward += landing_bonus
                        self.state = "Hold"
                        self.entered_hold = True
                        self.start_z = z_height
                        self.passed_inverted = False
                        self.lb_counter = 0
                        self.hold_counter = 0
                else:
                    self.lb_counter = 0
            case "Hold":
                linear_velocity_sq = sum(float(v) ** 2 for v in observation[13:16])
                angular_velocity_sq = sum(float(v) ** 2 for v in observation[16:19])
                r_hold_velocity = -weight_hold_velocity * (
                    linear_velocity_sq + angular_velocity_sq
                ) * self.dt
                r_hold_pose = (
                    torso_up * self.dt - weight_hold_height * height_gain ** 2 * self.dt
                )
                base_reward += r_hold_velocity + r_hold_pose
                if stable:
                    base_reward += weight_stable * self.dt

                self.hold_counter += 1
                if self.hold_counter >= hold_steps:
                    self.state = "Takeoff"
                    self.start_z = z_height
                    self.passed_inverted = False
                    self.hold_counter = 0

        if penalize_off_axis_motion:
            r_lateral = -weight_y * float(observation[14]) ** 2 * self.dt
            r_roll = -weight_roll * float(observation[16]) ** 2 * self.dt
            r_yaw = -weight_yaw * float(observation[18]) ** 2 * self.dt
            base_reward += r_lateral + r_roll + r_yaw

        info["entered_flip"] = self.entered_flip
        info["entered_land"] = self.entered_land
        info["entered_hold"] = self.entered_hold

        return observation, base_reward, terminated, truncated, info
    
register(id="AntBackflip-v0",
         entry_point="matt_ant_env:AntBackflipEnv", 
         max_episode_steps=1000
)