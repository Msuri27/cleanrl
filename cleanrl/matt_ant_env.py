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
        self.passed_inverted = False
        self.state =  "Takeoff"
        self.lb_counter = 0
        self.entered_flip = False
        self.entered_land = False

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
        weight_height = 2.0
        weight_z = 3.0
        weight_flip = 1.0
        weight_y = 0.05
        weight_pitch = 0.05
        weight_roll = 0.02
        weight_yaw = 0.02

        # phase transition bonuses
        launch_bonus = 5.0
        flip_bonus = 10.0
        landing_bonus = 5.0
        launch_height_gain = 0.25
        stable_landing_steps = 50

        # phase one
        z_height = float(observation[0])
        height_gain = z_height - self.start_z
        vertical_velocity = float(observation[15])
        penalize_off_axis_motion = self.state in ("Flip", "Land")

         # phase two and three
        w, qx, qy, qz = map(float, observation[1:5])
        torso_up = 1.0 - 2.0 * (qx * qx + qy * qy)
        pitch_velocity = float(observation[17])

        match self.state:
            case "Takeoff":
                r_takeoff = weight_z * max(vertical_velocity, 0.0) * self.dt
                base_reward += r_takeoff

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
                r_land = -weight_height * height_gain ** 2 * self.dt
                r_orientation = torso_up * self.dt
                base_reward += r_pitch + r_land + r_orientation

                stable = (
                    torso_up > 0.9
                    and abs(height_gain) < 0.1
                    and abs(vertical_velocity) < 0.1
                    and abs(float(observation[16])) < 0.2
                    and abs(pitch_velocity) < 0.2
                    and abs(float(observation[18])) < 0.2
                    and self._foot_contact_count() >= 2
                )

                if stable:
                    self.lb_counter += 1
                    if self.lb_counter == stable_landing_steps:
                        base_reward += landing_bonus
                        # state reset
                        self.state = "Takeoff"
                        self.start_z = z_height
                        self.passed_inverted = False
                        self.lb_counter = 0
                else:
                    self.lb_counter = 0

        if penalize_off_axis_motion:
            r_lateral = -weight_y * float(observation[14]) ** 2 * self.dt
            r_roll = -weight_roll * float(observation[16]) ** 2 * self.dt
            r_yaw = -weight_yaw * float(observation[18]) ** 2 * self.dt
            base_reward += r_lateral + r_roll + r_yaw

        info["entered_flip"] = self.entered_flip
        info["entered_land"] = self.entered_land

        return observation, base_reward, terminated, truncated, info
    
register(id="AntBackflip-v0",
         entry_point="matt_ant_env:AntBackflipEnv", 
         max_episode_steps=1000
)