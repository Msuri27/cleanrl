import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from gymnasium.envs.mujoco.ant_v4 import AntEnv

class AntBackflipEnv(AntEnv):
    phases = ("Takeoff", "Flip", "Land", "Hold")

    def __init__(self, **kwargs):
        super().__init__(terminate_when_unhealthy=False, **kwargs)

    def _augment_observation(self, observation):
        phase = np.zeros(len(self.phases), dtype=observation.dtype)
        phase[self.phases.index(self.state)] = 1.0
        land_elapsed_fraction = min(self.land_elapsed_steps * self.dt / 10.0, 1.0)
        return np.concatenate((observation, phase, [land_elapsed_fraction]))

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        
        self.start_z = float(observation[0])
        self.episode_initial_z = self.start_z
        self.episode_max_height_gain = 0.0
        self.takeoff_peak_height_gain = 0.0
        self.passed_inverted = False
        self.state =  "Takeoff"
        self.land_elapsed_steps = 0
        self.episode_max_land_elapsed_steps = 0
        self.lb_counter = 0
        self.hold_counter = 0
        self.hold_stable_counter = 0
        self.episode_max_stable_hold_steps = 0
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
        weight_land_delay = 1.5
        weight_hold_velocity = 0.05

        # phase transition bonuses
        launch_bonus = 5.0
        flip_bonus = 6.0
        landing_bonus = 5.0
        launch_height_gain = 0.3
        stable_landing_steps = 5
        hold_steps = 40

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
        qx, qy = map(float, observation[2:4])
        torso_up = 1.0 - 2.0 * (qx * qx + qy * qy)
        pitch_velocity = float(observation[17])
        takeoff_completed = False
        hold_entered_this_step = False
        land_to_hold_seconds = 0.0

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
                self.takeoff_peak_height_gain = max(
                    self.takeoff_peak_height_gain, height_gain
                )
                r_takeoff = weight_z * max(vertical_velocity, 0.0) * self.dt
                r_takeoff_orientation = weight_takeoff_orientation * torso_up * self.dt
                base_reward += r_takeoff + r_takeoff_orientation

                if height_gain >= launch_height_gain and vertical_velocity > 0.0:
                    base_reward += launch_bonus 
                    self.state = "Flip"
                    self.entered_flip = True
                    takeoff_completed = True
            case "Flip":
                r_flip = weight_flip * max(-pitch_velocity, 0.0) * self.dt
                base_reward += r_flip
            
                if torso_up < 0.0:
                    self.passed_inverted = True 
                if self.passed_inverted and torso_up > 0.8:
                    base_reward += flip_bonus
                    self.state = "Land"
                    self.entered_land = True
                    self.land_elapsed_steps = 0
            case "Land":
                self.land_elapsed_steps += 1
                self.episode_max_land_elapsed_steps = max(
                    self.episode_max_land_elapsed_steps, self.land_elapsed_steps
                )
                base_reward -= weight_land_delay * self.dt
                r_pitch = -weight_pitch * pitch_velocity ** 2 * self.dt
                r_orientation = torso_up * self.dt
                base_reward += r_pitch + r_orientation

                if stable:
                    self.lb_counter += 1
                    base_reward += weight_stable * self.dt
                    if self.lb_counter == stable_landing_steps:
                        base_reward += landing_bonus
                        hold_entered_this_step = True
                        land_to_hold_seconds = self.land_elapsed_steps * self.dt
                        self.state = "Hold"
                        self.entered_hold = True
                        self.start_z = z_height
                        self.passed_inverted = False
                        self.lb_counter = 0
                        self.land_elapsed_steps = 0
                        self.hold_counter = 0
                        self.hold_stable_counter = 0
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
                    self.hold_stable_counter += 1
                    self.episode_max_stable_hold_steps = max(
                        self.episode_max_stable_hold_steps, self.hold_stable_counter
                    )
                else:
                    self.hold_stable_counter = 0

                self.hold_counter += 1
                if self.hold_counter >= hold_steps:
                    self.state = "Takeoff"
                    self.start_z = z_height
                    self.takeoff_peak_height_gain = 0.0
                    self.passed_inverted = False
                    self.hold_counter = 0
                    self.hold_stable_counter = 0

        if penalize_off_axis_motion:
            r_lateral = -weight_y * float(observation[14]) ** 2 * self.dt
            r_roll = -weight_roll * float(observation[16]) ** 2 * self.dt
            r_yaw = -weight_yaw * float(observation[18]) ** 2 * self.dt
            base_reward += r_lateral + r_roll + r_yaw

        info["entered_flip"] = self.entered_flip
        info["entered_land"] = self.entered_land
        info["entered_hold"] = self.entered_hold
        info["episode_max_stable_hold_steps"] = self.episode_max_stable_hold_steps
        info["episode_max_land_elapsed_seconds"] = (
            self.episode_max_land_elapsed_steps * self.dt
        )
        info["takeoff_completed"] = takeoff_completed
        info["takeoff_peak_height_gain"] = self.takeoff_peak_height_gain
        info["hold_entered_this_step"] = hold_entered_this_step
        info["land_to_hold_seconds"] = land_to_hold_seconds

        return observation, base_reward, terminated, truncated, info


class AntPhaseObservation(gym.ObservationWrapper):
    def __init__(self, env: AntBackflipEnv):
        super().__init__(env)
        base_space = env.observation_space
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError("AntBackflipEnv must use a Box observation space")

        self.ant_env = env
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((base_space.low, np.zeros(5, dtype=base_space.dtype))),
            high=np.concatenate((base_space.high, np.ones(5, dtype=base_space.dtype))),
            dtype=np.float64,
        )

    def observation(self, observation):
        return self.ant_env._augment_observation(observation)


def make_ant_backflip_env(**kwargs):
    return AntPhaseObservation(AntBackflipEnv(**kwargs))


register(id="AntBackflip-v0",
         entry_point="matt_ant_env:make_ant_backflip_env",
         max_episode_steps=1000
)