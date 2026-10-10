import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from gymnasium.envs.mujoco.ant_v4 import AntEnv

class AntBackflipEnv(AntEnv):
    phases = ("Takeoff", "Flip", "Land")

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
        self.three_foot_stable_steps = 0
        self.two_foot_stable_steps = 0
        self.completed_backflips = 0
        self.timely_landings = 0
        self.high_quality_landings = 0
        self.two_foot_landings = 0
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
        weight_z = 3.0
        weight_flip = 1.0
        weight_y = 0.025
        weight_pitch = 0.05
        weight_roll = 0.01
        weight_yaw = 0.01
        weight_takeoff_orientation = 0.5
        weight_active_phase_delay = 0.25
        weight_stable = 0.5
        weight_land_delay = 1.5

        # phase transition bonuses
        launch_bonus = 5.0
        flip_bonus = 6.0
        landing_bonus = 5.0
        two_foot_landing_bonus = 2.0
        launch_height_gain = 1.0
        three_foot_stability_steps = round(0.5 / self.dt)
        two_foot_stability_steps = round(1.0 / self.dt)
        successful_landing_deadline = 2.0

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
        stable_landing_this_step = False
        land_to_stable_seconds = 0.0

        stable_pose = False
        foot_contacts = 0
        if self.state == "Land":
            stable_pose = (
                torso_up > 0.9
                and abs(height_gain) < 0.1
                and abs(vertical_velocity) < 0.1
                and abs(float(observation[16])) < 0.2
                and abs(pitch_velocity) < 0.2
                and abs(float(observation[18])) < 0.2
            )
            foot_contacts = self._foot_contact_count()

        match self.state:
            case "Takeoff":
                self.takeoff_peak_height_gain = max(
                    self.takeoff_peak_height_gain, height_gain
                )
                base_reward -= weight_active_phase_delay * self.dt
                r_takeoff = weight_z * max(vertical_velocity, 0.0) * self.dt
                r_takeoff_orientation = (
                    weight_takeoff_orientation * (torso_up - 1.0) * self.dt
                )
                base_reward += r_takeoff + r_takeoff_orientation

                if height_gain >= launch_height_gain and vertical_velocity > 0.0:
                    base_reward += launch_bonus 
                    self.state = "Flip"
                    self.entered_flip = True
                    takeoff_completed = True
            case "Flip":
                base_reward -= weight_active_phase_delay * self.dt
                r_flip = weight_flip * max(-pitch_velocity, 0.0) * self.dt
                base_reward += r_flip
            
                if torso_up < 0.0:
                    self.passed_inverted = True 
                if self.passed_inverted and torso_up > 0.8:
                    base_reward += flip_bonus
                    self.state = "Land"
                    self.entered_land = True
                    self.completed_backflips += 1
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

                if stable_pose and foot_contacts >= 3:
                    self.three_foot_stable_steps += 1
                else:
                    self.three_foot_stable_steps = 0

                if stable_pose and foot_contacts >= 2:
                    self.two_foot_stable_steps += 1
                else:
                    self.two_foot_stable_steps = 0

                if stable_pose and foot_contacts >= 2:
                    base_reward += weight_stable * self.dt

                landing_quality = 0
                if self.three_foot_stable_steps >= three_foot_stability_steps:
                    landing_quality = 3
                    base_reward += landing_bonus
                    self.high_quality_landings += 1
                elif self.two_foot_stable_steps >= two_foot_stability_steps:
                    landing_quality = 2
                    base_reward += two_foot_landing_bonus
                    self.two_foot_landings += 1

                if landing_quality:
                    stable_landing_this_step = True
                    land_to_stable_seconds = self.land_elapsed_steps * self.dt
                    if land_to_stable_seconds <= successful_landing_deadline:
                        self.timely_landings += 1
                    self.state = "Takeoff"
                    self.start_z = z_height
                    self.takeoff_peak_height_gain = 0.0
                    self.passed_inverted = False
                    self.three_foot_stable_steps = 0
                    self.two_foot_stable_steps = 0
                    self.land_elapsed_steps = 0

        if penalize_off_axis_motion:
            r_lateral = -weight_y * float(observation[14]) ** 2 * self.dt
            r_roll = -weight_roll * float(observation[16]) ** 2 * self.dt
            r_yaw = -weight_yaw * float(observation[18]) ** 2 * self.dt
            base_reward += r_lateral + r_roll + r_yaw

        info["entered_flip"] = self.entered_flip
        info["entered_land"] = self.entered_land
        info["true_performance"] = self.completed_backflips
        info["timely_landings"] = self.timely_landings
        info["high_quality_landings"] = self.high_quality_landings
        info["two_foot_landings"] = self.two_foot_landings
        info["episode_max_land_elapsed_seconds"] = (
            self.episode_max_land_elapsed_steps * self.dt
        )
        info["takeoff_completed"] = takeoff_completed
        info["takeoff_peak_height_gain"] = self.takeoff_peak_height_gain
        info["stable_landing_this_step"] = stable_landing_this_step
        info["land_to_stable_seconds"] = land_to_stable_seconds

        return observation, base_reward, terminated, truncated, info


class AntPhaseObservation(gym.ObservationWrapper):
    def __init__(self, env: AntBackflipEnv):
        super().__init__(env)
        base_space = env.observation_space
        if not isinstance(base_space, gym.spaces.Box):
            raise TypeError("AntBackflipEnv must use a Box observation space")

        self.ant_env = env
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((base_space.low, np.zeros(4, dtype=base_space.dtype))),
            high=np.concatenate((base_space.high, np.ones(4, dtype=base_space.dtype))),
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