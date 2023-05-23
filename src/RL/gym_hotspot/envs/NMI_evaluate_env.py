import numpy as np
from scipy import stats
from copy import deepcopy

import gymnasium as gym
from gymnasium import spaces

from src.gaussian_plume_model.get_ppm_concentration import get_observations
from src.gaussian_plume_model.get_ppm_concentration import get_ppm_concentration
from src.DA.DA_tools import get_lognormal_distribution, get_entropy, get_kldiv
from src.DA.DA_tools import lognormal_to_normal, normal_to_lognormal
from src.DA.EnKF import EnKF
from src.DA.DA_wrappers import perform_EnKF_with_gaussian_plume_model
from src.RL.process.get_results import get_CRPS


class NMI_evaluate_v0(gym.Env):
    """
    This is a "mock" environment to process results.
    Not used for training.
    Flux_true is not randomly chosen in reset but has to be set by user.
    """

    def __init__(self):
        self.current_episode = -1

        # Define action space
        # 0: take one step in positive x direction
        # 1: take one step in negative x direction
        # 2: take one step in positive y direction
        # 3: take one step in negative y direction
        # 4: stay
        self.action_space = spaces.Discrete(5)

        self.nx = 10
        self.ny = 10
        self.nz = 1

        self.dx = 100
        self.dy = 100
        self.dz = 20  # 2D grid evaluated at z=10m

        self.x_range = np.arange(
            start=self.dx / 2, stop=self.dx * self.nx, step=self.dx
        )
        self.y_range = np.arange(
            start=self.dy / 2, stop=self.dy * self.ny, step=self.dy
        )
        self.z_range = np.arange(
            start=self.dz / 2, stop=self.dz * self.nz, step=self.dz
        )

        low = np.array([0, 0, 0], dtype=int)
        high = np.array(
            [
                self.nx - 1,
                self.ny - 1,
                16,
            ],
            dtype=int,
        )
        self.observation_space = spaces.Box(low, high, dtype=int)

        # Define known emission source location
        self.x_source = 1.5 * self.dx  # [m]
        self.y_source = 8.5 * self.dy  # [m]
        self.z_source = 0  # at ground level

        # Define known (atmospheric) parameters
        self.wind_speed = 4
        self.wind_direction = 320
        self.stability = 2
        self.background_ppm = 400
        self.temperature = 273.15 + 25
        self.pressure = 101325
        self.molar_mass = 44.01  # CO2

        # Define observation/measurement uncertainty
        self.obs_sigma = 30 / np.sqrt(12)  # [ppm]

        # Set RL variables
        self.max_RL_steps = 15
        self.max_RL_obs = self.max_RL_steps + 1

        # Set DA variables
        self.ensemble_size = 100
        self.DA_iter = 4
        self.prior_median = 0.1 * self.dx * self.dy  # 1000 g/s
        self.prior_scatter = 3
        self.DA_alpha = self.DA_iter
        self.DA_error = self.obs_sigma**2

    def set_initial_loc(self, x_loc, y_loc, z_loc):
        self.x_loc_init = x_loc
        self.y_loc_init = y_loc
        self.z_loc_init = z_loc

    def set_flux_true(self, flux_true):
        self.flux_true = flux_true

    def step(self, action):
        self.current_step += 1

        x_loc, y_loc, z_loc = self.loc[0], self.loc[1], self.loc[2]

        if action == 0:  # take one step in positive x direction
            next_x_loc = min(self.nx - 1, x_loc + 1)
            next_y_loc = y_loc
            next_z_loc = z_loc
        elif action == 1:  # take one step in negative x direction
            next_x_loc = max(0, x_loc - 1)
            next_y_loc = y_loc
            next_z_loc = z_loc
        elif action == 2:  # take one step in positive y direction
            next_x_loc = x_loc
            next_y_loc = min(self.ny - 1, y_loc + 1)
            next_z_loc = z_loc
        elif action == 3:  # take one step in negative y direction
            next_x_loc = x_loc
            next_y_loc = max(0, y_loc - 1)
            next_z_loc = z_loc
        else:  # stay
            next_x_loc = x_loc
            next_y_loc = y_loc
            next_z_loc = z_loc

        next_concentration = self.observations[
            self.current_obs + 1, next_x_loc, next_y_loc, next_z_loc
        ]

        # Convert location from grid cell to meter for Gaussian Plume Model
        next_x = (next_x_loc + 0.5) * self.dx
        next_y = (next_y_loc + 0.5) * self.dy
        next_z = (next_z_loc + 0.5) * self.dz

        flux_after_DA = perform_EnKF_with_gaussian_plume_model(
            flux=self.flux,
            observation=next_concentration,
            ensemble_size=self.ensemble_size,
            DA_iter=self.DA_iter,
            DA_alpha=self.DA_alpha,
            DA_error=self.DA_error,
            wind_speed=self.wind_speed,
            wind_direction=self.wind_direction,
            x_receiver=next_x,
            y_receiver=next_y,
            z_receiver=next_z,
            x_source=self.x_source,
            y_source=self.y_source,
            z_source=self.z_source,
            stability=self.stability,
            background_ppm=self.background_ppm,
            temperature=self.temperature,
            pressure=self.pressure,
            molar_mass=self.molar_mass,
        )
        self.flux = flux_after_DA  # update flux

        # Update current loc, state and obs parameters
        self.current_obs += 1
        self.loc = (int(next_x_loc), int(next_y_loc), int(next_z_loc))
        self.state = np.array([self.loc[0], self.loc[1], self.current_obs], dtype=int)

        reward = 0 # This is not used

        terminated = False
        if self.current_step == (self.max_RL_steps - 1):
            terminated = True

        return self.state, reward, terminated, False, {"flux": self.flux}

    def reset(self, seed=None, options=None):
        self.current_step = -1
        self.current_obs = 0
        self.current_episode += 1

        # Generate synthetic observations with noise for this RL episode
        self.observations = get_observations(
            self.max_RL_obs,
            self.obs_sigma,
            self.nx,
            self.ny,
            self.nz,
            self.dx,
            self.dy,
            self.dz,
            self.flux_true,
            self.wind_speed,
            self.wind_direction,
            self.x_source,
            self.y_source,
            self.z_source,
            self.stability,
            self.background_ppm,
            self.temperature,
            self.pressure,
            self.molar_mass,
        )

        # Set prior
        self.flux = get_lognormal_distribution(
            self.prior_median, self.prior_scatter, self.ensemble_size
        )
        self.flux_prior = deepcopy(self.flux)

        observation = self.observations[
            0, self.x_loc_init, self.y_loc_init, self.z_loc_init
        ]
        x = (self.x_loc_init + 0.5) * self.dx
        y = (self.y_loc_init + 0.5) * self.dy
        z = (self.z_loc_init + 0.5) * self.dz

        # First observation at starting location
        flux_after_DA = perform_EnKF_with_gaussian_plume_model(
            flux=self.flux,
            observation=observation,
            ensemble_size=self.ensemble_size,
            DA_iter=self.DA_iter,
            DA_alpha=self.DA_alpha,
            DA_error=self.DA_error,
            wind_speed=self.wind_speed,
            wind_direction=self.wind_direction,
            x_receiver=x,
            y_receiver=y,
            z_receiver=z,
            x_source=self.x_source,
            y_source=self.y_source,
            z_source=self.z_source,
            stability=self.stability,
            background_ppm=self.background_ppm,
            temperature=self.temperature,
            pressure=self.pressure,
            molar_mass=self.molar_mass,
        )
        self.flux = flux_after_DA

        self.loc = (int(self.x_loc_init), int(self.y_loc_init), int(self.z_loc_init))
        self.state = np.array([self.loc[0], self.loc[1], self.current_obs], dtype=int)

        return self.state, {"flux": self.flux}

    def close(self):
        None