import numpy as np
from src.DA.EnKF import EnKF
from src.DA.DA_tools import normal_to_lognormal, lognormal_to_normal
from src.gaussian_plume_model.get_ppm_concentration import get_ppm_concentration


def perform_EnKF_with_gaussian_plume_model(
    flux,
    observation,
    ensemble_size,
    DA_iter,
    DA_alpha,
    DA_error,
    wind_speed,
    wind_direction,
    x_receiver,
    y_receiver,
    z_receiver,
    x_source,
    y_source,
    z_source,
    stability,
    background_ppm,
    temperature,
    pressure,
    molar_mass,
    seed=None,
):
    """
    Perform the Ensemble Kalman Filter with predictions from the Gaussian Plume Model

    Parameters
    ----------
    flux: array
        Ensemble of the mass emission rate of the hotspot [g/s]
        Lognormal distribution
    observation:
        Gas concentration measured at position x_receiver, y_receiver, z_receiver [ppm]
    ensemble_size: scalar(int)
        Number of ensemble members
    DA_iter: scalar (int)
        Number of filter iterations
    DA_alpha: scalar
        Observation error inflation coefficient
    DA_error: scalar or array (n_obs or n_obs x n_obs)
        Observation error covariance matrix
    wind_speed: float
        Magnitude of the wind vector [m/s]
    wind_direction: float
        With wind direction = 0 for wind blowing from +y to -y direction.
    x_receiver, y_receiver, z_receiver: float
        x, y, z Position of receiver [m]
    x_source, y_source, z_source:
        x, y, z Position of emission source [m]
        (z_source is the stack height)
    stability: int
        Atmospheric stability parameter, whereby:
        1 - very unstable
        2 - moderately unstable
        3 - slightly unstable
        4 - neutral
        5 - moderately stable
        6 - very stable
    background_ppm: scalar (int or float)
        Background gas concentration [ppm]
    temperature: float
        air temperature in degrees Kelvin [K]
    pressure: float
        atmospheric pressure [Pa]
        1 Atmosphere = 101325 Pascal
    molar_mass: float
        molar mass [g/mol]
        molar mass of CO2: 44.01 g/mol
    seed: scalar (int)
        included for testing purposes only

    Returns
    -------
    (updated) flux: array
        Ensemble of the mass emission rate of the hotspot [g/s]
    """

    # Get prediction
    prediction = np.zeros(ensemble_size)
    for i in range(ensemble_size):
        prediction[i] = get_ppm_concentration(
            flux[i],
            wind_speed,
            wind_direction,
            x_receiver,
            y_receiver,
            z_receiver,
            x_source,
            y_source,
            z_source,
            stability,
            background_ppm,
            temperature,
            pressure,
            molar_mass,
        )

    for _ in range(DA_iter):
        # Ensemble Kalman Filter
        prior = lognormal_to_normal(flux)
        posterior = EnKF(prior, observation, prediction, DA_alpha, DA_error, seed)
        flux = normal_to_lognormal(
            posterior[0]
        )  # postorior is 2D array with shape (n_vars, ensemble_size), modify to 1D array using [0]

        # Update prediction
        prediction = np.zeros(ensemble_size)
        for i in range(ensemble_size):
            prediction[i] = get_ppm_concentration(
                flux[i],
                wind_speed,
                wind_direction,
                x_receiver,
                y_receiver,
                z_receiver,
                x_source,
                y_source,
                z_source,
                stability,
                background_ppm,
                temperature,
                pressure,
                molar_mass,
            )

    return flux
