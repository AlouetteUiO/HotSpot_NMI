import numpy as np
from src.gaussian_plume_model.gaussian_function import gaussian_function


def conc_to_ppm(concentration, temperature, pressure, molar_mass):
    """
    Converts gas concentration in g/s to ppm (in proportion of volume)

    See
    https://www.lenntech.com/calculators/ppm/converter-parts-per-million.htm#Documentation

    Parameters
    ----------
    concentration: array or scalar (float)
        gas concentration [g/s]
    temperature: float
        air temperature in degrees Kelvin [K]
    pressure: float
        atmospheric pressure [Pa]
        1 Atmosphere = 101325 Pascal
    molar_mass: float
        molar mass [g/mol]
        molar mass of CO2: 44.01 g/mol

    Returns
    -------
    concentration in ppm (in proportion of volume): float
    """

    R = 8.314510  # universal gas constant [(m3*Pa)/(K*mol)]

    molar_volume = R * temperature / pressure  # molar volume [m3/mol]

    conversion_factor = molar_volume / molar_mass * 1e6

    return concentration * conversion_factor


def get_ppm_concentration(
    emission_rate,
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
):
    """
    Return the gas concentration in ppm at location x_receiver, y_receiver, z_receiver as computed by the gaussian plume model.

    Parameters
    ----------
    emission_rate, wind_speed, wind_direction, x_receiver, y_receiver, z_receiver, x_source, y_source, z_source, stability
        See gaussian_function
    background_ppm: scalar (int or float)
        Background gas concentration [ppm]
    temperature, pressure, molar_mass
        See conc_to_ppm

    Returns
    -------
    concentration_pmm + background_ppm: scalar (float)
        Gas concentration at position x_receiver, y_receiver, z_receiver [ppm]
    """

    concentration = gaussian_function(
        emission_rate,
        wind_speed,
        wind_direction,
        x_receiver,
        y_receiver,
        z_receiver,
        x_source,
        y_source,
        z_source,
        stability,
    )

    concentration_ppm = conc_to_ppm(concentration, temperature, pressure, molar_mass)

    return concentration_ppm + background_ppm


def get_ppm_concentration_field(
    nx,
    ny,
    nz,
    dx,
    dy,
    dz,
    emission_rate,
    wind_speed,
    wind_direction,
    x_source,
    y_source,
    z_source,
    stability,
    background_ppm,
    temperature,
    pressure,
    molar_mass,
):
    """
    Return the gas concentration field in ppm as computed by the gaussian plume model.

    Parameters
    ----------
    nx, ny, nz: scalar (int)
        Number of grid cells in x, y, z direction
    dx, dy, dz: scalar (int or float)
        Grid cell size in x, y, z direction [m]
        For 2D concentration field (nz=1), the evaluation height is at dz/2
    emission_rate, wind_speed, wind_direction, x_receiver, y_receiver, z_receiver, x_source, y_source, z_source, stability
        See gaussian_function
    background_ppm: scalar (int or float)
        Background gas concentration [ppm]
    temperature, pressure, molar_mass
        See conc_to_ppm

    Returns
    -------
    concentration_field_ppm + background_ppm: array (float)
        Gas concentration field [ppm]
    """

    x_range = np.arange(start=dx / 2, stop=dx * nx, step=dx)
    y_range = np.arange(start=dy / 2, stop=dy * ny, step=dy)
    z_range = np.arange(start=dz / 2, stop=dz * nz, step=dz)

    concentration_field = np.zeros((nx, ny, nz))
    for x_loc, x_receiver in enumerate(x_range):
        for y_loc, y_receiver in enumerate(y_range):
            for z_loc, z_receiver in enumerate(z_range):
                concentration_field[x_loc, y_loc, z_loc] = gaussian_function(
                    emission_rate,
                    wind_speed,
                    wind_direction,
                    x_receiver,
                    y_receiver,
                    z_receiver,
                    x_source,
                    y_source,
                    z_source,
                    stability,
                )

    concentration_field_ppm = conc_to_ppm(
        concentration_field, temperature, pressure, molar_mass
    )

    return concentration_field_ppm + background_ppm


def get_observations(
    n_eval,
    obs_sigma,
    nx,
    ny,
    nz,
    dx,
    dy,
    dz,
    emission_rate,
    wind_speed,
    wind_direction,
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
    Returns synthetic observations of the the gas concentration field
    in ppm as computed by the gaussian plume model. The observations
    include measurment noise.

    Parameters
    ----------
    n_eval: scalar (int)
        Number of observed gas concentration fields
    obs_sigma: scalar (float)
        Measurement uncertainty
        Standard deviation of measurement noise
    nx, ny, nz: scalar (int)
        Number of grid cells in x, y, z direction
    dx, dy, dz: scalar (int or float)
        Grid cell size in x, y, z direction [m]
        For 2D concentration field (nz=1), the evaluation height is at dz/2
    emission_rate, wind_speed, wind_direction, x_receiver, y_receiver, z_receiver, x_source, y_source, z_source, stability
        See gaussian_function
    background_ppm: scalar (int or float)
        Background gas concentration [ppm]
    temperature, pressure, molar_mass
        See conc_to_ppm
    seed: scalar (int)
        Included for testing purposes only

    Returns
    -------
    observations: array (size: n_eval, nx, ny, nz)
        Synthetic observations of gas concentration field
        Including measurement noise
    """

    concentration_field_ppm = get_ppm_concentration_field(
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        emission_rate,
        wind_speed,
        wind_direction,
        x_source,
        y_source,
        z_source,
        stability,
        background_ppm,
        temperature,
        pressure,
        molar_mass,
    )
    observations = np.repeat(
        concentration_field_ppm[np.newaxis, :, :, :],
        repeats=n_eval,
        axis=0,
    )
    # Noise added by random samples from a
    # Gaussian probability distribution N(mu = 0, sigma = obs_sigma)
    rng = np.random.default_rng(seed)
    noise = obs_sigma * rng.standard_normal(size=(n_eval, nx, ny, nz))

    return observations + noise
