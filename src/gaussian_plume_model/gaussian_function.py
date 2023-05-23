import numpy as np
from src.gaussian_plume_model.get_sigma import get_sigma


def gaussian_function(
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
):
    """
    Return the gas concentration at location x, y, z as computed by the gaussian plume model.

    Parameters
    ----------
    emission_rate: float
        Mass emission rate [g/s]
        E.g. for source of 0.25 g/(s*m2) and grid size of 100m x 100m,
        the mass emission rate is 0.25 * 100 * 100 = 2500 g/s,
        when treating the entire cell as an emission source.
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

    Returns
    -------
    concentration: float
        Gas concentration at position x, y, z [g/m3]
    """

    # Shift coordinate system so that the source is at (0, 0)
    x = x_receiver - x_source
    y = y_receiver - y_source

    # Distance between source and receiver (magnitude of vector)
    distance = np.sqrt(x**2 + y**2)

    # Wind speed components in x and y direction
    u = wind_speed * np.sin((wind_direction - 180) * np.pi / 180)
    v = wind_speed * np.cos((wind_direction - 180) * np.pi / 180)

    # Dot product of the two vectors (needed to compute the angle between the two vectors)
    dot_product = u * x + v * y

    # Angle between wind and position (x,y)
    #   arccos is defined in range [-1, 1]
    #   cannot divide a number by zero
    magnitude_product = wind_speed * distance
    if magnitude_product != 0:
        alpha = np.arccos(max(-1, min(1, dot_product / magnitude_product)))
    else:
        alpha = 0

    # Compute downwind and crosswind using the dot product.
    #   In general, the dot product is given by:
    #       a * b = ||a|| * ||b|| * cos(theta)
    #   where a and b are vectors.
    downwind = np.cos(alpha) * distance
    crosswind = np.sin(alpha) * distance

    # Compute concentration if downwind > 0, otherwise concentration is zero
    if downwind > 0:
        # Get the standard deviations [m] of the gaussian plume based on stability and downwind distance [m]
        sigma_y, sigma_z = get_sigma(stability, downwind)

        concentration = (
            emission_rate
            / (2 * np.pi * wind_speed * sigma_y * sigma_z)
            * np.exp(-(crosswind**2) / (2 * sigma_y**2))
            * (
                np.exp(-((z_receiver - z_source) ** 2) / (2 * sigma_z**2))
                + np.exp(-((z_receiver + z_source) ** 2) / (2 * sigma_z**2))
            )
        )

    else:
        concentration = 0

    return concentration
