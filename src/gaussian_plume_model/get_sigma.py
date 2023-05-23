import numpy as np


def get_sigma(stability, downwind):
    """
    Return the dispersion coefficient (standard deviation / sigma) of the Gaussian plume.
    This function uses the sigma values from EPA's ISC Model.

    Parameters
    ----------
    stability : int
        Atmospheric stability parameter, whereby:
        1 - very unstable
        2 - moderately unstable
        3 - slightly unstable
        4 - neutral
        5 - moderately stable
        6 - very stable
    downwind : float
        Downwind distance from source [m]

    Returns
    -------
    sigma_y : float
        Standard deviation of the Gaussian plume in lateral direction (cross wind) [m]
    sigma_z : float
        Standard deviation of the Gaussian plume in vertical direction [m]
    """

    if stability not in [1, 2, 3, 4, 5, 6]:
        raise Exception("Atmospheric stability parameter should be 1, 2, 3, 4, 5 or 6.")
    if downwind <= 0:
        raise Exception("Downwind distance should positive.")

    if stability == 1:  # very unstable
        # To compute standard deviation in vertical direction
        downwind_bins = np.array([0, 100, 150, 200, 250, 300, 400, 500])
        a_values = np.array(
            [122.800, 158.080, 170.220, 179.520, 217.410, 258.89, 346.75, 453.85]
        )
        b_values = np.array(
            [0.94470, 1.05420, 1.09320, 1.12620, 1.26440, 1.40940, 1.7283, 2.1166]
        )
        # To compute standard deviation in lateral direction (cross wind)
        c = 24.1670
        d = 2.5334

    elif stability == 2:  # moderately unstable
        # To compute standard deviation in vertical direction
        downwind_bins = np.array([0, 200, 400])
        a_values = np.array([90.673, 98.483, 109.3])
        b_values = np.array([0.93198, 0.98332, 1.09710])
        # To compute standard deviation in lateral direction (cross wind)
        c = 18.3330
        d = 1.8096

    elif stability == 3:  # slightly unstable
        # To compute standard deviation in vertical direction
        a = 61.141
        b = 0.91465
        # To compute standard deviation in lateral direction (cross wind)
        c = 12.5
        d = 1.0857

    elif stability == 4:  # neutral
        # To compute standard deviation in vertical direction
        downwind_bins = np.array([0, 300, 1000, 3000, 10000, 30000])
        a_values = np.array([34.459, 32.093, 32.093, 33.504, 36.650, 44.053])
        b_values = np.array([0.86974, 0.81066, 0.64403, 0.60486, 0.56589, 0.51179])
        # To compute standard deviation in lateral direction (cross wind)
        c = 8.3330
        d = 0.72382

    elif stability == 5:  # moderately stable
        # To compute standard deviation in vertical direction
        downwind_bins = np.array([0, 100, 300, 1000, 2000, 4000, 10000, 20000, 40000])
        a_values = np.array(
            [24.26, 23.331, 21.628, 21.628, 22.534, 24.703, 26.970, 35.420, 47.618]
        )
        b_values = np.array(
            [
                0.83660,
                0.81956,
                0.75660,
                0.63077,
                0.57154,
                0.50527,
                0.46713,
                0.37615,
                0.29592,
            ]
        )
        # To compute standard deviation in lateral direction (cross wind)
        c = 6.25
        d = 0.54287

    else:  # stability == 6: # very stable
        # To compute standard deviation in vertical direction
        downwind_bins = np.array(
            [0, 200, 700, 1000, 2000, 3000, 7000, 15000, 30000, 60000]
        )
        a_values = np.array(
            [
                15.209,
                14.457,
                13.953,
                13.953,
                14.823,
                16.187,
                17.836,
                22.651,
                27.074,
                34.219,
            ]
        )
        b_values = np.array(
            [
                0.81558,
                0.78407,
                0.68465,
                0.63227,
                0.54503,
                0.46490,
                0.41507,
                0.32681,
                0.27436,
                0.21716,
            ]
        )
        # To compute standard deviation in lateral direction (cross wind)
        c = 4.1667
        d = 0.36191

    # To compute standard deviation in vertical direction based on downwind value
    if stability != 3:
        index = np.argmax(np.where(downwind > downwind_bins))
        a = a_values[index]
        b = b_values[index]

    # Compute standard deviation in vertical direction with values a, b
    sigma_z = a * (downwind / 1000) ** b
    if stability == 1 or stability == 2 or stability == 3:
        sigma_z = min(sigma_z, 5000)

    # Compute standard deviation in lateral direction (cross wind) with values c, d
    theta = 0.017453293 * (c - d * np.log(np.abs(downwind) / 1000))
    sigma_y = 465.11628 * downwind / 1000 * np.tan(theta)

    return sigma_y, sigma_z
