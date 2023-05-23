import numpy as np
from pytest import approx
from src.gaussian_plume_model.gaussian_function import gaussian_function
from src.gaussian_plume_model.get_sigma import get_sigma
from src.gaussian_plume_model.get_ppm_concentration import conc_to_ppm
from src.gaussian_plume_model.get_ppm_concentration import get_ppm_concentration_field


def test_get_sigma():
    """
    Example calculation from
    http://faculty.washington.edu/markbenj/CEE357/CEE%20357%20air%20dispersion%20models.pdf
    """
    sigma_y, sigma_z = get_sigma(stability=4, downwind=500)
    assert sigma_y == approx(36.1, abs=1e-1)
    assert sigma_z == approx(18.3, abs=1e-1)


def test_gaussian_function():
    """
    Example calculation from
    http://faculty.washington.edu/markbenj/CEE357/CEE%20357%20air%20dispersion%20models.pdf
    """
    concentration = gaussian_function(
        emission_rate=10,
        wind_speed=6,
        wind_direction=270,
        x_receiver=500,
        y_receiver=0,
        z_receiver=0,
        x_source=0,
        y_source=0,
        z_source=50,
        stability=4,
    )
    assert concentration == approx(1.92e-5, abs=1e-7)
    concentration = gaussian_function(
        emission_rate=10,
        wind_speed=6,
        wind_direction=90,
        x_receiver=500,
        y_receiver=0,
        z_receiver=0,
        x_source=0,
        y_source=0,
        z_source=50,
        stability=4,
    )
    assert concentration == approx(0)


def test_conc_to_ppm():
    """
    Result from calculator
    https://www.lenntech.com/calculators/ppm/converter-parts-per-million.htm
    """
    concentration_in_ppm = conc_to_ppm(
        concentration=1, temperature=273.15, pressure=1e5, molar_mass=44.01
    )
    assert concentration_in_ppm == approx(516, abs=1e-1)


def test_get_ppm_concentration_field():
    """
    Test based on output of previous implementation of model
    """
    nx, ny, nz = 10, 10, 1
    dx, dy, dz = 100, 100, 20
    emission_rate = 2500
    wind_speed = 4
    wind_direction = 315
    x_source = 150
    y_source = 850
    z_source = 0
    stability = 2
    background_ppm = 400

    C = get_ppm_concentration_field(
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
        temperature=273.15 + 25,
        pressure=101325,
        molar_mass=44.01,
    )

    # Concentration upwind of the source is equal to the background concentration
    assert C[0, 0, 0] == approx(background_ppm)
    # The maximum concentration is in grid cell (2, 7, 0)
    assert np.unravel_index(np.argmax(C), C.shape) == (2, 7, 0)
    # The concentration decreases further downwind from the source
    assert (
        C[2, 7, 0]
        > C[3, 6, 0]
        > C[4, 5, 0]
        > C[5, 4, 0]
        > C[6, 3, 0]
        > C[7, 2, 0]
        > C[8, 1, 0]
        > C[9, 0, 0]
    )
    # The concentration decreases in cross-wind direction
    assert C[5, 4, 0] > C[6, 5, 0] > C[7, 6, 0] > C[8, 7, 0] > C[9, 8, 0]
