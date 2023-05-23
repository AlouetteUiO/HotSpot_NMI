import numpy as np
from scipy.stats import shapiro
from pytest import approx
from src.DA.EnKF import EnKF
from src.DA.DA_wrappers import perform_EnKF_with_gaussian_plume_model


def test_EnKF_1obs():
    """
    Test Ensemble Kalman Filter for one observation
    """

    def forward_model(a):
        """Force = mass * acceleration"""
        m = 1000  # mass
        F = m * a
        return F

    def get_synthetic_observations(a_true, obs_sigma, nr_obs):
        """Get noisy syntehtics observations for force (n_obs)"""
        F_true = forward_model(a_true)
        rng = np.random.default_rng()
        noise = obs_sigma * rng.standard_normal(size=nr_obs)
        F_obs = F_true + noise
        return F_obs

    def get_prior(a_mean, a_sigma, ensemble_size):
        """Get prior of acceleration (n_vars x ensemble_size)"""
        rng = np.random.default_rng()
        a_prior = a_mean + a_sigma * rng.standard_normal(size=ensemble_size)
        return a_prior

    def get_prediction(a_prior):
        """Get prediction of force (n_obs x ensemble_size)"""
        F_pred = forward_model(a_prior)
        return F_pred

    a_true = 5
    a_mean = 4
    a_sigma = 2
    ensemble_size = 1000
    n_obs = 1
    obs_sigma = 0.3
    DA_alpha = 1

    F_obs = get_synthetic_observations(a_true, obs_sigma, n_obs)
    a_prior = get_prior(a_mean, a_sigma, ensemble_size)
    F_pred = get_prediction(a_prior)

    a_posterior = EnKF(
        prior=a_prior,
        observations=F_obs,
        prediction=F_pred,
        DA_alpha=DA_alpha,
        DA_error=obs_sigma**2,
    )

    # Mean of posterior is closer to true value than mean of prior
    assert abs(np.mean(a_prior) - a_true) > abs(np.mean(a_posterior) - a_true)

    # The implementation of EnKF is a rewritten version of another code
    # Test against the original code
    a_prior = np.array(
        [[-4.16169214, 10.99731434, 6.62645354, 22.08527451, 18.05984661]]
    )
    F_pred = get_prediction(a_prior)
    a_posterior = EnKF(
        prior=a_prior,
        observations=50024.46634244,
        prediction=F_pred,
        DA_alpha=10,
        DA_error=50**2,
        seed=1234,
    )
    assert np.mean(a_posterior[0]) == approx(50.019686, abs=1e-5)
    assert np.std(a_posterior[0]) == approx(0.141295, abs=1e-5)


def test_perform_EnKF_with_gaussian_plume_model_wrapper():
    """
    Test performing the Ensemble Kalman Filter (2 iterations) with predictions of the Gaussian Plume Model.
    The test results are only based on the output of the original code, 
    not on literature or values otherwise known to be true.
    """

    flux = perform_EnKF_with_gaussian_plume_model(
        flux=np.array([171.70200047, 1072.9597456, 2256.80994785]),
        observation=388.292026031485,
        ensemble_size=3,
        DA_iter=2,
        DA_alpha=4,
        DA_error=(30 / np.sqrt(12)) ** 2,
        wind_speed=4,
        wind_direction=315,
        x_receiver=150,
        y_receiver=150,
        z_receiver=10,
        x_source=150,
        y_source=850,
        z_source=0,
        stability=2,
        background_ppm=400,
        temperature=273.15 + 25,
        pressure=101325,
        molar_mass=44.01,
        seed=1234,
    )

    assert flux[0] == approx(171.70198156)
    assert flux[1] == approx(1072.95970441)
    assert flux[2] == approx(2256.80992696)
