import numpy as np
from scipy.linalg import sqrtm


def EnKF(prior, observations, prediction, DA_alpha, DA_error, seed=None):
    """
    Ensemble Kalman Filter

    Parameters
    ----------
    prior: array (n_vars x ensemble_size)
        prior ensemble matrix
        assumed to be a Gaussian probability distribtion
    observations: array (n_obs)
        observation vector
        assumed to be a Gaussian probability distribtion
    prediction: array (n_obs x ensemble_size)
        predicted observation ensemble matrix
        follows from the forward model with prior input values
    DA_alpha: scalar
        observation error inflation coefficient
    DA_error: scalar or array (n_obs or n_obs x n_obs)
        observation error covariance matrix
    seed: scalar (int)
        included for testing purposes only

    n_vars: number of variables
    n_obs: number of observations
    ensemble_size: number of ensemble members

    Returns
    -------
    posterior: array (n_vars x ensemble_size)
        posterior ensemble matrix
        assumed to be a Gaussian probability distribtion
    """

    # Get dimensions of input arrays: n_obs, n_vars, ensemble_size
    n_obs = np.size(observations)
    if prior.ndim == 1:
        n_vars = 1
        ensemble_size = prior.size
    elif prior.ndim == 2:
        n_vars = prior.shape[0]
        ensemble_size = prior.shape[1]
    else:
        raise Exception("Prior should be 1D or 2D array")

    # Check if dimensions of observations, prior and prediction match
    if n_obs == 1:
        if prediction.size != ensemble_size:
            raise Exception(
                f"- n_obs = {n_obs} - Dimensions of observations (n_obs = {n_obs}), prior (n_vars x ensemble_size = {n_vars} x {ensemble_size}) \
                and prediction (n_obs x ensemble_size = {n_obs} x {ensemble_size}) do not match"
            )
    else:  # n_obs > 1
        if (prediction.shape[0] != n_obs) or (prediction.shape[1] != ensemble_size):
            raise Exception(
                f"- n_obs = {n_obs} - Dimensions of observations (n_obs = {n_obs}), prior (n_vars x ensemble_size = {n_vars} x {ensemble_size}) \
                and prediction (n_obs x ensemble_size = {n_obs} x {ensemble_size}) do not match"
            )

    # Reshape prior and prediction to explicitly create 2D arrays if 1D arrays are given
    prior = prior.reshape(n_vars, ensemble_size)
    prediction = prediction.reshape(n_obs, ensemble_size)

    # Check dimension of input variable DA_error
    # and create observation error covariance matrix with dimensions n_obs x n_obs
    if np.isscalar(DA_error):
        R = DA_error * np.identity(n_obs)
    elif DA_error.ndim == 1:
        if DA_error.size == n_obs:
            R = np.diag(DA_error)
        else:
            raise Exception(f"Dimension of 1D DA_error array should be n_obs = {n_obs}")
    elif DA_error.ndim == 2:
        if DA_error.shape[0] == DA_error.shape[1] == n_obs:
            pass
        else:
            raise Exception(
                f"Dimension of 2D DA_error array should be n_obs x n_obs = {n_obs} x {n_obs}"
            )
    else:
        raise Exception("Error should be scalar, 1D or 2D array")

    # Anomaly calculations
    prior_mean = np.mean(prior, axis=1)  # ensemble mean (for each variable)
    A = prior - prior_mean.reshape(n_vars, 1)

    prediction_mean = np.mean(prediction, axis=1)  # ensemble mean (for each observation)
    B = prediction - prediction_mean.reshape(n_obs, 1)
    B_transpose = B.T

    C_AB = A @ B_transpose
    C_BB = B @ B_transpose

    # Create perturbed observations
    R_sqrt = sqrtm(R)  # Square root of the observation error covariance matrix
    rng = np.random.default_rng(seed)
    perturbations = (
        np.sqrt(DA_alpha) * R_sqrt @ rng.standard_normal((n_obs, ensemble_size))
    )
    observations_perturbed = (
        np.outer(observations, np.ones(ensemble_size)) + perturbations
    )

    # Create scaled observation error covariance matrix with dimensions n_obs x n_obs
    R_scaled = (ensemble_size * DA_alpha) * R

    # Analysis step
    Kalman_gain = C_AB @ (np.linalg.inv(C_BB + R_scaled))
    innovation = observations_perturbed - prediction
    posterior = prior + Kalman_gain @ innovation

    return posterior
