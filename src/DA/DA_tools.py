import numpy as np
from scipy import stats


def normal_to_lognormal(normal_distr):
    """
    Convert a normal distribution to a lognormal distribution

    Parameters
    ----------
    normal_distr: array (float)
        values with a normal distributions

    Returns
    -------
    lognormal_distr: array (float)
        values with a lognormal distribution
    """
    lognormal_distr = np.exp(normal_distr)
    return lognormal_distr


def lognormal_to_normal(lognormal_distr):
    """
    Convert a lognormal distribution to a normal distribution

    Parameters
    ----------
    lognormal_distr: array (float)
        values with a lognormal distribution

    Returns
    -------
    normal_distr: array (float)
        values with a normal distributions
    """
    normal_distr = np.log(lognormal_distr)
    return normal_distr


def get_entropy(flux):
    """
    Computes the entropy of an ensemble-based representation of a lognormal probability distribution
    Includes scaling of flux with /10 to go from g/s to g/(s*m2) (= /(dx*dy) = /10000) to mg/(s*m2) (= *1000)

    1) scale /10
    2) lognormal to normal
    3) compute mu and sigma
    4) compute entropy

    Parameters
    ----------
    flux: array
        lognormal distribution of flux

    Returns
    -------
    entropy: float
        information entropy in nats
    """
    flux = flux / 10
    flux = lognormal_to_normal(flux)
    mu = np.mean(flux)
    sigma = np.std(flux)
    entropy = stats.lognorm.entropy(s=sigma, loc=0, scale=np.exp(mu))
    return entropy


def get_kldiv(posterior, prior):
    """
    Computes the KL divergence in nats
    https://stats.stackexchange.com/questions/289323/kl-divergence-of-multivariate-lognormal-distributions

    1) scale /10
    2) lognormal to normal
    3) compute mu and sigma
    4) compute KL divergence

    Parameters
    ----------
    posterior: array
        estimated posterior of flux
    prior: array
        estimated prior of flux

    Returns
    -------
    KLdiv: float
        Kullback-Leibler divergence in nats
    """
    posterior = posterior / 10
    posterior = lognormal_to_normal(posterior)
    mu_post = np.mean(posterior)
    sigma_post = np.std(posterior)
    prior = prior / 10
    prior = lognormal_to_normal(prior)
    mu_prior = np.mean(prior)
    sigma_prior = np.std(prior)
    KLdiv = (
        np.log(sigma_prior / sigma_post)
        + ((sigma_post**2 + (mu_post - mu_prior) ** 2) / (2 * sigma_prior**2))
        - 0.5
    )
    return KLdiv


def get_lognormal_distribution(median, scatter, ensemble_size, seed=None):
    """
    Get lognormal distribution of ensemble_size, given the median and scatter

    Parameters
    ----------
    median: scalar (float or int)
        median of the lognormal distribution
    scatter: scalar (float or int)
        scatter of the lognormal distribution
    ensemble_size: scalar (int)
        number of values in the distribtuion
    seed: scalar (int)
        included for testing purposed only

    Returns
    -------
    lognormal_distr: array (float)
        values with a lognormal distribution
    """
    rng = np.random.default_rng(seed)
    lognormal_distr = np.exp(
        np.log(median) + np.log(scatter) * rng.standard_normal(ensemble_size)
    )
    return lognormal_distr
