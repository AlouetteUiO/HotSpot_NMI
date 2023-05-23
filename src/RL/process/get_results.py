import numpy as np
import json
import properscoring as ps
import itertools
import gymnasium as gym
import src.RL.gym_hotspot

from src.gaussian_plume_model.get_ppm_concentration import get_observations
from src.DA.DA_tools import get_lognormal_distribution, lognormal_to_normal
from src.DA.DA_wrappers import perform_EnKF_with_gaussian_plume_model


def make_epsilon_greedy_policy(Q, epsilon):
    def policy_fn(state):
        action_probs = np.invert(np.isnan(Q[state])).astype(float)
        best_action = np.nanargmax(Q[state])
        action_probs = action_probs * epsilon / np.count_nonzero(action_probs)
        action_probs[best_action] += 1.0 - epsilon
        return action_probs

    return policy_fn


def json_to_Q(data):
    """
    Modify Q as saved in json-format to be used in RL code:
    - keys (strings) to tuple,
    - null to np.NaN,
    - and values (lists) to arrays
    """
    return dict(
        (
            tuple(map(int, key[1:-1].split(", "))),
            np.array([np.NaN if i == None else i for i in value]),
        )
        for (key, value) in data.items()
    )


def get_CRPS(probability_distribution, true_value):
    """Computes the Continuously Ranked Probability Score [same unit as observed variable]"""
    CRPS = ps.crps_ensemble(true_value, probability_distribution)
    return CRPS


def get_RMSE(probability_distribution, true_value):
    """Computes the Root Mean Square Error [same unit as observed variable]"""
    RMSE = np.sqrt(np.mean((probability_distribution - true_value) ** 2))
    return RMSE


def RL_results(file_name, env_name, n_eval, flux_true, x_init, y_init):
    """
    Process RL results by performing DA with the optimal policy found during training.

    Parameters
    ----------
    file_name: string
        json file with results from training and evaluation phase
    n_eval: scalar (int)
        Number of episodes run to smooth out random selection of ensemble
        members in the prior flux
    ensemble_size: scalar (int)
        Number of ensemble members
    prior_median, prior_scatter: scalar (float or int)
        Median and scatter of prior lognormal probability distribution
    DA_iter: scalar (int)
        Number of filter iterations
    flux_true: scalar (int)
        True surface flux for which to compute the performance
    x_init, y_init: scalar (int)
        Starting position of the drone
    """

    # Load json results file
    f = open(file_name)
    data = json.load(f)
    max_RL_steps = data["max_RL_steps"]
    max_RL_obs = max_RL_steps + 1

    # Get results
    Q_json = data["Q"]
    Q = json_to_Q(Q_json)
    greedy_policy = make_epsilon_greedy_policy(Q, epsilon=0.0)

    # Load Gym environment
    env = gym.make(env_name)

    # Keep track of fluxes
    fluxes = np.zeros((n_eval, max_RL_obs, env.ensemble_size))  # to save lognormal pdfs

    for i_eval in range(n_eval):  # Run a greedy policy n_eval times
        env.set_initial_loc(x_init, y_init, 0)
        env.set_flux_true(flux_true)
        state, flux = env.reset()
        state = tuple(state)

        # Keep track of statistics
        fluxes[i_eval, 0, :] = flux["flux"]

        # Process one episode
        for i_obs in range(1, max_RL_obs):
            action_probs = greedy_policy(state)
            rng = np.random.default_rng()
            action = rng.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, _, _, _, flux = env.step(action)
            next_state = tuple(next_state)
            state = next_state

            # Keep track of statistics
            fluxes[i_eval, i_obs, :] = flux["flux"]

    RMSE = np.zeros(max_RL_obs)
    CRPS = np.zeros(max_RL_obs)
    RMSE_std = np.zeros(max_RL_obs)
    CRPS_std = np.zeros(max_RL_obs)
    # scale fluxes by /10 to go from g/s to g/(s*m2) (= /(dx*dy) = /10000) to mg/(s*m2) (= *1000)
    for i_obs in range(max_RL_obs):
        RMSE_result = [
            get_RMSE(flux / 10, flux_true / 10) for flux in fluxes[:, i_obs, :]
        ]
        RMSE[i_obs] = np.mean(RMSE_result)
        RMSE_std[i_obs] = np.std(RMSE_result)
        CRPS_result = [
            get_CRPS(flux / 10, flux_true / 10) for flux in fluxes[:, i_obs, :]
        ]
        CRPS[i_obs] = np.mean(CRPS_result)
        CRPS_std[i_obs] = np.std(CRPS_result)

    # To save statistics to json file
    data.update({"process_results_n_eval": n_eval})
    data.update({f"RMSE_{flux_true}_{x_init}{y_init}": RMSE.tolist()})
    data.update({f"CRPS_{flux_true}_{x_init}{y_init}": CRPS.tolist()})
    data.update({f"RMSE_std_{flux_true}_{x_init}{y_init}": RMSE_std.tolist()})
    data.update({f"CRPS_std_{flux_true}_{x_init}{y_init}": CRPS_std.tolist()})

    json_object_results = json.dumps(data, indent=4)
    with open(file_name, "w") as f:
        f.write(json_object_results)

    return None


def RL_path(file_name, env_name, flux_true, x_init, y_init):
    """
    run one episode with obs_sigma = 0
    and greedy action selection
    """
    # Load json results file
    f = open(file_name)
    data = json.load(f)

    # Get results
    Q_json = data["Q"]
    Q = json_to_Q(Q_json)
    greedy_policy = make_epsilon_greedy_policy(Q, epsilon=0.0)

    # Load Gym environment
    env = gym.make(env_name)

    env.set_initial_loc(x_init, y_init, 0)
    env.set_flux_true(flux_true)
    state, _ = env.reset()
    state = tuple(state)

    # Keep track of fluxes
    locs = [env.loc]

    # Process one episode
    for _ in itertools.count():
        action_probs = greedy_policy(state)
        rng = np.random.default_rng()
        action = rng.choice(np.arange(len(action_probs)), p=action_probs)

        next_state, _, terminated, _, _ = env.step(action)
        next_state = tuple(next_state)
        locs.append(env.loc)

        if terminated:
            break

        state = next_state

    data.update({f"optimal_policy_locs_{x_init}{y_init}": locs})
    json_object_results = json.dumps(data, indent=4)
    with open(file_name, "w") as f:
        f.write(json_object_results)

    return None


def RL_for_figure(file_name, env_name, n_eval, flux_true, x_init, y_init):
    """
    run episodes with an optimal policy
    and greedy action selection
    """
    # Load json results file
    f = open(file_name)
    data = json.load(f)

    # Get results
    Q_json = data["Q"]
    Q = json_to_Q(Q_json)
    greedy_policy = make_epsilon_greedy_policy(Q, epsilon=0.0)

    # Load Gym environment
    env = gym.make(env_name)

    mean_posterior = np.zeros(n_eval)
    std_posterior = np.zeros(n_eval)

    for i_eps in range(n_eval):
        env.set_initial_loc(x_init, y_init, 0)
        env.set_flux_true(flux_true)
        state, _ = env.reset()
        state = tuple(state)

        # Process one episode
        for _ in itertools.count():
            action_probs = greedy_policy(state)
            rng = np.random.default_rng()
            action = rng.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, _, terminated, _, _ = env.step(action)
            next_state = tuple(next_state)

            if terminated:
                mean_posterior[i_eps] = np.mean(lognormal_to_normal(env.flux / 10))
                std_posterior[i_eps] = np.std(lognormal_to_normal(env.flux / 10))
                break

            state = next_state

    data.update(
        {f"mean_spaghetti_{flux_true}__{x_init}{y_init}": mean_posterior.tolist()}
    )
    data.update(
        {f"std_spaghetti_{flux_true}__{x_init}{y_init}": std_posterior.tolist()}
    )
    json_object_results = json.dumps(data, indent=4)
    with open(file_name, "w") as f:
        f.write(json_object_results)

    return None


if __name__ == "__main__":
    n_eval_stats = 10  # compute the CRPS and the RMSE over n_eval episodes
    n_eval_figure = (
        10  # get mean and std of final estimated posterior for n_eval_figure runs
    )
    file_name = "data/results_v0_example.json"
    env_name = "NMI_evaluate-v0"
    fluxes = [2000, 2500, 3000]
    RL_path(
        file_name=file_name,
        env_name=env_name,
        flux_true=2500,  # does not matter what flux, path is the same
        x_init=0,
        y_init=0,
    )
    RL_for_figure(
        file_name=file_name,
        env_name=env_name,
        n_eval=n_eval_figure,
        flux_true=2500,  # here is does matter what true flux you choose
        x_init=0,
        y_init=0,
    )
    for flux in fluxes:
        RL_results(
            file_name=file_name,
            env_name=env_name,
            n_eval=n_eval_stats,
            flux_true=flux,
            x_init=0,
            y_init=0,
        )
