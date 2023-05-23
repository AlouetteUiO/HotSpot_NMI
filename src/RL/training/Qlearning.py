import numpy as np
import itertools
import time
from datetime import date
from collections import defaultdict
import sys
import json

import gymnasium as gym
import src.RL.gym_hotspot


def make_epsilon_greedy_policy(Q, epsilon):
    def policy_fn(state):
        """
        Epsilon-greedy action selection

        Parameters
        ----------
        state: tuple
            state of agent given by (n_obs, x_loc, y_loc)

        Returns
        -------
        action_probs: array (size: env.action_space.n)

        Probability of selecting the action with the largest state-action value is:
        (1 - epsilon) + epsilon / number of possible actions
        Probability of selection any other action is:
        epsilon / number of possible actions
        Note: epsilon = 1.0 -> full exploration, select action at random
              epsilon = 0.0 -> full exploitation, select best known action
        Note: NaN actions are excluded from selection
        """
        action_probs = np.invert(np.isnan(Q[state])).astype(float)
        best_action = np.nanargmax(Q[state])
        action_probs = action_probs * epsilon / np.count_nonzero(action_probs)
        action_probs[best_action] += 1.0 - epsilon
        return action_probs

    return policy_fn


def update_Q(env, Q, state):
    """
    Create new key, value pair in Q-dict for state
    Note: Actions taking the agent off the grid are excluded from action-selection (NaN)
    """
    Q[state] = np.zeros(env.action_space.n)
    if state[0] == env.nx - 1:  # if x_loc == nx -> NOT move in +x (action 0)
        Q[state][0] = np.NaN
    if state[0] == 0:  # if x_loc == 0  -> NOT move in -x (action 1)
        Q[state][1] = np.NaN
    if state[1] == env.ny - 1:  # if y_loc == ny -> NOT move in +y (action 2)
        Q[state][2] = np.NaN
    if state[1] == 0:  # if y_loc == 0  -> NOT move in -y (action 3)
        Q[state][3] = np.NaN
    return Q


def Q_learning(
    env,
    n_episodes,
    discount_factor,
    epsilon_max,
    epsilon_min,
    epsilon_decay,
    alpha_max,
    alpha_min,
    alpha_decay,
):
    """
    Q-learning algorithm

    Parameters
    ----------
    env: class
        Gym environment 
    n_episodes: scalar (int)
        Number of training episodes
    discount_factor: scalar
        Discount factor
    epsilon_max, epsilon_min, epsilon_decay: scalar (float)
        To compute exploration rate epsilon by
        epsilon = max(epsilon_min, (epsilon_decay**i_episode) * epsilon_max)
    alpha_max, alpha_min, alpha_decay: scalar (float)
        To compute learning rate alpha by
        alpha = max(alpha_min, (alpha_decay**i_episode) * alpha_max)

    Returns
    -------
    perf_counter: scalar (float)
        Wall-clock time (comparable to using a stopwatch)
    process_time: scalar(float)
        Time for single-tasked computer (should not count time cpu is running anything else, e.g. sleep)
    episode_rewards: list (size: n_episodes)
        Sum of rewards per episode
    episode_locs: list of list (size: (n_episodes, env.max_RL_obs)
        Locations of the agent given in array [x_loc, y_loc, z_loc]
    Q: dict
        Stores state-action values
        Keys: tuple
            state of agent given by (n_obs, x_loc, y_loc)
        Values: array (size: env.action_space.n)
            corresponding state-action values for each possible
            action given the state in the key
    """

    start_perf_counter = time.perf_counter()
    start_process_time = time.process_time()

    Q = defaultdict()

    # Keep track of useful statistics
    episode_rewards = np.zeros(n_episodes)

    for i_episode in range(n_episodes):

        # Print episode number
        if (i_episode + 1) % 10 == 0:
            print(f"\rEpisode {i_episode+1}/{n_episodes}")
            sys.stdout.flush()

        rng = np.random.default_rng() # Initial location
        x_init = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=1, p=[10/36, 2/36, 2/36, 2/36, 2/36, 2/36, 2/36, 2/36, 2/36, 10/36])[0]
        if (x_init == 0) or (x_init == 9):
            y_init = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=1)[0]
        else:
            y_init = rng.choice([0, 9], size=1)[0]
        z_init = 0

        # Compute exploration (epsilon) and learning rate (alpha) with exponential decay method
        epsilon = max(epsilon_min, (epsilon_decay**i_episode) * epsilon_max)
        alpha = max(alpha_min, (alpha_decay**i_episode) * alpha_max)

        policy = make_epsilon_greedy_policy(Q, epsilon)

        # Reset environment for new episode
        env.set_initial_loc(x_init, y_init, z_init)
        state, _ = env.reset()
        state = tuple(state)

        if state not in Q:
            Q = update_Q(env, Q, state)

        for _ in itertools.count():

            # Take a step
            action_probs = policy(state)
            rng = np.random.default_rng()
            action = rng.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, terminated, _, _ = env.step(action)
            next_state = tuple(next_state)

            # Update statistics
            episode_rewards[i_episode] += int(reward)

            if tuple(next_state) not in Q:
                Q = update_Q(env, Q, tuple(next_state))

            # Temporal Difference Update
            best_next_action = np.nanargmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            # print(f"updated state {state}, action {action}, with reward {reward}")

            if terminated:
                break

            state = next_state

    perf_counter = time.perf_counter() - start_perf_counter
    process_time = time.process_time() - start_process_time

    return perf_counter, process_time, episode_rewards, Q


def Q_to_json(data):
    """
    Modify Q to be saved in json-format:
    - keys (tuple) to strings,
    - np.NaN to None (null in json),
    - and values (array) to list
    """
    return dict(
        (str(key), [None if np.isnan(i) else i for i in value])
        for (key, value) in data.items()
    )


def save_results(
    env,
    env_name,
    json_name,
    n_episodes,
    discount_factor,
    epsilon_max,
    epsilon_min,
    epsilon_decay,
    alpha_max,
    alpha_min,
    alpha_decay,
    perf_counter,
    process_time,
    episode_rewards,
    Q,
):
    """
    Save results in json-file
    """

    Q_json = Q_to_json(Q)

    dict_results = {
        "env_name": env_name,
        "date": str(date.today()),
        "n_episodes": n_episodes,
        "discount_factor": discount_factor,
        "epsilon_max": epsilon_max,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "alpha_max": alpha_max,
        "alpha_min": alpha_min,
        "alpha_decay": alpha_decay,
        "x_source": env.x_source,
        "y_source": env.y_source,
        "z_source": env.z_source,
        "nx": env.nx,
        "ny": env.ny,
        "nz": env.nz,
        "dx": env.dx,
        "dy": env.dy,
        "dz": env.dz,
        "x_range": env.x_range.tolist(),
        "y_range": env.y_range.tolist(),
        "z_range": env.z_range.tolist(),
        "wind_speed": env.wind_speed,
        "wind_direction": env.wind_direction,
        "stability": env.stability,
        "background_ppm": env.background_ppm,
        "temperature": env.temperature,
        "pressure": env.pressure,
        "molar_mass": env.molar_mass,
        "obs_sigma": env.obs_sigma,
        "max_RL_steps": env.max_RL_steps,
        "max_RL_obs": env.max_RL_obs,
        "ensemble_size": env.ensemble_size,
        "DA_iter": env.DA_iter,
        "DA_alpha": env.DA_alpha,
        "DA_error": env.DA_error,
        "prior_median": env.prior_median,
        "prior_scatter": env.prior_scatter,
        "perf_counter_sec": perf_counter,
        "process_time_sec": process_time,
        "episode_rewards": episode_rewards.tolist(),
        "Q": Q_json,
    }

    env_version = env_name[-2:]
    json_object_results = json.dumps(dict_results, indent=4)
    if json_name:
        file_name = f"data/results_{env_version}_{json_name}.json"
    else:
        file_name = f"data/results_{env_version}.json"
    with open(file_name, "w") as f:
        f.write(json_object_results)

    return None


def RL_training(
    env_name,
    n_episodes,
    discount_factor,
    epsilon_max,
    epsilon_min,
    epsilon_decay,
    alpha_max,
    alpha_min,
    alpha_decay,
    save=False,
    json_name=None,
):
    """
    Reinforcement learning (Q-learning) algorithm for a synthetic experiment to train
    a drone to quantify the emission rate of a greenhouse gas hotspot
    in custom Gym environment AIMeet2022.

    Parameters
    ---------
    env_name: string
    n_episodes, discount_factor, epsilon_max, epsilon_min, epsilon_decay, alpha_max, alpha_min, alpha_decay
        See Q_learning function
    save: boolean
        If True, results are saved in folder data/
    json_name: string
        Added to the file name of the json output results file
    """

    env = gym.make(env_name)

    perf_counter, process_time, episode_rewards, Q = Q_learning(
        env,
        n_episodes,
        discount_factor,
        epsilon_max,
        epsilon_min,
        epsilon_decay,
        alpha_max,
        alpha_min,
        alpha_decay,
    )

    env.close()

    if save:
        save_results(
            env,
            env_name,
            json_name,
            n_episodes,
            discount_factor,
            epsilon_max,
            epsilon_min,
            epsilon_decay,
            alpha_max,
            alpha_min,
            alpha_decay,
            perf_counter,
            process_time,
            episode_rewards,
            Q,
        )

    return perf_counter, process_time, episode_rewards, Q


if __name__ == "__main__":
    RL_training(
        env_name="NMI-v0",
        n_episodes=500,
        discount_factor=1.0,
        epsilon_max=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9988,
        alpha_max=0.1,
        alpha_min=0.1,
        alpha_decay=1.0,
        save=True,
        json_name="example",
    )
