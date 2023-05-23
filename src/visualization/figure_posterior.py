import numpy as np
import json
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import itertools
from src.DA.DA_tools import get_lognormal_distribution, lognormal_to_normal

import gymnasium as gym
import src.RL.gym_hotspot

from src.gaussian_plume_model.get_ppm_concentration import get_ppm_concentration_field
from src.gaussian_plume_model.get_ppm_concentration import get_ppm_concentration

def get_conc_field(file_name, flux_true):

    # Load json results file and read out parameters
    f = open(file_name)
    data = json.load(f)

    nx = data["nx"]
    ny = data["ny"]
    nz = data["nz"]
    dx = data["dx"]
    dy = data["dy"]
    dz = data["dz"]
    z_range = data["z_range"]
    wind_speed = data["wind_speed"]
    wind_direction = data["wind_direction"]
    x_source = data["x_source"]
    y_source = data["y_source"]
    z_source = data["z_source"]
    stability = data["stability"]
    background_ppm = data["background_ppm"]
    temperature = data["temperature"]
    pressure = data["pressure"]
    molar_mass = data["molar_mass"]
    obs_sigma = data["obs_sigma"]

    # Get concentration field for left subplot
    concentration_field = get_ppm_concentration_field(
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        flux_true,
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
    # Reverse y-index such that [0, 0] is not top left but bottom left
    # and get 2D array
    concentration_field = concentration_field[:, ::-1, 0]
    concentration_field = np.swapaxes(concentration_field, 0, 1)

    # Get contour at background_ppm + obs_sigma for left subplot
    x = np.linspace(0, nx * dy, 1000, endpoint=True)
    y = np.linspace(0, ny * dy, 1000, endpoint=True)
    contour = np.zeros((len(x), len(y)))
    for i, xx in enumerate(x):
        xx = int(xx)
        for j, yy in enumerate(y):
            yy = int(yy)
            contour[i, j] = get_ppm_concentration(
                flux_true,
                wind_speed,
                wind_direction,
                xx,
                yy,
                z_range[0],
                x_source,
                y_source,
                z_source,
                stability,
                background_ppm,
                temperature,
                pressure,
                molar_mass,
            )
    X, Y = np.meshgrid(x / dx, y / dy)
    levels = [background_ppm + obs_sigma]
    # Reverse y-index such that [0, 0] is not top left but bottom left
    contour = contour[:, ::-1]
    contour = np.swapaxes(contour, 0, 1)

    return concentration_field, X, Y, contour, levels


def get_results_for_flux(file_name, flux_true, x_init, y_init):

    # Load json results file and read out parameters
    f = open(file_name)
    data = json.load(f)

    ny = data["ny"]
    mean_flux = data[f"mean_spaghetti_{flux_true}__{x_init}{y_init}"]
    std_flux = data[f"std_spaghetti_{flux_true}__{x_init}{y_init}"]
    locs = data[f"optimal_policy_locs_{x_init}{y_init}"]

    x_locs = [loc[0] + 0.5 for loc in locs]
    y_locs = [ny - loc[1] - 0.5 for loc in locs] 

    return mean_flux, std_flux, x_locs, y_locs

def make_figure_results(file_name, n_eval, fig_name=None):
    """
    Make figure with true concentration field and optimal path of agent in left subfigure
    and prior and posterior probability distribution for emission rate in right subfigure

    Parameters
    ----------
    file_name: string
        json file with results from training, evaluation and processing phase
    """

    # Load json results file and read out parameters
    f = open(file_name)
    data = json.load(f)

    ny = data["ny"]
    dx = data["dx"]
    dy = data["dy"]
    x_source = data["x_source"]
    y_source = data["y_source"]
    # Get locations of the agent (middle of grid cell)
    # Reverse y-index such that [0, 0] is not top left but bottom left
    x_loc_source = x_source / dx  # from meter to grid index
    y_loc_source = ny - (y_source / dy)

    mean_0_00, std_0_00, x_locs_0_00, y_locs_0_00 = get_results_for_flux(file_name=file_name, flux_true=2500, x_init=0, y_init=0)
    assert n_eval <= len(mean_0_00)

    concentration_field_2500, X_2500, Y_2500, contour_2500, levels_2500 = get_conc_field(file_name=file_name, flux_true=2500)

    # Figure settings
    SMALL_SIZE = 12
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 12

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, (ax0, ax1) = plt.subplots(
        nrows=1, ncols=2, sharex=False, sharey=False, figsize=(12, 5)
    )

    ax0.set_aspect("equal")
    ax0 = sns.heatmap(
        concentration_field_2500,
        linewidth=0.5,
        cmap="Reds",
        annot=False,
        norm=LogNorm(),
        ax=ax0,
    )
    cbar = ax0.collections[0].colorbar
    cbar.mappable.set_clim(vmin=400, vmax=650)
    cbar.set_ticks([400, 500, 600])
    cbar.set_ticklabels(["400", "500", "600"])
    cbar.set_label(r"CO$_2$ concentration [ppm]")

    ax0.set_xticks([0, 5, 10])
    ax0.set_xticklabels(["0", "500", "1000"])
    ax0.set_yticks([0, 5, 10])
    ax0.set_yticklabels(["1000", "500", "0"], rotation=0)
    ax0.set_xlabel(r"x [m]")
    ax0.set_ylabel(r"y [m]")

    CS = ax0.contour(X_2500, Y_2500, contour_2500, levels_2500, colors="grey")
    ax0.clabel(
        CS,
        colors="grey",
        inline=True,
        fmt=r"$400 + 30/\sqrt{12}$ ppm",
        fontsize=12,
        manual=[(9, 4)],
    )
    ax0.plot(x_locs_0_00, y_locs_0_00, c="#1b9e77", linewidth=2, alpha=0.5, linestyle="-")
    ax0.scatter(
        x_loc_source,
        y_loc_source,
        marker="x",
        color="black",
        s=70,
        label="hotspot cell",
    )
    ax0.scatter(x_locs_0_00[0], y_locs_0_00[0], marker="s", c="#1b9e77", s=70, label="initial location")
    ax0.tick_params(left=False, bottom=False)

    # These are the mean and std of the prior (lognormal) distribution transformed to a normal distribution
    prior_mean = 6.92
    prior_std = 1.10

    # The probability distributions are scaled by /10 to go from
    # g/s to g/(s*m2) (= /(dx*dy) = /10000) to mg/(s*m2) (= *1000)
    x = np.linspace(1, 400, 10000)
    ax1.axvline(
        250, color="black", linestyle="--", linewidth=2, label="true flux"
    )
    ax1.plot(
            x,
            lognorm.pdf(x, s=prior_std, loc=0, scale=np.exp(prior_mean) / 10),
            color="black",
            linewidth=2,
            label="prior",
    )
    for i in range(n_eval):
        ax1.plot(
            x,
            lognorm.pdf(x, s=std_0_00[i], loc=0, scale=np.exp(mean_0_00[i])),
            color="#1b9e77",
            alpha=0.3,
            linestyle="-",
        )
    ax1.legend(loc='upper left')
    ax1.set_xlabel(r"Hotspot flux [mg CO$_2$ m$^{-2}$ s$^{-1}$]")
    ax1.set_ylabel(r"Probability density")

    plt.tight_layout()
    plt.savefig(f"src/visualization/results_{fig_name}.png")
    plt.show()

    return None


if __name__ == "__main__":
    file_name = "data/results_v0_CRPS.json"
    fig_name = "CRPS"
    n_eval = 5
    make_figure_results(file_name, n_eval=n_eval, fig_name=fig_name)
