from minmass_estimation import calc_minmass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
import os
import json
import tifffile as tif
from skimage import io
from gen_particle import gen_particle_sim, measure_SNR
import trackpy as tp
import glob
import pandas as pd

snr_limit = 3.0  # Signals must at least be x times the noise std
exp_particles = 30
diameter = 15
preprocess = True
save = False


"return signal positions at frame 0"


def show_frame(
    frame,
    img,
    labels,
    strength,
    noise,
    snr_limit,
    n_particles=20,
    diameter=diameter,
    preprocess=True,
):
    try:
        minmass, _, image = calc_minmass(
            img[0], n_particles, snr=snr_limit, diameter=diameter, preprocess=preprocess
        )
        no_particles = False
    except ValueError:
        minmass = np.max(img)
        image = img[frame]
        no_particles = True

    df = tp.locate(
        img[frame],
        diameter=diameter,
        minmass=minmass,
        invert=False,
        noise_size=1,
        preprocess=preprocess,
    )
    df_all = []
    df_all_zero = []
    for i in range(img.shape[0]):
        df_all.append(
            tp.locate(
                img[i],
                diameter=diameter,
                minmass=minmass,
                invert=False,
                noise_size=1,
                preprocess=preprocess,
            )
        )
        df_all_zero.append(
            tp.locate(
                img[i],
                diameter=diameter,
                minmass=0,
                invert=False,
                noise_size=1,
                preprocess=preprocess,
            )
        )

    df_all = pd.concat(df_all)
    df_all_zero = pd.concat(df_all_zero)
    print(df_all.shape)
    print(img.shape)
    mass = df.mass
    signal = np.mean(df.signal)
    snr = np.mean(signal / float(noise))
    print(df.keys())
    x_pred = df.x
    y_pred = df.y
    sig_pos = np.where(labels[frame] != 0)
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax[0, 0].imshow(img[frame], cmap="gray")
    ax[0, 0].text(
        0.05, 0.90, f"SNR: {snr:.2f}", transform=ax[0, 0].transAxes, color="w"
    )
    ax[0, 0].text(
        0.05,
        0.85,
        f"Integrated signal strength: {int(strength):.0f}",
        transform=ax[0, 0].transAxes,
        color="w",
    )
    ax[1, 0].imshow(img[frame], cmap="gray")

    ax[1, 0].scatter(
        sig_pos[1], sig_pos[0], s=60, facecolors="none", edgecolors="yellow"
    )
    ax[1, 1].imshow(img[frame], cmap="gray")
    ax[1, 1].set_title("Raw image w detections")

    ax[1, 1].scatter(x_pred, y_pred, s=80, facecolors="none", edgecolors="yellow")

    ax[0, 1].imshow(image, cmap="gray")
    # ax[0, 1].scatter(x_pred, y_pred, s=80, facecolors="none", edgecolors="green")
    ax[0, 0].set_title("Raw image")
    ax[1, 0].set_title("Raw image with signal positions")
    ax[0, 1].set_title("Preprocessed image w detections")
    if no_particles:
        ax[0, 2].set_title("No particles found")
        ax[1, 2].set_title("No particles found")
    if not no_particles:
        ax[0, 2].set_title("Mass histogram")
        ax[0, 2].hist(
            df_all.mass,
            bins=20,
            # color="r",
            alpha=0.5,
            log=False,
            range=(0.9 * np.min(df_all_zero.mass), 1.1 * np.max(df_all_zero.mass)),
            label="Detected mass",
        )
        ax[0, 2].axvline(
            np.float(strength),
            0,
            1,
            color="k",
            linestyle="--",
            label="True signal mass",
        )
        ax[0, 2].hist(
            df_all_zero.mass,
            bins=20,
            # color="r",
            alpha=0.5,
            log=False,
            range=(0.9 * np.min(df_all_zero.mass), 1.1 * np.max(df_all_zero.mass)),
        )
        ax[0, 2].legend()

        ax[0, 2].set(xlim=(0, 1.1 * np.max(df.mass)))
        ax[1, 2].set_title("Raw mass histogram")
        ax[1, 2].hist(
            df_all.raw_mass,
            bins=20,
            color="r",
            alpha=0.5,
            log=False,
            density=False,
            label="Detected raw mass",
            zorder=3,
            # range=(0.9 * np.min(df.signal), 1.1 * np.max(df.signal)),
        )
        ax[1, 2].hist(
            df_all_zero.raw_mass,
            bins=20,
            color="b",
            alpha=0.5,
            log=False,
            density=False,
            # range=(0.9 * np.min(df.signal), 1.1 * np.max(df.signal)),
        )
        ax[1, 2].legend()
    # ax[2].hist(image.flatten(), bins=30, color="b", alpha=0.5, log=True)
    # ax[2].set_title("H)
    if save:
        plt.savefig(
            f"minmass_tests/{file}_test.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()
    return


files = glob.glob("D:/Emily/simtests/*.tif")
print(files)

for file in files:
    if file.endswith("labels.tif"):
        continue
    else:
        img = io.imread(file, plugin="tifffile")
        print(img.shape)
        intensity = file.split("_")[3]
        noise = file.split("_")[6]
        print(intensity)
        labels = io.imread(file[:-4] + "labels.tif")
        show_frame(
            0, img, labels, intensity, noise, snr_limit, n_particles=exp_particles
        )
