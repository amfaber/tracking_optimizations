import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from matplotlib.animation import FuncAnimation
import imageio
import os
import json
import tifffile as tif
from skimage import io


def create_trajectory(mean_trans, sigma_trans, fov_dimx, fov_dimy, steps):
    rots = np.random.uniform(-np.pi, np.pi, steps)
    trans = np.random.normal(mean_trans, sigma_trans, steps)
    xs = np.cumsum(np.cos(rots) * trans) + np.random.uniform(
        0.1 * fov_dimx, 0.9 * fov_dimx
    )
    ys = np.cumsum(np.sin(rots) * trans) + np.random.uniform(
        0.1 * fov_dimy, 0.9 * fov_dimy
    )
    return xs, ys


# def create_background(fov_dimx, fov_dimy, mu_noise, sigma_noise, steps):
#     return np.random.normal(mu_noise, sigma_noise, [fov_dimx, fov_dimy, steps])


def create_background(fov_dimx, fov_dimy, mu_noise, sigma_noise, steps):
    return (
        np.random.poisson(sigma_noise * sigma_noise, [fov_dimx, fov_dimy, steps])
        + mu_noise
    )


def convert_uint8(array):
    ### (0-255)
    min = np.min(array)
    array = array - min
    array = array / np.max(array)
    array = array * 255
    array = np.array(array, dtype=np.uint8)
    return array


def convert_uint16(array):
    ### (0-255)
    min = np.min(array)
    array = array - min
    array = array / np.max(array)
    array = array * 65535
    array = np.array(array, dtype=np.uint16)
    return array


def save_image(filename, img):
    imageio.imwrite(filename + ".jpg", img)
    return


def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = y_ - y
    x_pad = x_ - x
    return np.pad(
        a,
        ((y_pad // 2, y_pad // 2 + y_pad % 2), (x_pad // 2, x_pad // 2 + x_pad % 2)),
        mode="constant",
    )


def mkdirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def save_vid(filename, img, fps):
    if np.min(img) != 0:
        img = convert_uint8(img)
    imageio.mimwrite(filename + ".mp4", img.T, fps=fps)
    return


def measure_SNR(s, a):
    ss = s[a != 0]
    bs = s[a == 0]
    snr = (np.mean(ss) - np.mean(bs)) / np.std(bs)
    return np.array(snr, dtype=np.float16)


def measure_peak(s, a):
    particles = np.unique(a)
    peaks = []
    for p in particles:
        if p != 0:
            peaks.append(np.max(s[a == p]))
    return np.array(peaks, dtype=np.float16)


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def gen_particle_sim(
    n_p,
    mean_trans,
    sigma_trans,
    signal_strength,
    mu_noise,
    sigma_noise,
    sigma_psf,
    fov_dimx,
    fov_dimy,
    n_steps,
    existing_img=None,
    approximate_background=False,
    pulsate=0,
):
    if existing_img is not None:
        n_steps = existing_img.shape[0]
    if approximate_background is True and existing_img is not None:
        bckdist = np.percentile(existing_img, 99.99)
        signal_std = np.std(existing_img[existing_img > bckdist])
        signal_mean = np.mean(existing_img[existing_img > bckdist])
        existing_img[existing_img > bckdist] = np.mean(existing_img)
        background = existing_img.transpose(1, 2, 0)
        background_shuffled = np.copy(background).flatten()
        idx = np.random.permutation(background_shuffled.shape[0])
        background_shuffled = background_shuffled[idx]
        background_shuffled = background_shuffled.reshape(background.shape)
        background = background_shuffled
        print("std of signals", signal_std, "mean of signals", signal_mean)
        existing_img = np.zeros_like(existing_img)
    else:
        background = create_background(
            fov_dimx=fov_dimx,
            fov_dimy=fov_dimy,
            mu_noise=mu_noise,
            sigma_noise=sigma_noise,
            steps=n_steps,
        )
    class_id = 0
    # Allocate arrays
    annotations = np.zeros_like(background)
    signals = np.zeros_like(background)
    bboxes = np.zeros((n_steps, n_p, 6))

    print("Insert trajectories")
    # Move trajectories to grid representation
    for i in range(n_p):
        xs, ys = create_trajectory(
            mean_trans,
            sigma_trans=sigma_trans,
            fov_dimx=fov_dimx,
            fov_dimy=fov_dimy,
            steps=n_steps,
        )
        for c, x in enumerate(xs):
            print(f"inserting particle, {i}, at frame, {c}")
            try:
                signals[int(np.round(x)), int(np.round(ys[c])), c] = signal_strength[
                    i
                ] * np.random.uniform(1 - pulsate, 1 + pulsate)
                annotations[int(np.round(x)), int(np.round(ys[c])), c] = i + 1
            except Exception as e:
                print(e)
                pass
    for frame in range(n_steps):
        print("get bboxes")
        # Picks out centers of bounding boxes
        locs = np.argwhere(signals[:, :, frame] != 0)

        # Centers, width and height are set relative to image dimensions
        locs = locs / np.array([fov_dimx, fov_dimy])
        w, h = 6 * sigma_psf / fov_dimx, 6 * sigma_psf / fov_dimy
        for counter, xywh in enumerate(locs):
            bboxes[frame, counter, :] = [
                0,
                class_id,
                locs[counter, 0],
                locs[counter, 1],
                w,
                h,
            ]

        # Applies psf to signals and adds background noise
        psfs = ndi.gaussian_filter(signals[:, :, frame], sigma_psf)
        background[:, :, frame] = background[:, :, frame] + psfs

        if existing_img is not None:
            background[:, :, frame] = (
                existing_img[frame, :, :] + background[:, :, frame]
            )

    # background = convert_uint16(background)
    snr = measure_SNR(background, annotations)
    particle_peaks = measure_peak(background, annotations)
    return background, annotations, np.round(snr, 2), particle_peaks, bboxes


n_particles = 1
mu_trans = 0
sigma_trans = 0
integrated_signal_strength = 0
ss_list = np.random.normal(
    integrated_signal_strength, 0.0 * integrated_signal_strength, n_particles
)
offset = 1000
sigma_noise = 50
sigma_psf = 1.1
fov_dimx = 512
fov_dimy = 512
n_steps = 50

save = True

if __name__ == "__main__":

    b, a, snr, peaks, bboxes = gen_particle_sim(
        n_p=n_particles,
        mean_trans=mu_trans,
        sigma_trans=sigma_trans,
        signal_strength=ss_list,
        mu_noise=offset,
        sigma_noise=sigma_noise,
        sigma_psf=sigma_psf,
        fov_dimx=fov_dimx,
        fov_dimy=fov_dimy,
        n_steps=n_steps,
        existing_img=None,
        pulsate=0,
    )

    # print(np.max(b))
    # print(snr, peaks)
    # plt.imshow(b[:, :, 0])
    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(b[:, :, 0])
    ax[1].hist(b[:, :, :].flatten(), bins=256)
    plt.show()

    # imgids = np.array(np.arange(0, 100, 1), dtype=np.int32)
    # # print(imgids)

    # xp_img2 = tif.imread("D:/Emily/no_particles.tif", key=imgids)

    # sigma_noise = np.std(exp_img2)
    # print(sigma_noise)
    # b, a, snr, peaks, bboxes = gen_particle_sim(
    #     n_p=n_particles,
    #     mean_trans=mu_trans,
    #     sigma_trans=sigma_trans,
    #     signal_strength=ss_list,
    #     sigma_noise=sigma_noise,
    #     sigma_psf=sigma_psf,
    #     fov_dimx=fov_dimx,
    #     fov_dimy=fov_dimy,
    #     n_steps=n_steps,
    #     existing_img=exp_img2,
    # )
    # print("99th percentile of data", np.percentile(exp_img2, 99.9))

    # print("99th percentile of simulated", np.percentile(b, 99.9))
    # print(snr, peaks)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].hist(exp_img2.flatten(), bins=100, range=(2100, 3000))
    # ax[1].hist(b.flatten(), bins=100, range=(2100, 3000))
    # plt.show()
    # fig, ax = plt.subplots(1, 3, figsize=(7, 7))
    # ax.imshow(exp_img2[5, :400, :400], origin="lower", cmap="gray")
    # plt.show()

    # fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    # for i in range(1):
    #     # set overall title
    #     fig.suptitle(
    #         f"Experimental data with additional simulated signals of SNR = {snr:.2f}"
    #     )
    #     ax[0].clear()
    #     ax[0].imshow(b[:, :, i], origin="lower", cmap="gray")
    #     ax[0].title.set_text("Data with simulated signals and markers")
    #     ax[0].scatter(
    #         bboxes[0, :, 3] * fov_dimx,
    #         bboxes[0, :, 2] * fov_dimy,
    #         s=80,
    #         facecolors="none",
    #         edgecolors="r",
    #     )
    #     ax[1].imshow(b[:, :, i], origin="lower", cmap="gray")
    #     ax[1].title.set_text("Data with simulated signals")
    #     ax[2].imshow(
    #         exp_img2[
    #             i,
    #             :,
    #         ],
    #         origin="lower",
    #         cmap="gray",
    #     )
    #     ax[2].title.set_text("Raw data")
    #     plt.savefig(
    #         "D:/Emily/Data_with_added_signals_and_markers.png", dpi=300, bbox_inches="tight"
    #     )
    #     plt.pause(0.01)

    # tif.imsave(
    #     "D:/Emily/particles_good_added_signals.tif",
    #     b.transpose(2, 0, 1).astype(np.uint16),
    #     imagej=True,
    # )

    # tif.imsave(
    #     "D:/Emily/particles_good_added_signals_labels.tif",
    #     a.transpose(2, 0, 1).astype(np.uint16),
    #     imagej=True,
    # )

    # b, a, snr, peaks, bboxes = gen_particle_sim(
    #     n_p=n_particles,
    #     mean_trans=mu_trans,
    #     sigma_trans=sigma_trans,
    #     signal_strength=ss_list,
    #     mu_noise=mu_noise,
    #     sigma_noise=sigma_noise,
    #     sigma_psf=sigma_psf,
    #     fov_dimx=fov_dimx,
    #     fov_dimy=fov_dimy,
    #     n_steps=n_steps,
    #     existing_img=exp_img2,
    #     approximate_background=True, pulsate = 0.25
    # )

    if save:
        tif.imsave(
            f"D:/Emily/simulated_i_int_{integrated_signal_strength}_sig_noise_{sigma_noise}_.tif",
            b.transpose(2, 0, 1).astype(np.uint16),
            imagej=True,
            metadata={"axes": "TYX"},
        )

        tif.imsave(
            f"D:/Emily/simulated_i_int_{integrated_signal_strength}_sig_noise_{sigma_noise}_labels.tif",
            a.transpose(2, 0, 1).astype(np.uint16),
            imagej=True,
            metadata={"axes": "TYX"},
        )
