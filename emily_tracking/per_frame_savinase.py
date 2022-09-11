from pathlib import Path
import hatzakis_lab_tracking as hlt
import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
import hashlib
from tqdm import tqdm
from functools import partial
import os

if __name__ == "__main__":
    folder = Path("sample_vids")
    no_par_video = str(folder / "Experiment_Process_001_20220823.tif")
    video_location = str(folder / "c_20.tif")
    video_signal = str(folder / "s_20.tif")
    save_path = str(folder / "output_new" / "full_tracked")
    os.makedirs(save_path, exist_ok = True)
    params = hlt.Params(
        lip_int_size = 9,  #originally 15 for first attemps
        lip_BG_size = 80,   # originally 40 for first attemps
        gap_size = 2, # adding gap
        sep = 15,   # afstand mellem centrum af to partikler, 7 er meget lidt så skal være højere ved lavere densitet
        mean_multiplier = 10,  #hvor mange partikler finder den, around 1-3, lavere giver flere partikler
        object_size = 11, #diameter used in tp.locate, odd integer
        n_processes = 8,
        fit_processes = 4,
        save_path = save_path,
        pixel_size = 1,
    )
    np_particle_movie = hlt.image_loader_video(no_par_video)
    movie_hash = hashlib.sha1(np_particle_movie).hexdigest()
    try:
        with open(Path(save_path) / "random_pix_fits.pkl", "rb") as file:
            Gs_mean, offs_mean, hash = pickle.load(file)
        if hash != movie_hash:
            raise ValueError
    except (FileNotFoundError, ValueError):
        n_random_pix = 400
        with mp.Pool(params.fit_processes, initializer = partial(np.seterr, all = "ignore")) as pool:
            pixfits = list(tqdm(pool.imap(partial(hlt.fit_random_pixel, np_particle_movie), range(n_random_pix)), total = n_random_pix))
        # pixfits = [fit_random_pixel(np_particle_movie) for i in tqdm(range(n_random_pix))] #400 pixels to fit


        Gs, offs, phots, pvals = (
        np.array([1 / res.x[1] for res, p in pixfits]),
        np.array([res.x[2] for res, p in pixfits]),
        np.array([res.x[0] for res, p in pixfits]),
        np.array([p for res, p in pixfits]),
        )
        Gs_mean = np.mean(Gs[pvals > 0.01])
        offs_mean = np.mean(offs[pvals > 0.01])
        with open(Path(save_path) / "random_pix_fits.pkl", "wb") as file:
            pickle.dump((Gs_mean, offs_mean, movie_hash), file)
    
    data_type = np.float32
    video = hlt.image_loader_video(str(video_signal)).astype(dtype = data_type)
    # should not be needed for Sara
    np.subtract(video, offs_mean.astype(data_type), out = video)
    np.divide(video, Gs_mean.astype(data_type), out = video)
    photon_movie = video
    # photon_movie = (video - offs_mean.astype(data_type)) / Gs_mean.astype(data_type)
    corrected_mov = hlt.Correct_illumination_profile(photon_movie, 0, save_path)
    # hlt.calibration(params, video = corrected_mov, save = False)
    tracked_df, msd_df = hlt.tracker(corrected_mov, "0", params)
    tracked_df.to_csv(Path(params.save_path)  / "df.csv")


