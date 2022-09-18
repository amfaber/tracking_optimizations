#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import uuid
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import trackpy as tp
from pathlib import Path
import os

import hatzakis_lab_tracking as hlt




params = hlt.Params(
    lip_int_size = 8,  #originally 15 for first attemps
    lip_BG_size = 70,   # originally 40 for first attemps
    gap_size = 2, # adding gap
    
    dynamic_sep = 7,   # afstand mellem centrum af to partikler, 7 er meget lidt så skal være højere ved lavere densitet
    dynamic_mean_multiplier = 12,  #hvor mange partikler finder den, around 1-3, lavere giver flere partikler
    dynamic_object_size = 11, #diameter used in tp.locate, odd integer
    dynamic_search_range = 4,
    dynamic_memory = 0,
    
    static_sep = 8,
    static_mean_multiplier = 4,
    static_object_size = 11,

    n_processes = os.cpu_count(),
    fit_processes = 4,
)



def fix_ax_probs(ax,x_label,y_label):
    ax.set_ylabel(y_label,size = 10)
    ax.set_xlabel(x_label,size = 10)
    ax.tick_params(axis= "both", which = "major", labelsize = 8)
    ax.tick_params(axis= "both", which = "minor", labelsize = 8)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = "upper right",frameon = False)
    return ax


def extract_traces_average(static_paths, dynamic_paths, no_particle_path, save_path, params):
    save_path = Path(save_path)
    os.makedirs(save_path, exist_ok = True)
    ### no particle video
    Gs_mean, offs_mean = hlt.load_or_fit_random_pixels(no_particle_path, save_path, params.fit_processes)

    for counter, (dynamic_path, static_path) in tqdm(enumerate(zip(dynamic_paths, static_paths))):
        if not os.path.exists(save_path / f"{counter}_"):           # creates a folder for saving the stuff in if it does not exist
            os.makedirs(save_path / f"{counter}_")
        
        static_video = hlt.image_loader_video(static_path)
        static_averaged_over_frames= np.mean(static_video, axis = 0) #mean of all frames in dynamic_path, we detect on an average still image of all videos
        del static_video
        static_overall_mean = np.mean(static_averaged_over_frames) #mean of pixel value

        static_df = tp.locate(static_averaged_over_frames, params.static_object_size, \
            minmass = static_overall_mean * params.static_mean_multiplier, \
            separation=params.static_sep) #11 is diameter of roi used for position determination, odd number
        static_df = tp.refine_leastsq(static_df, static_averaged_over_frames, 5) # refine coordinates of close particles, adds the possibility  of a local mean noise correction termend "background"

        static_df['particle'] = range(len(static_df))

        print("Found particles: ", len(static_df))

        data_type = np.float64
        chunker = hlt.VideoChunker(dynamic_path,
        gb_limit = 2,
        # n_chunks = 1,
        dtype = data_type,
        offset = offs_mean,
        division = Gs_mean)

        chunker.get_illumination_profile_across_chunks()
        dynamic_signal_in_static_positions = []
        dynamic_features = []
        for frame_correction, vid_chunk in chunker:
            dynamic_signal_chunk = []
            dynamic_features.append(
                tp.batch(
                    vid_chunk,
                    params.dynamic_object_size, 
                    minmass = chunker.mean_frame.mean() * params.dynamic_mean_multiplier, 
                    separation = params.dynamic_sep, 
                    processes = params.n_processes
                    )
                )
            dynamic_features[-1]["frame"] += frame_correction

            for i in range(len(vid_chunk)):
                dynamic_signal_chunk.append(static_df[["y", "x", "particle"]].copy())
                dynamic_signal_chunk[-1]['frame'] = i
            dynamic_signal_chunk = pd.concat(dynamic_signal_chunk, ignore_index = True)
            for _, i_pos, j_pos, particle in static_df[["y", "x", "particle"]].itertuples():
                part_mask = dynamic_signal_chunk["particle"] == particle
                array_list = hlt.get_intensity_and_background_constant_position(
                                                i_pos, j_pos, vid_chunk, params, return_means = True)
                roi_intensities = pd.DataFrame(array_list).T
                dynamic_signal_chunk = hlt.modify_track_df_with_roi_intensities(
                            dynamic_signal_chunk, roi_intensities, "red", include_means = True, row_mask = part_mask)
                

            dynamic_signal_chunk["frame"] += frame_correction
            dynamic_signal_in_static_positions.append(dynamic_signal_chunk)
            del vid_chunk
        dynamic_features = pd.concat(dynamic_features)
        dynamic_tracks = tp.link_df(dynamic_features, params.dynamic_search_range, memory = params.dynamic_search_range)

        dynamic_signal_in_static_positions = pd.concat(dynamic_signal_in_static_positions, ignore_index = True)
        dynamic_signal_in_static_positions.sort_values(["frame", "particle"], ascending = True, ignore_index = True)
        dynamic_signal_in_static_positions['video_counter'] = counter

        this_uuid = uuid.uuid4().hex
        dynamic_tracks['particle'] = dynamic_tracks["particle"].astype(str) + f"_{this_uuid}"
        
        this_uuid = uuid.uuid4().hex
        dynamic_signal_in_static_positions['particle'] = dynamic_signal_in_static_positions["particle"].astype(str) + f"_{this_uuid}"
        static_df["particle"] = static_df["particle"].astype(str) + f"_{this_uuid}"

        
        dynamic_tracks.to_csv(save_path / f"{counter}_/_dynamic_tracks.csv")
        dynamic_signal_in_static_positions.to_csv(save_path / f"{counter}_/_dynamic_signal_in_static_positions.csv", \
             header=True, index=None, sep=',', mode='w')
        static_df.to_csv(save_path / f"{counter}_/_static_df.csv")

        params.to_yaml(save_path / f"{counter}_/params.yml")

        plot_static_particles(static_averaged_over_frames, static_df, counter, save_path)
        plot_hists(dynamic_signal_in_static_positions, counter, save_path)
        plot_dynamic_signal_for_static_particle(dynamic_signal_in_static_positions, static_averaged_over_frames, params, counter, save_path)


def plot_static_particles(static_averaged_over_frames, static_df, counter, save_path):
    fig, ax = plt.subplots(figsize = (10,10))
    ax.imshow(static_averaged_over_frames)
    ax.plot(static_df.x.values, static_df.y.values, '.', color = "red", alpha = 0.5)
    fig.tight_layout()
    fig.savefig(save_path / f"{counter}_/full_locate.png")

    plt.close('all')
    
def plot_dynamic_signal_for_static_particle(
    dynamic_signal_in_static_positions, 
    static_averaged_over_frames, 
    params, 
    counter, 
    save_path,
    particle_uuid = False):

    for par, dat in dynamic_signal_in_static_positions.groupby('particle'):

        fig,ax = plt.subplots(4,1,figsize = (9,9))

        ax[0].plot(dat.frame.values, dat.red_int_corrected.values, color = "firebrick", alpha =0.8, label ='Corrected')


        ax[1].plot(dat.frame.values, dat.red_int.values, color = "salmon", alpha =0.8, label ='Raw')
        ax[0].legend()
        ax[1].legend()
        ax[0] = fix_ax_probs(ax[0], 'Frame', 'Intensity')
        ax[1] = fix_ax_probs(ax[1], 'Frame', 'Intensity')

        ax[2].plot(dat.frame.values, dat.red_int_mean.values, color = "darkred", alpha =0.8, label ='Mean')
        ax[2].legend()
        ax[2] = fix_ax_probs(ax[2], 'Frame', 'Intensity')
        
        vid_shape = static_averaged_over_frames.shape
        vid_shape = (1, *vid_shape)
        sig_mask, bg_mask = hlt.make_smallmask_sig_bg(dat["y"].iat[0], dat["x"].iat[0], vid_shape, params)
        
        vid_window = static_averaged_over_frames[bg_mask[:2]]
        ax[3].imshow(vid_window)
        ax[3].imshow(np.where(sig_mask[2], 1, np.nan), vmin = 0, vmax = 1, cmap = "Greens", alpha = 0.4)
        ax[3].imshow(np.where(bg_mask[2], 1, np.nan), vmin = 0, vmax = 1, cmap = "Reds", alpha = 0.4)
        fig.tight_layout()
        if not particle_uuid:
            par = par.split('_')[0]
        fig.savefig(save_path / f"{counter}_" / f"{par}.png")
        plt.close('all')

def plot_hists(dynamic_signal_in_static_positions, counter, save_path):
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(dynamic_signal_in_static_positions.red_int_corrected.values, 50, density = True, alpha = 0.7, ec = 'black', color = "firebrick", label = "Corrected")
    ax = fix_ax_probs(ax, 'Intensity', 'Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path / f"{counter}_" / 'corrected_hist.png')

    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(dynamic_signal_in_static_positions.red_int.values, 50, density = True, alpha = 0.7,ec = 'black',color = "salmon",label = "Raw")
    ax = fix_ax_probs(ax, 'Intensity', 'Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path / f"{counter}_" / 'raw_hist.png')

    plt.close('all')

# ----------------------------- Run -------------------------------- #

if __name__ == "__main__":
    folder = Path("sample_vids")
    no_particle_path = str(folder / "Experiment_Process_001_20220823.tif")
    video_static = [str(folder / "c_20.tif")]
    video_signal = [str(folder / "s_20.tif")]

    save_path = str(folder / "chunking_new")

    extract_traces_average(video_static, video_signal, no_particle_path, save_path, params = params) #check mean multiplier, size threshold and number of particles
