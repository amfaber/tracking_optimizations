#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:28:39 2021

@author: SørenBohr
"""

# """
# Created on Fri Jul  2 08:48:32 2021

# @author: frejabohr
# """
#%%
import pandas as pd
import numpy as np

import pickle
from time import time
import uuid
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import trackpy as tp
from pathlib import Path
import os
from functools import partial
import hashlib

import hatzakis_lab_tracking as hlt



#%%

params = hlt.Params(
    lip_int_size = 9,  #originally 15 for first attemps
    lip_BG_size = 80,   # originally 40 for first attemps
    gap_size = 2, # adding gap
    sep = 8,   # afstand mellem centrum af to partikler, 7 er meget lidt så skal være højere ved lavere densitet
    mean_multiplier = 4,  #hvor mange partikler finder den, around 1-3, lavere giver flere partikler
    object_size = 11, #diameter used in tp.locate, odd integer
    n_processes = 0,
    fit_processes = 4,
)


##Not used BG + gap

#15/6-21 changed from 6 to 10

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

#first reverse green videos


def cmask_plotter(index, array, BG_size, int_size,gap_size):
    a, b = index
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    r2 = x * x + y * y
    mask = r2 <= int_size  # radius squared - but making sure we dont do the calculation in the function - slow
    mask2 = r2 <= int_size + gap_size   # to make a "gab" between BG and roi

    BG_mask = (r2 <= BG_size)
    BG_mask = np.bitwise_xor(BG_mask, mask2)
    return mask, BG_mask



def set_zero_df(df):
    def correct_data(x):
        box = np.ones(2)/2

        #corr_val = np.min(np.convolve(x, box, mode='same')) #value of lowest state in trace

        return x-np.min(np.convolve(x, box, mode='same')), 0

       # if corr_val < 1000:
        #    return x-np.min(np.convolve(x, box, mode='same')), 0

        #elif corr_val > 1000 and corr_val < 3500:
         #   return x-np.min(np.convolve(x, box, mode='same')) + 2500, 1

        #elif corr_val > 3500 and corr_val < 6500:
         #   return x-np.min(np.convolve(x, box, mode='same')) + 5000, 2

        #elif corr_val > 6500:
         #   return x-np.min(np.convolve(x, box, mode='same')) + 7500, 3

    def gr(grp):
        grp['set_zero'], grp['lowest_state'] = correct_data(grp.red_int_gausscorrected.values)
        return grp
    return df.groupby(['particle']).apply(gr)

def plotParsDf(df,savepath):
    df = pd.read_csv(df)
    df = set_zero_df(df)
    grp = df.groupby('particle')

    for name,dat in tqdm(grp):
        fig,ax = plt.subplots(figsize = (6,3))
        ax.plot(dat.frame.values,dat.set_zero.values)
        fig.savefig(str(savepath+str(name)+'.png'))
        plt.close('all')




# ----------------------------- Main Function ------------------------------ #

#Function that extract traces
#extract_traces_average(video_location,video_signal, no_par_video, save_path)

def extract_traces_average(location, video, no_par_video, save_path, params):
    save_path = Path(save_path)
    counter = 0
    ### no particle video
    np_particle_movie = hlt.image_loader_video(no_par_video)
    movie_hash = hashlib.sha1(np_particle_movie).hexdigest()
    try:
        with open(save_path / "random_pix_fits.pkl", "rb") as file:
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
        with open(save_path / "random_pix_fits.pkl", "wb") as file:
            pickle.dump((Gs_mean, offs_mean, movie_hash), file)

    for video,loc in tqdm(zip(video,location)):
        if not os.path.exists(save_path / f"{counter}_"):                      # creates a folder for saving the stuff in if it does not exist
            os.makedirs(save_path / f"{counter}_")
        
        loc = hlt.image_loader_video(loc)
        ####only detect on first 500 frames. added 16/7-21
        loc_averaged_over_frames= np.mean(loc,axis = 0) #mean of all frames in video, we detect on an average still image of all videos
        del loc
        mean=np.mean(loc_averaged_over_frames) #mean of pixel value

        f = tp.locate(loc_averaged_over_frames, params.object_size, minmass=mean * params.mean_multiplier, separation=params.sep) #11 is diameter of roi used for position determination, odd number

        data_type = np.float64
        video = hlt.image_loader_video(video).astype(dtype = data_type)
        # should not be needed for Sara
        np.subtract(video, offs_mean.astype(data_type), out = video)
        np.divide(video, Gs_mean.astype(data_type), out = video)
        photon_movie = video
        # photon_movie = (video - offs_mean.astype(data_type)) / Gs_mean.astype(data_type)
        corrected_mov = hlt.Correct_illumination_profile(photon_movie, counter, save_path)


        #imsave(str(main_save_path+'smoothed.tif'),video[:1000]) #saves smoothed video

        ##end move to other function

#        plt.imshow(video_still)
#        tp.annotate(f, video_still)

        fig,ax = plt.subplots(figsize = (10,10))
        ax.imshow(loc_averaged_over_frames)
        ax.plot(f.x.values,f.y.values,'.',color = "red",alpha =0.5)
        fig.tight_layout()
        fig.savefig(save_path / f"{counter}_" / "full_locate_loc.pdf")
        fig.savefig(save_path / f"{counter}_" / "full_locate_loc.png")

        plt.close('all')
#

        # added SSRB 23-04-21 start
        #f = f[f.ecc<0.25] # accept only round-ish particles to avoid doubles
        f = tp.refine_leastsq(f,loc_averaged_over_frames,5) # refine coordinates of close particles, adds the possibility  of a local mean noise correction termend "background"


        fig,ax = plt.subplots(figsize = (10,10))
        ax.imshow(loc_averaged_over_frames)
        ax.plot(f.x.values,f.y.values,'.',color = "red",alpha = 0.5)
        fig.tight_layout()
        fig.savefig(save_path / f"{counter}_" / "full_locate.pdf")
        fig.savefig(save_path / f"{counter}_" / "full_locate.pdf")

        plt.close('all')



        bg_tmp = f['background'].values
        bg_corr_tmp = np.mean(bg_tmp[bg_tmp!=0])

        f =f.replace({'background': {0: bg_corr_tmp}})



        # added SSRB 23-04-21 end

        #f = f[f.mass > 3957] #excludes particles with a higher mass than this


        f['particle'] = range(len(f))

        print ("Found particles: ", len(f))
        new = []
        for i in range(len(corrected_mov)):
            f['frame'] = i
            new.append(f)
        new = pd.concat(new, ignore_index = True)

        with hlt.create_pool_context(params.n_processes, corrected_mov) as pool:
            new = hlt.signal_extractor_no_pos(corrected_mov, new, 'red', params, pool = pool, include_means = True)
        new['video_counter'] = counter


        a = new['particle'].tolist()
        z = uuid.uuid4().hex
        a = [str(str(x)+'_'+str(z)) for x in a]
        new['particle'] = a

        f["particle"] = f["particle"].astype(str) + f"_{z}"
        f[["particle", "signal"]].to_csv(save_path / "meaned_int.csv")


        # adding "set_zero column"
        full = set_zero_df(new)


        full.to_csv(save_path / f"{counter}_" / '_signal_df.csv', header=True, index=None, sep=',', mode='w')

        ## new
        if not os.path.exists(save_path / f"{counter}_"):                      # creates a folder for saving the stuff in if it does not exist
            os.makedirs(save_path / f"{counter}_")
        with open(save_path / f"{counter}_" / 'rois.txt', "w") as text_file:
            text_file.write(f"ROI: {params.lip_int_size}\n")
            text_file.write(f"BG ROI: {params.lip_BG_size}\n")
            text_file.write(f"GAP size: {params.gap_size}\n")
            text_file.write(f"sep: {params.sep}\n")
            text_file.write(f"meanmultiplier: {params.mean_multiplier}\n")
            text_file.write(f"diameter: {params.object_size}\n")

        grp = full.groupby('particle')
        import matplotlib
        my_cmap = matplotlib.cm.Reds.copy()
        my_cmap.set_under('k', alpha=0)
        my_cmap.set_bad('k', alpha=0)

        my_cmap2 = matplotlib.cm.Greens.copy()
        my_cmap2.set_under('k', alpha=0)
        my_cmap2.set_bad('k', alpha=0)
        size_maker = np.ones(loc_averaged_over_frames.shape)

        for par,dat in grp:
            x_aq = dat.x.values[1]

            y_aq = dat.y.values[1]
            ind = (y_aq,x_aq)  # dont ask - leave it here, it just makes sure the below runs


            mask, BG_mask = cmask_plotter(ind, size_maker, params.lip_BG_size, params.lip_int_size, params.gap_size)

            fig,ax =plt.subplots(4,1,figsize = (9,9))

            #
            ax[0].plot(dat.frame.values,dat.red_int_corrected.values,color = "firebrick",alpha =0.8,label ='Corrected')
            ax[0].plot(dat.frame.values,dat.red_int_gausscorrected.values,color = "grey",alpha =0.8,label ='Gaussian corrected')
            ax[0].plot(dat.frame.values,dat.set_zero.values,color = "black",alpha =0.8,label ='Zero corrected')


            ax[1].plot(dat.frame.values,dat.red_int.values,color = "salmon",alpha =0.8,label ='Raw')
            ax[0].legend()
            ax[1].legend()
            ax[0] = fix_ax_probs(ax[0],'Frame','Photon')
            ax[1] = fix_ax_probs(ax[1],'Frame','Photon')

            ax[2].plot(dat.frame.values,dat.red_int_mean.values,color = "darkred",alpha =0.8,label ='Mean')
            ax[2].legend()
            ax[2] = fix_ax_probs(ax[2],'Frame','Photon')


            ax[3].imshow(loc_averaged_over_frames)
            ax[3].imshow(BG_mask , cmap=my_cmap2,interpolation='none',vmin = 0.1,alpha = 0.4)
            ax[3].imshow(mask , cmap=my_cmap2,interpolation='none',vmin = 0.1,alpha = 0.4)
            ax[3].set_xlim(x_aq-20,x_aq+20)
            ax[3].set_ylim(y_aq-20,y_aq+20)
            fig.tight_layout()
            fig.savefig(save_path / f"{counter}_" / f"{par}.png")
            plt.close('all')


        fig,ax = plt.subplots(figsize = (2.5,2.5))
        ax.hist(full.red_int_corrected.values,50,range= (-300,300),density = True, alpha = 0.7,ec = 'black',color = "firebrick",label = "Corrected")
        ax = fix_ax_probs(ax,'Photons','Density')
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path / f"{counter}_" / 'corrected_hist.pdf')

        fig,ax = plt.subplots(figsize = (2.5,2.5))
        ax.hist(full.red_int_gausscorrected.values, 50,range= (-300,300),density = True, alpha = 0.7,ec = 'black',color = "grey",label = "Gaussian corrected")
        ax = fix_ax_probs(ax,'Photons','Density')
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path / f"{counter}_" / "gausscorrected_hist.pdf")

        fig,ax = plt.subplots(figsize = (2.5,2.5))
        ax.hist(full.set_zero.values, 50,range= (-300,300), density = True, alpha = 0.7,ec = 'black',color = "grey",label = "Gaussian corrected")
        ax = fix_ax_probs(ax,'Photons','Density')
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path / f"{counter}_" / 'set_zero_hist.pdf')


        fig,ax = plt.subplots(figsize = (2.5,2.5))
        ax.hist(full.red_int.values,50,range= (-300,10000),density = True, alpha = 0.7,ec = 'black',color = "salmon",label = "Raw")
        ax = fix_ax_probs(ax,'Photons','Density')
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path / f"{counter}_" / 'raw_hist.pdf')

        plt.close('all')


        counter +=1


# ----------------------------- Run -------------------------------- #

if __name__ == "__main__":
    # drive_name = r"D:"
    folder = Path("sample_vids")

    # no_par_video = (fr'{drive_name}\Emily\background\Experiment_Process_001_20220823.tif')
    no_par_video = str(folder / "Experiment_Process_001_20220823.tif")

    # streptavidin, 20 frames videos
    # video_location = [fr"{drive_name}\Emily\_20220831\Tifs\c_20.tif"]
    video_location = [str(folder / "c_20.tif")]


    # protease signal, 2000 frames videos
    # video_signal = [rf"{drive_name}\Emily\_20220831\Tifs\s_20.tif"]
    video_signal = [str(folder / "s_20.tif")]


    save_path = str(folder / "output_new")

    extract_traces_average(video_location, video_signal, no_par_video, save_path, params = params) #check mean multiplier, size threshold and number of particles

    ## here for the casein







    # loc = image_loader_video('/Volumes/INTENSO/PUK/Data/19OCT21/tiffs/Experiment_Process_001_20211019.tif')
    # video_still= np.mean(loc,axis = 0) #mean of all frames in video, we detect on an average still image of all videos

    #   #mean_multiplier = 0.5
    # mean=np.mean(video_still) #mean of pixel value


    #   # forloop til gøre det per frame
    # f = tp.locate(video_still, diameter, minmass=mean * 0.6, separation=sep) #11 is diameter of roi used for position determination, odd number


    # fig,ax = plt.subplots()
    # ax.imshow(video_still)
    # ax.plot(f.x+5,f.y,'r.',alpha =0.5)


    # Averager hele videoen - jeg vil have det per frame !
    # Threshold skal ændres


    ###plot no par movie fit

    """

    def fit_random_pixel(movie, plot=True):
        counts, edges = np.histogram(
            movie[
                :,
                np.random.randint(len(movie[0])),
                np.random.randint(len(movie[0])),
            ],
            bins=100,
            density=False,
        )
        bw = edges[1] - edges[0]
        centers = edges[:-1] + bw / 2

        def chi2_gain(p):
            yfit = np.sum(counts) * bw * np.array([func(n, *p) for n in centers])
            return np.sum((counts[counts > 3] - yfit[counts > 3]) ** 2 / yfit[counts > 3])

        res = minimize(
            chi2_gain,
            [5, 1 / 53, 2000],
            bounds=[[0, None], [1e-9, None], [0, None]],
        )

        Gs = int(1 / res.x[1])
        Off = int(res.x[2])


        if plot:
            fig,ax = plt.subplots(figsize = (2.5,2.5))
            ax.bar(centers, counts, bw, color = "grey", alpha = 0.7)
            ax.plot(
                centers,
                np.sum(counts) * bw * np.array([func(n, *res.x) for n in centers]),
                color="red", linestyle = "--", label = str("Gain:" + str(Gs) + "\nOffset: " + str(Off))
            )
            ax = fix_ax_probs(ax,'Pixel value','Count')
            fig.tight_layout()
            fig.savefig(str(save_path+'EMCCD_calibration.pdf'))
            plt.close('all')

        return (res, stats.chi2.sf(res.fun, np.sum(counts > 3) - 3))



    no_par_video = '/Users/sarableshoy/Documents/Nano/PUK/Data/20OCT21/pH 11/Experiment_Process_002_20211020.tif'
    save_path = '/Volumes/INTENSO/PUK/Data/19OCT21/Treated/'
    np_particle_movie = image_loader_video(no_par_video)
    pixfits = [fit_random_pixel(np_particle_movie, plot = True) for i in tqdm(range(2))] #400 pixels to fit

    $

    Gs, offs, phots, pvals = (
    np.array([1 / res.x[1] for res, p in pixfits]),
    np.array([res.x[2] for res, p in pixfits]),
    np.array([res.x[0] for res, p in pixfits]),
    np.array([p for res, p in pixfits]),)

    """
