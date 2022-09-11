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

import pickle
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uuid
from glob import glob
from tqdm import tqdm
from skimage import io
from tifffile import imsave
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.special import ive
import scipy.stats as stats
import multiprocessing as mp
import trackpy as tp
import pandas as pd
from skimage import feature
import scipy.ndimage as ndimage
from pathlib import Path
import os
from functools import partial


#%%


lip_int_size = 9  #originally 15 for first attemps

##Not used BG + gap
lip_BG_size = 80   # originally 40 for first attemps
gap_size = 2 # adding gap

#15/6-21 changed from 6 to 10
sep = 8   # afstand mellem centrum af to partikler, 7 er meget lidt så skal være højere ved lavere densitet
mean_multiplier = 4  #hvor mange partikler finder den, around 1-3, lavere giver flere partikler
diameter = 11 #diameter used in tp.locate, odd integer


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

def image_loader_video(video):

    images_1 = io.imread(video)
    return np.asarray(images_1)

def signal_extractor_no_pos(video, final_df, red_blue,roi_size,bg_size,gap_size):  # change so taht red initial is after appearance timing

    global lip_int_size
    global lip_BG_size
    lip_int_size= roi_size
    lip_BG_size = bg_size

    def get_bck_pixels(image,cutoff = 0.985,plot=False):
        #function added by Henrik which gives mask of background pixels=True
        import scipy.stats as stats
        xmin,xmax = np.percentile(image,0.1),np.percentile(image,99.9)

        bins = 200
        bw = (xmax-xmin)/bins
        mean,sig = np.mean(image),np.std(image)
        alpha_guess,beta_guess = (mean/sig)**2,sig**2/mean
        pval_mov = 1-bw*stats.gamma.pdf(image,alpha_guess,scale=beta_guess)
        background_pix =  pval_mov<cutoff
        if plot:
            fig,ax = plt.subplots(1,2,figsize=(11,5))
            c,edges = np.histogram(image.flatten(),range=(xmin,xmax),bins=bins)
            centers = edges[:-1]+bw/2
            x,y,sy = centers[c>0],c[c>0],np.sqrt(c[c>0])

            mean,sig = np.mean(image),np.std(image)
            alpha_guess,beta_guess = (mean/sig)**2,sig**2/mean

            ax[0].plot(x,np.sum(y)*bw*stats.gamma.pdf(x,alpha_guess,scale=beta_guess),color="darkred")
            ax[0].errorbar(x,y,sy,color="dimgrey",fmt=".",capsize=2)

            pval_mov = 1-bw*stats.gamma.pdf(image,alpha_guess,scale=beta_guess)
            background_pix =  pval_mov<cutoff
            im = ax[1].imshow(image,cmap="Greys")
            fig.colorbar(im,ax=ax[1])
            ax[1].imshow(background_pix,alpha=0.2,cmap="Reds")
            ax[0].axvline(x=stats.gamma.ppf(cutoff,alpha_guess,scale=beta_guess),color="k",linestyle="--")
        return background_pix
    # avoid calculating bck_pixels in each line of data frame, construct a 3d array first
#
#

#    for frame in tqdm(video):                       #SSRB added 16-03-2021
#        bck_video.append(get_bck_pixels(frame))
#        #SSRB added 16-03-2021
#
#    bck_video = np.asarray(bck_video)#SSRB added 16-03-2021
#
    def cmask(index, array, BG_size, int_size,gap_size):
        """
        Only used for size and hence bck doesnt matter
        """
        a, b = index
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size + gap_size   # to make a "gab" between BG and roi

        BG_mask = (x * x + y * y <= lip_BG_size)


        return (np.sum((array[mask]))), np.median(((array[BG_mask])))

    final_df = final_df.sort_values(['frame'], ascending=True)

    def df_extractor2(row):
        b, a = row['x'], row['y'] #b,a
        frame = int(row['frame'])
        array = video[frame]

        #bck_array = bck_video[frame] #SSRB added 16-03-2021

        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size+gap_size  # to make a "gab" between BG and roi
        BG_mask = (x * x + y * y <= lip_BG_size)
        """Henrik added multiplication with get_bck_pixels"""
        BG_mask = np.bitwise_xor(BG_mask, mask2)#*bck_array #SSRB added 16-03-2021


#quantile = 0.5 = median for BG, try to change

        return np.sum((array[mask])), np.quantile(((array[BG_mask])),0.50),np.mean((array[mask])),np.mean((array[BG_mask])) # added np in sum


    size_maker = np.ones(video[0].shape)
    ind = 25, 25  # dont ask - leave it here, it just makes sure the below runs
    mask_size, BG_size = cmask(ind, size_maker, lip_BG_size, lip_int_size, gap_size)
    mask_size = np.sum(mask_size)

    a = final_df.apply(df_extractor2, axis=1)
    # a = df_extractor2(final_df, video)

    intensity = []
    bg = []
    mean = []
    b_mean = []
    for line in a:
        i, b, m, bm = line
        bg.append(b)
        intensity.append(i)
        mean.append(m)
        b_mean.append(bm)
    if red_blue == 'blue' or red_blue == 'Blue':
        final_df['blue_int'] = intensity
        final_df['blue_int_mean'] = mean
        final_df['blue_bg'] = bg
        final_df['blue_bg_mean'] = b_mean
        final_df['blue_int_corrected'] = (final_df['blue_int']) - (final_df['blue_bg']*mask_size)
        final_df['blue_int_gausscorrected'] = (final_df['blue_int']) - (final_df['background']*mask_size)

    elif red_blue == 'red' or red_blue == 'Red':
        final_df['red_int'] = intensity
        final_df['red_int_mean'] = mean

        final_df['red_bg'] = bg
        final_df['red_bg_mean'] = b_mean
        final_df['red_int_corrected'] = (final_df['red_int']) - (final_df['red_bg']*mask_size)
        final_df['red_int_gausscorrected'] = (final_df['red_int']) - (final_df['background']*mask_size)

    else:
        final_df['green_int'] = intensity
        final_df['green_int_mean'] = mean
        final_df['green_bg'] = bg
        final_df['green_bg_mean'] = b_mean
        final_df['green_int_corrected'] = (final_df['green_int']) - (final_df['green_bg']*mask_size)
        final_df['green_int_gausscorrected'] = (final_df['green_int']) - (final_df['background']*mask_size)


    return final_df


def cmask_plotter(index, array, BG_size, int_size,gap_size):
    a, b = index
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
    mask2 = x * x + y * y <= lip_int_size+gap_size   # to make a "gab" between BG and roi

    BG_mask = (x * x + y * y <= lip_BG_size)
    BG_mask = np.bitwise_xor(BG_mask, mask2)
    return mask,BG_mask



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





def func(s, E_l, gamma, offset):
    if s - offset == 0:
        return np.exp(-E_l)
    elif s - offset < 0:
        return 0
    else:
        logbessel = np.log(
            ive(1, 2 * np.sqrt(gamma * (s - offset) * E_l))
        ) + 2 * np.sqrt(gamma * (s - offset) * E_l)
        return np.exp(
            0.5 * np.log(gamma * E_l / (s - offset))
            + (-E_l - gamma * (s - offset))
            + logbessel
        )


def fit_random_pixel(movie, plot=False):
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
    if plot:
        plt.bar(centers, counts, bw)

        plt.plot(
            centers,
            np.sum(counts) * bw * np.array([func(n, *res.x) for n in centers]),
            color="C1",
        )
    return (res, stats.chi2.sf(res.fun, np.sum(counts > 3) - 3))


def Correct_illumination_profile(photon_movie, counter, save_path):
    save_path = Path(save_path)
    from skimage.filters import gaussian
    import torch
    meanmov = np.mean(photon_movie, axis=0)
    gauss_meanmov = gaussian(meanmov, 30)
    cuda_gauss = torch.tensor(gauss_meanmov).to("cuda")
    cuda_gauss = cuda_gauss / cuda_gauss.max()
    # gauss_meanmov = gauss_meanmov / gauss_meanmov.max()

    # scaled_movie = photon_movie / gauss_meanmov
    scaled_movie = torch.tensor(photon_movie).to("cuda") / cuda_gauss
    scaled_movie.numpy()
    plot=False

    if plot:
        ind = np.random.randint(len(photon_movie))
        fig, ax = plt.subplots(1, 3, figsize=(9, 4.5))
        ax[0].imshow(
            photon_movie[ind],
            vmin=np.percentile(photon_movie[ind], 20),
            vmax=np.percentile(photon_movie[ind], 99),
            cmap="Greys_r",
        )
        ax[1].imshow(gauss_meanmov)
        ax[2].imshow(
            scaled_movie[ind],
            vmin=np.percentile(photon_movie[ind], 20),
            vmax=np.percentile(photon_movie[ind], 99),
            cmap="Greys_r",
        )
        for a in ax:
            a.set(xlabel="Column", ylabel="Row")
        ax[0].set(title="Original movie")
        ax[1].set(title="Estimated illumination profile")
        ax[2].set(title="Corrected movie")
        fig.tight_layout()
        fig.savefig(save_path / f"{counter}_" / "illumination.pdf")
    return scaled_movie


# ----------------------------- Main Function ------------------------------ #

#Function that extract traces
#extract_traces_average(video_location,video_signal, no_par_video, save_path)

def extract_traces_average(location, video, no_par_video, save_path):
    save_path = Path(save_path)
    from tifffile import imsave
    #mean_multiplier = 0.5
    counter = 0


    ### no particle video
    pix_fits = False
    with mp.Pool(8) as pool:
        if pix_fits:
            np_particle_movie = image_loader_video(no_par_video)
            n_random_pix = 400
            # pixfits = list(tqdm(pool.imap(partial(fit_random_pixel, np_particle_movie), range(400)), total = 400))
            pixfits = [fit_random_pixel(np_particle_movie) for i in tqdm(range(400))] #400 pixels to fit


            Gs, offs, phots, pvals = (
            np.array([1 / res.x[1] for res, p in pixfits]),
            np.array([res.x[2] for res, p in pixfits]),
            np.array([res.x[0] for res, p in pixfits]),
            np.array([p for res, p in pixfits]),
            )
            with open("random_pix_fits.pkl", "wb") as file:
                pickle.dump((Gs, offs, phots, pvals), file)
        else:
            with open("random_pix_fits.pkl", "rb") as file:
                Gs, offs, phots, pvals = pickle.load(file)
    ####



    for video,loc in tqdm(zip(video,location)):
        start = time()

        if not os.path.exists(save_path / f"{counter}_"):                      # creates a folder for saving the stuff in if it does not exist
            os.makedirs(save_path / f"{counter}_")

        video = image_loader_video(video)
        loc = image_loader_video(loc)

        #Move to the other function:
        #imsave(str(main_save_path+'norm.tif'),video[:1000]) #saves normal video

        import scipy.ndimage as ndimage

        # should not be needed for Sara
        #video = ndimage.gaussian_filter(video,sigma = (3,0,0),order = 0) #smooth 3 sigma in z, try to change this
        print(f"1 {time()-start}")
        photon_movie = (video - np.mean(offs[pvals > 0.01])) / np.mean(Gs[pvals > 0.01])
        print(f"2 {time()-start}")
        corrected_mov = Correct_illumination_profile(photon_movie, counter, save_path)
        print(f"3 {time()-start}")


        #imsave(str(main_save_path+'smoothed.tif'),video[:1000]) #saves smoothed video

        ##end move to other function
        ####only detect on first 500 frames. added 16/7-21

        video_still= np.mean(loc,axis = 0) #mean of all frames in video, we detect on an average still image of all videos

        #mean_multiplier = 0.5
        mean=np.mean(video_still) #mean of pixel value


        # forloop til gøre det per frame
        f = tp.locate(video_still, diameter, minmass=mean * mean_multiplier, separation=sep) #11 is diameter of roi used for position determination, odd number

#        plt.imshow(video_still)
#        tp.annotate(f, video_still)

        fig,ax = plt.subplots(figsize = (10,10))
        ax.imshow(video_still)
        ax.plot(f.x.values,f.y.values,'.',color = "red",alpha =0.5)
        fig.tight_layout()
        fig.savefig(save_path / f"{counter}_" / "full_locate_loc.pdf")
        fig.savefig(save_path / f"{counter}_" / "full_locate_loc.png")

        plt.close('all')
#

        # added SSRB 23-04-21 start
        #f = f[f.ecc<0.25] # accept only round-ish particles to avoid doubles
        f = tp.refine_leastsq(f,video_still,5) # refine coordinates of close particles, adds the possibility  of a local mean noise correction termend "background"


        fig,ax = plt.subplots(figsize = (10,10))
        ax.imshow(video_still)
        ax.plot(f.x.values,f.y.values,'.',color = "red",alpha =0.5)
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
        new =  pd.DataFrame()
        for i in range(len(corrected_mov)):
            f['frame'] = i
            new= new.append(f, ignore_index = True)

        new = signal_extractor_no_pos(corrected_mov, new, 'red',lip_int_size,lip_BG_size,gap_size)
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
            text_file.write("ROI: " +str(lip_int_size) +"\n")
            text_file.write("BG ROI: " +str(lip_BG_size) +"\n")
            text_file.write("GAP size: " +str(gap_size) +"\n")
            text_file.write("sep: " +str(sep) +"\n")
            text_file.write("meanmultiplier: " +str(mean_multiplier) +"\n")
            text_file.write("diameter: " +str(diameter) +"\n")


#new end

        grp = full.groupby('particle')
        import matplotlib
        my_cmap = matplotlib.cm.Reds
        my_cmap.set_under('k', alpha=0)
        my_cmap.set_bad('k', alpha=0)

        my_cmap2 = matplotlib.cm.Greens
        my_cmap2.set_under('k', alpha=0)
        my_cmap2.set_bad('k', alpha=0)
        size_maker = np.ones(video_still.shape)

        for par,dat in grp:
            x_aq = dat.x.values[1]

            y_aq = dat.y.values[1]
            ind = (y_aq,x_aq)  # dont ask - leave it here, it just makes sure the below runs


            mask, BG_mask = cmask_plotter(ind, size_maker, lip_BG_size, lip_int_size,gap_size)

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


            ax[3].imshow(video_still)
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


    save_path = str(folder / "output")

    extract_traces_average(video_location, video_signal, no_par_video, save_path) #check mean multiplier, size threshold and number of particles

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

# %%
