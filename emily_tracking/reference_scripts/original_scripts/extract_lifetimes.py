import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import uuid
from glob import glob
from tqdm import tqdm
from skimage import io
from tifffile import imsave
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import random, tifffile, os, copy
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.special import ive
import scipy.stats as stats
import multiprocessing as mp
from functools import partial
import trackpy as tp
import pandas as pd
import scipy
from skimage import feature
import scipy.ndimage as ndimage



# ------------------- Defining functions ------------------- #

lip_int_size = 9  #originally 15 for first attemps

##Not used BG + gap
lip_BG_size = 40   # originally 40 for first attemps
gap_size = 2 # adding gap

#15/6-21 changed from 6 to 10
sep = 8   # afstand mellem centrum af to partikler, 7 er meget lidt så skal være højere ved lavere densitet
mean_multiplier = 2  #hvor mange partikler finder den, around 1-3, lavere giver flere partikler
diameter = 11 #diameter used in tp.locate, odd integer


# Sara: sets axis
def fix_ax_probs(ax,x_label,y_label):
    ax.set_ylabel(y_label,size = 12)
    ax.set_xlabel(x_label,size = 12)
    ax.tick_params(axis= "both", which = "major", labelsize = 10)
    ax.tick_params(axis= "both", which = "minor", labelsize = 10)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = "upper right",frameon = False)
    return ax

#first reverse green videos

def image_loader_video(video):

    images_1 = io.imread(video)
    return np.asarray(images_1)

# Sara: signal extractor no/number position?
def signal_extractor_no_pos(video, final_df, red_blue, roi_size, bg_size, gap_size):  # changed so that red initial is after appearance timing

    global lip_int_size
    global lip_BG_size
    lip_int_size= roi_size
    lip_BG_size = bg_size

    # Sara: Seperate particles and background? Determine particle size (radius)
    def cmask(index, array, BG_size, int_size, gap_size):
        """
        Only used for size and hence bck doesnt matter
        """
        a, b = index
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size + gap_size   # to make a "gab" between BG and roi. Seperate background from ROI.

        BG_mask = (x * x + y * y <= lip_BG_size)


        return (np.sum((array[mask]))), np.median(((array[BG_mask])))

    # Sara: df = density function? Or differential? Think it is differential?
    final_df = final_df.sort_values(['frame'], ascending=True)

    # Sara: Why calculate size again?
    def df_extractor2(row):
        b, a = row['x'], row['y'] # b, a
        frame = int(row['frame'])
        array = video[frame]

        #bck_array = bck_video[frame] #SSRB added 16-03-2021

        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size + gap_size  # to make a "gab" between BG and roi
        BG_mask = (x * x + y * y <= lip_BG_size)
        """Henrik added multiplication with get_bck_pixels"""
        BG_mask = np.bitwise_xor(BG_mask, mask2)#*bck_array #SSRB added 16-03-2021


#quantile = 0.5 = median for BG, try to change

        return np.sum((array[mask])), np.quantile(((array[BG_mask])),0.50),np.mean((array[mask])),np.mean((array[BG_mask])) # added np in sum


    size_maker = np.ones(video[0].shape)
    ind = 25, 25  # dont ask - leave it here, it just makes sure the below runs
    mask_size, BG_size = cmask(ind, size_maker, lip_BG_size, lip_int_size, gap_size)
    mask_size = np.sum(mask_size)


    # Sara: a gives sum of mask (r^2, array), quantile + median of BG mask (array), mean of mask + BG mask. Ascending values.
    a = final_df.apply(df_extractor2, axis=1)


    intensity = []
    bg = []
    mean = []
    b_mean = []

    # sara: Sorting channels?
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

    elif red_blue == 'red' or red_blue == 'Red':
        final_df['red_int'] = intensity
        final_df['red_int_mean'] = mean
        final_df['red_bg'] = bg
        final_df['red_bg_mean'] = b_mean
        final_df['red_int_corrected'] = (final_df['red_int']) - (final_df['red_bg']*mask_size)

    else:
        final_df['green_int'] = intensity
        final_df['green_int_mean'] = mean
        final_df['green_bg'] = bg
        final_df['green_bg_mean'] = b_mean
        final_df['green_int_corrected'] = (final_df['green_int']) - (final_df['green_bg']*mask_size)

    print ("i worked ")
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


# Sara: sætte diffential funktonen lig 0?
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

    from skimage.filters import gaussian

    meanmov = np.mean(photon_movie, axis=0)
    gauss_meanmov = gaussian(meanmov, 30)
    gauss_meanmov = gauss_meanmov / np.max(gauss_meanmov)

    scaled_movie = photon_movie / gauss_meanmov

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
        fig.savefig(save_path+str(counter)+'_/'+"illumination"+'.pdf')
    return np.asarray(scaled_movie,dtype = 'float32')


# ------------------------------ Main Function -------------------------------- #

def extract_traces_average(videos, save_path):
    from tifffile import imsave
    #mean_multiplier = 0.5
    counter = 0


    ### no particle video

    # np_particle_movie = image_loader_video(no_par_video)

    # pixfits = [fit_random_pixel(np_particle_movie) for i in tqdm(range(400))] #400 pixels to fit


    # Gs, offs, phots, pvals = (
    # np.array([1 / res.x[1] for res, p in pixfits]),
    # np.array([res.x[2] for res, p in pixfits]),
    # np.array([res.x[0] for res, p in pixfits]),
    # np.array([p for res, p in pixfits]),)


    for video in tqdm(videos):

        if not os.path.exists(str(save_path+str(counter)+'_/')):              # creates a folder for saving the stuff in if it does not exist
            os.makedirs(str(save_path+str(counter)+'_/'))

        video = image_loader_video(video)

        import scipy.ndimage as ndimage



        # photon_movie = (video - np.mean(offs[pvals > 0.01])) / np.mean(Gs[pvals > 0.01])
        corrected_mov = video#Correct_illumination_profile(photon_movie, counter, save_path)


        imsave(str(save_path+str(counter)+'photon_ILM_corr.tif'), corrected_mov) #saves smoothed video


        # minmass: intensity, can be changed based on the data
        # 11 area of particle - where we fit the gaussian.
        # 5 (antal pixels it moves between frames, max 5, min 2), memory (may the particle reappear - NO!)

        f=tp.batch(corrected_mov, 11, minmass = 2500, separation = 7, processes = 1)
        new = tp.link_df(f, 4, memory=0)
        del(f)
        new["duration"] = new.groupby("particle")["particle"].transform(len)


        # new = signal_extractor_no_pos(corrected_mov, new, 'red',lip_int_size,lip_BG_size,gap_size)   # intensitet
        new = signal_extractor_no_pos(corrected_mov, new, 'red',lip_int_size,lip_BG_size,gap_size)
        new['video_counter'] = counter


        a = new['particle'].tolist()
        z = uuid.uuid4().hex
        a = [str(str(x)+'_'+str(z)) for x in a]
        new['particle'] = a
        new.to_csv(str(save_path+str(counter)+'_/'+'_signal_df.csv'), header=True, index=None, sep=',', mode='w')
        counter +=1

# -------------------------------- Run code -------------------------------- #
#Savinase
videos = [r"D:\Emily\\_20220831\Tifs\s_30.tif"]
#          r"D:\Emily\\_20220831\Tifs\s_20.tif",
#         r"D:\Emily\\_20220831\Tifs\s_80.tif"]

#['/Users/sarableshoy/Documents/Nano/PUK/Data/15SEP21/tifs/Experiment_Process_001_20210915.tif','/Users/sarableshoy/Documents/Nano/PUK/Data/15SEP21/tifs/Experiment_Process_002_20210915.tif',
#'/Users/sarableshoy/Documents/Nano/PUK/Data/15SEP21/tifs/Experiment_Process_003_20210915.tif', '/Users/sarableshoy/Documents/Nano/PUK/Data/15SEP21/tifs/Experiment_Process_004_20210915.tif',
#'/Users/sarableshoy/Documents/Nano/PUK/Data/15SEP21/tifs/Experiment_Process_005_20210915.tif', '/Users/sarableshoy/Documents/Nano/PUK/Data/15SEP21/tifs/Experiment_Process_006_20210915.tif',
#'/Users/sarableshoy/Documents/Nano/PUK/Data/15SEP21/tifs/Experiment_Process_007_20210915.tif', '/Users/sarableshoy/Documents/Nano/PUK/Data/15SEP21/tifs/Experiment_Process_008_20210915.tif',
#'/Users/sarableshoy/Documents/Nano/PUK/Data/15SEP21/tifs/Experiment_Process_009_20210915.tif', '/Users/sarableshoy/Documents/Nano/PUK/Data/15SEP21/tifs/Experiment_Process_010_20210915.tif']


# Sara: Negative control to substract background.
# test = image_loader_video( '/Volumes/Min Zhang/protease/5nM casein_20220210/t2.tif')

# f = tp.locate(test[100],11,minmass =2500,separation =4)


# fig,ax = plt.subplots()
# ax.imshow(test[100])
# ax.plot(f.x,f.y,'r.')

extract_traces_average(videos, 'D:\Emily\_20220831\Tifs\lifetime\lifetime_test')

# ----------------------------- Use to optimize ---------------------------- #

# Check amount of particles found, by optimizing. (Compare w. Fiji trackMate)
#corr_movie = '/Users/sarableshoy/Documents/Nano/PUK/Data/15SEP21/Treated/0photon_ILM_corr.tif'
#corrected_mov = image_loader_video(corr_movie)
#tp.locate(corrected_mov[1000], 11, minmass = 265, separation = 7)


# -------------------------------- Plotting -------------------------------- #

# def extractdurations(listOfDfs):
#
#     durations = []
#     for path in listOfDfs:
#         tmp = pd.read_csv(path)
#         tmp = tmp.drop_duplicates(subset=['particle'],keep ='last')
#         durations.extend(tmp.duration)
#     return durations
#
# listOfDfs = ['D:\Emily\_20220831\Tifs\lifetime0_\_signal_df.csv']
# #             '/Volumes/Min Zhang/protease/5nM casein/lifetime1/0_/_signal_df.csv',
# #             '/Volumes/Min Zhang/protease/5nM casein/lifetime1/1_/_signal_df.csv',
# #             '/Volumes/Min Zhang/protease/5nM casein/lifetime1/2_/_signal_df.csv']# paths to all datafranes that should be plottet together
#
# #listOfDfs2 = ['/Volumes/Min Zhang/protease/ph9&10/lifetime/0_/_signal_df.csv']
#  #            '/Volumes/Min Zhang/protease/5nM casein/lifetime/11_/_signal_df.csv',
#  #            '/Volumes/Min Zhang/protease/5nM casein/lifetime/12_/_signal_df.csv']
# #listOfDfs3 = ['/Volumes/Min Zhang/protease/ph9&10/lifetime/4_/_signal_df.csv']
# allDurations = extractdurations(listOfDfs)
# print(np.max(allDurations))
# #allDurations2 = extractdurations(listOfDfs2)
# #allDurations3 = extractdurations(listOfDfs3)
#
#
# main = (r"D:\Emily\_20220831\Tifs\lifetime2")
#
# fig,ax = plt.subplots(figsize = (3,3) )
# #darkslategray
# ax.hist(allDurations,bins = 35,density = True, ec = 'black',color = 'darkslategray',range = (0,100),alpha = 0.8,label ='savinase pH 7.8' )
# #ax.hist(allDurations2,bins = 35,density = True, ec = 'red', color = 'grey',range = (0.5,35.5),alpha = 0.8,label ='savinase pH 9' ) # if you need two in the same figure
# #ax.hist(allDurations3,bins = 35,density = True, ec = 'green', color = 'lightgrey',range = (0.5,35.5),alpha = 0.8,label ='savinase pH 10' )
#
# ax = fix_ax_probs(ax, 'duration [frames]', 'Density')
# #plt.ylim((0,1))
# ax.set_yscale('log')
# fig.tight_layout()
# fig.savefig(str(main+'phb15log_t1_.pdf'))
