import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.ndimage as ndimage
import multiprocessing as mp
from contextlib import nullcontext
from functools import partial
import yaml
import os
import re
from skimage import io
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import ive
import scipy.stats as stats
import uuid
import tifffile
from tqdm import tqdm
import pickle
import hashlib

# Tracking parameters

class Params:
    defaults = {
        # 'mean_multiplier': 0.8,
        # 'sep': 5,
        # 'object_size': 9,
        # 'lip_int_size': 9,
        # 'lip_BG_size': 60,
        # 'memory': 1,
        # 'search_range': 5,
        # 'duration_filter': 20,
        # 'tracking_time': 0.036,
        # 'pixel_size': 0.18333,
        # 'n_processes': 8,
        # 'save_path': None,
        # 'exp_type_folders': None,
        # 'exp_types': None,
        # 'gap_size': 0,
        }
    values = defaults.copy()

    def __init__(self, parampath = None, **kwargs):
        super().__setattr__("_path", parampath)
        self.__dict__.update(self.defaults)
        if parampath is not None:
            with open(parampath, 'r') as f:
                yaml_contents = yaml.safe_load(f)
                self.values.update(yaml_contents)
        self.values.update(kwargs)
        self.__dict__.update(self.values)
    
    def __getitem__(self, key):
        return self.values[key]
    
    def __setitem__(self, key, value):
        self.values[key] = value
        self.__dict__.update(self.values)

    def __setattr__(self, key, value):
        self.values[key] = value
        self.__dict__.update(self.values)
    
    def to_yaml(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.values, f)

    def __str__(self):
        return str(self.values)
    
    def __repr__(self):
        return str(self.values)
    
    def update_mm_and_sep(self, path):
        with open(path, 'r+') as f:
            contents = f.read()
            contents = re.sub(r'mean_multiplier: \d+\.?\d*', 'mean_multiplier: ' + str(self.mean_multiplier), contents)
            contents = re.sub(r'sep: \d+\.?\d*', 'sep: ' + str(self.sep), contents)
            f.seek(0)
            f.write(contents)

def isnotebook():
    try:
        result = get_ipython()
    except NameError as e:
        result = None
    return result

def calibration(params, video_location = None, video = None, save = True, save_location = None):
    if isinstance(params, str):
        params = Params(params)
    ipython = isnotebook()
    if ipython is not None:
        ipython.run_line_magic("matplotlib", "")
    
    if video_location is not None:
        video = image_loader_video(video_location)

    run = True
    mean_multiplier = params.mean_multiplier
    sep = params.sep
    object_size = params.object_size
    plt.ion()
    fig, ax = plt.subplots()
    while run:
        mean = np.mean(video[0])

        first_frame = video[0]
        first_frame = ndimage.gaussian_filter(first_frame, sigma = (0.5, 0.5), order =0)

        
        f = tp.locate(first_frame, params.object_size, invert=False, minmass = mean*mean_multiplier, separation = sep)
        tp.annotate(f, first_frame, ax = ax)
        plt.show()
        plt.pause(0.05)
        print(f"Found: {len(f)} particles")
        print(f"Current mean multiplier: {mean_multiplier}\nCurrent separation: {sep}\nCurrent object_size: {object_size}")
        raw = input("Enter new mean multiplier, or 'q' to quit and save: ")
        if raw == "q":
            run = False
        else:
            mean_multiplier = float(raw)
            sep = input("Enter separation in pixels: ")
            sep = int(sep)
            object_size = input("Enter object size in pixels (has to be an odd integer): ")
            object_size = int(object_size)
            ax.clear()
    params.mean_multiplier = mean_multiplier
    params.sep = sep
    params.object_size = object_size
    if save:
        if save_location:
            params.update_mm_and_sep(save_location)
        else:
            params.update_mm_and_sep(params._path)
    
    return params


def image_loader_video(video):
    im = io.imread(video)
    return np.asarray(im)

def mp_msd(groupbby_tup, **msd_kwargs):
    return groupbby_tup[0], tp.msd(groupbby_tup[1], **msd_kwargs)



def step_tracker(df, params):
    microns_per_pixel = params.pixel_size
    steps = []
    
    df['x'] = df['x'] * microns_per_pixel 
    df['y'] = df['y'] * microns_per_pixel 
    
    group_all = df.groupby('particle')
    x_step = []
    y_step = []
    
    # easiest: compute step in x, step in y and then steps
    for name, group in group_all:
        x_list = group.x.tolist()
        x_tmp = [y - x for x,y in zip(x_list,x_list[1:])] 
        x_tmp.insert(0, 0.)
        
        y_list = group.y.tolist()
        y_tmp = [y - x for x,y in zip(y_list,y_list[1:])]
        y_tmp.insert(0, 0.)
        y_step.extend(y_tmp)
        x_step.extend(x_tmp)
        step_tmp = [np.sqrt(y**2+x**2) for y,x in zip(y_tmp,x_tmp)]

        steps.extend(step_tmp)  
        
    df['x_step'] = x_step  
    df['y_step'] = y_step 
    df['steplength'] = steps 

    return df


def fast_signal_extraction(positions, frames, r, padded_video, params, return_means = False, backend = None, device = None):
    if backend is None:
        try:
            import torch
            
            backend = torch
        except ModuleNotFoundError:
            try:
                import bottleneck
                backend = bottleneck
            except ModuleNotFoundError:
                backend = np
        
    if backend.__name__ == "torch":
        using_torch = True
        ap = backend
        if device is None:
            device = "cuda" if ap.cuda.is_available() else "cpu"
        if not isinstance(padded_video, ap.Tensor):
            padded_video = ap.tensor(padded_video).to(device)
    else:
        using_torch = False
        ap = np
    
    sub_pix = positions % 1
    pix = positions.astype(int)

    r_ceil = np.ceil(r).astype(int)
    difference_for_dimension = lambda dim: ap.moveaxis(ap.arange(-r_ceil + 1, r_ceil + 1) - sub_pix[:, dim].reshape(-1, 1, 1), -1, dim + 1)
    xs = difference_for_dimension(0)
    ys = difference_for_dimension(1)
    if using_torch:
        xs = xs.to(device)
        ys = ys.to(device)
    r2 = xs**2 + ys**2
    inds_for_dimension = lambda dim: ap.moveaxis((
        ap.arange(-r_ceil + 1, r_ceil + 1)) + pix[:, dim].reshape(-1, 1, 1), -1, dim+1).clip(-1, padded_video.shape[dim + 1] - 1)
    xinds = inds_for_dimension(0)
    yinds = inds_for_dimension(1)

    rectangles = padded_video[frames[:, None, None], xinds, yinds]

    sigmasks = r2 <= params.lip_int_size
    if params.gap_size != 0:
        bg_inner = r2 <= params.lip_int_size + params.gap_size
    else:
        bg_inner = sigmasks

    bgmasks = ap.logical_xor(r2 <= r**2, bg_inner)
    
    if using_torch:
        nan = ap.tensor(float("nan"), dtype = ap.float64).to(device)
    else:
        nan = ap.nan
    
    signals = ap.where(sigmasks, rectangles, nan).reshape(positions.shape[0], -1)
    backgrounds = ap.where(bgmasks, rectangles, nan).reshape(positions.shape[0], -1)

    sig_sums = backend.nansum(signals, axis = 1)
    bg_medians = backend.nanmedian(backgrounds, axis = 1)
    sig_sizes = backend.nansum(sigmasks.reshape(positions.shape[0], -1), axis = 1)
    out = [sig_sums, bg_medians, sig_sizes]
    if return_means:
        sig_means = sig_sums / sig_sizes
        if using_torch:
            bg_sums = backend.nansum(backgrounds, axis = 1)
            bg_sizes = backend.nansum(bgmasks.reshape(positions.shape[0], -1), axis = 1)
            bg_means = bg_sums / bg_sizes
        else:
            bg_means = backend.nansum(backgrounds, axis = 1)
        out.extend([sig_means, bg_means])
    return out


def getinds(smallmask, i_pos, j_pos, video_shape, r):
    i_lower = int(i_pos) - r + 1
    j_lower = int(j_pos) - r + 1
    i_upper = int(i_pos) + r + 1
    j_upper = int(j_pos) + r + 1
    out_of_bounds = i_lower < 0 or j_lower < 0 or\
                    i_upper > video_shape[1] or\
                    j_upper > video_shape[2]
    if out_of_bounds:
        i_inds = slice(max(i_lower, 0),
                        min(i_upper, video_shape[1]))
        j_inds = slice(max(j_lower, 0),
                        min(j_upper, video_shape[2]))
        smallmask_i_lower = max(-i_lower, 0)
        smallmask_j_lower = max(-j_lower, 0)
        smallmask_i_upper = min(video_shape[1]-i_upper, 0)
        smallmask_j_upper = min(video_shape[2]-j_upper, 0)
        smallmask_i_upper = None if smallmask_i_upper == 0 else smallmask_i_upper
        smallmask_j_upper = None if smallmask_j_upper == 0 else smallmask_j_upper
        smallmaskslice = slice(smallmask_i_lower, smallmask_i_upper), slice(smallmask_j_lower, smallmask_j_upper)
        # print(smallmaskslice)
        smallmask = smallmask[smallmaskslice]
    else:
        i_inds = slice(int(i_pos) - r + 1, int(i_pos) + r + 1)
        j_inds = slice(int(j_pos) - r + 1, int(j_pos) + r + 1)
    # i_inds, j_inds = j_inds, i_inds
    return i_inds, j_inds, smallmask

def make_smallmask_sig_bg(i_pos, j_pos, video_shape, params):
    sqrt_BG_size = int(np.ceil(np.sqrt(params.lip_BG_size)))
    
    all_is, all_js = np.ogrid[-sqrt_BG_size-(i_pos%1)+1:+sqrt_BG_size-(i_pos%1)+1,
                    -sqrt_BG_size-(j_pos%1)+1:+sqrt_BG_size-(j_pos%1)+1]
    all_r2s = all_is**2 + all_js**2
    sig = all_r2s <= params.lip_int_size
    if params.gap_size != 0:
        bg = np.logical_xor(all_r2s <= params.lip_BG_size, all_r2s <= params.lip_int_size + params.gap_size)
    else:
        bg = np.logical_xor(all_r2s <= params.lip_BG_size, sig)
    
    sig_inds = getinds(sig, i_pos, j_pos, video_shape, sqrt_BG_size)
    bg_inds = getinds(bg, i_pos, j_pos, video_shape, sqrt_BG_size)
    return sig_inds, bg_inds

def get_intensity_and_background(i_pos, j_pos, video, frame, params, return_means = False):
    sig_inds, bg_inds = make_smallmask_sig_bg(i_pos, j_pos, video.shape, params)

    frame = int(frame)
    signal = video[frame, sig_inds[0], sig_inds[1]][sig_inds[2]]
    background = video[frame, bg_inds[0], bg_inds[1]][bg_inds[2]]


    out = [np.sum(signal), np.median(background), sig_inds[2].sum()]
    if return_means:
        out.extend([np.mean(signal), np.mean(background)])
    return out

def get_intensity_and_background_constant_position(i_pos, j_pos, video, params, return_means = True):
    sig_inds, bg_inds = make_smallmask_sig_bg(i_pos, j_pos, video.shape, params)
    signal = video[:, sig_inds[0], sig_inds[1]][:, sig_inds[2]]
    background = video[:, bg_inds[0], bg_inds[1]][:, bg_inds[2]]

    out = [np.sum(signal, axis = 1), np.median(background, axis = 1), np.repeat(sig_inds[2].sum(), video.shape[0])]
    if return_means:
        out.extend([np.mean(signal, axis = 1), np.mean(background, axis = 1)])
    # out = np.vstack(out)
    return out

def share_video_with_subprocesses(raw_array, shape):
    global _video
    _video = np.asarray(raw_array).reshape(shape)


def wrap_int_bg(row, params, **kwargs):
    return get_intensity_and_background(row[1], row[0], _video, row[2], params, **kwargs)

def modify_track_df_with_roi_intensities(track_df, roi_intensities, red_blue, include_means, row_mask = slice(None)):
    roi_columns = ["int", "bg", "sig_pixel_size"]
    if include_means:
        roi_columns.extend(["int_mean", "bg_mean"])
    
    roi_intensities.columns = roi_columns

    roi_intensities["int_corrected"] = roi_intensities["int"] - roi_intensities["bg"] * roi_intensities["sig_pixel_size"]
    red_blue = red_blue.lower()
    for column_name in roi_intensities.columns:
        track_df.loc[row_mask, f"{red_blue}_{column_name}"] = roi_intensities[column_name].to_numpy()

    if "background" in track_df:
        track_df.loc[row_mask, f'{red_blue}_int_gausscorrected'] = ((track_df.loc[row_mask, f'{red_blue}_int']) - \
            track_df.loc[row_mask, 'background']*track_df.loc[row_mask, f"{red_blue}_sig_pixel_size"]).to_numpy()
    
    return track_df

def signal_extractor_no_pos(video, full_tracked, red_blue, params, pool = None, include_means = False):  # change so taht red initial is after appearance timing
    
    full_tracked = full_tracked.sort_values('frame', ignore_index = True, kind = "stable")
    
    if pool is None:
        roi_intensities = full_tracked.apply(lambda row: get_intensity_and_background(
                row['y'], row['x'], video, row['frame'], params, return_means = include_means), axis=1, result_type = "expand")
    else:
        roi_intensities = pool.map(partial(wrap_int_bg, params = params, return_means = include_means),
                full_tracked[['x', 'y', 'frame']].to_numpy())
        roi_intensities = pd.DataFrame(roi_intensities)

    full_tracked = modify_track_df_with_roi_intensities(full_tracked, roi_intensities, red_blue, include_means)
    
    return full_tracked

def create_pool_context(n_processes, video, initializer = share_video_with_subprocesses):
    if n_processes > 1:
        print("")
        raw_array_video = mp.RawArray(np.ctypeslib.as_ctypes_type(video.dtype), video.size)
        np_for_copying = np.asarray(raw_array_video).reshape(video.shape)
        np.copyto(np_for_copying, video)
        # print("Time to copy video: ", end-start)
        context_man = mp.get_context("spawn").Pool(n_processes, initializer = initializer,
                initargs = (raw_array_video, video.shape))
    else:
        context_man = nullcontext()
    return context_man

def mp_imsd(traj, mpp, fps, max_lagtime=100, statistic='msd', pos_columns=None, pool = None):
    if pool is None:
        ids = []
        msds = []
        for pid, ptraj in traj.groupby('particle'):
            msds.append(tp.msd(ptraj, mpp, fps, max_lagtime, False, pos_columns))
            ids.append(pid)
    else:
        results = pool.map(partial(mp_msd, mpp = mpp, fps = fps,
                        max_lagtime = max_lagtime, detail = False,
                        pos_columns = pos_columns), traj.groupby('particle'))
        
        ids, msds = zip(*results)
    results = tp.motion.pandas_concat(msds, keys=ids)
    results = results.swaplevel(0, 1)[statistic].unstack()
    lagt = results.index.values.astype('float64')/float(fps)
    results.set_index(lagt, inplace=True)
    results.index.name = 'lag time [s]'
    return results

def tracker(videoinp, replicate, params):
    mean = np.mean(videoinp[0])
    
    video = ndimage.gaussian_filter(videoinp, sigma=(0,0.5,0.5), order=0) #dont use

    
    # ----- USE THIS WHEN AVG SUBSTRACKTING -----
    #mean_vid = np.mean(video,axis = 0) #finds avg video, use only when tracking in same layer in shor period of time 
    #sub_vid = video - mean_vid     
    #sub_vid =ndimage.gaussian_filter(sub_vid,sigma =(0,0.5,0.5), order =0) # smooting 
    
    #video = sub_vid
    
    full = tp.batch(video, params.object_size, invert = False, minmass = mean*params.mean_multiplier,
                    separation = params.sep, processes = params.n_processes); # mac har problemer med multiprocessing, derfor processing = 1
                                                               # processes = 8 burde kunne køre på erda. (8 kerner)
    #check for subpixel accuracy
    tp.subpx_bias(full)
    plt.savefig(Path(params.save_path) / f'{replicate}subpix.png')
    # plt.show()
    
        
    context_man = create_pool_context(params.n_processes, video)

    with context_man as pool:
        full = signal_extractor_no_pos(video, full, 'red', params = params, pool = pool)
        full_tracked = tp.link_df(full, params.search_range, memory=params.memory) #5 pixel search range, memory =2 
        full_tracked = tp.filter_stubs(full_tracked, params.duration_filter) # filter aour short stuff
        
        
        full_tracked = step_tracker(full_tracked, params)
        full_tracked['particle'] = full_tracked['particle'].transform(int)
        
        full_tracked['duration'] = full_tracked.groupby('particle')['particle'].transform(len)
        
        msd_df = mp_imsd(full_tracked, params.pixel_size, float(params.tracking_time),
        max_lagtime = full_tracked["duration"].max(), pool = pool)

    msd_df=msd_df.stack().reset_index()
    msd_df.columns = ['time', 'particle','msd']
    return full_tracked, msd_df 

def runner_tracker(video_location, experiment_type, replicate, params): 
    
    video = image_loader_video(video_location)
    
    full_tracked, msd_df = tracker(video, replicate, params)    
    
    msd_df['experiment_type'] = str(experiment_type)
    msd_df['replicate'] = str(replicate)
    
    full_tracked['experiment_type'] = str(experiment_type)
    full_tracked['replicate'] = str(replicate)
    
    
    a  =full_tracked['particle'].tolist()
    z = uuid.uuid4().hex
    a = [str(str(x)+'_'+str(z)) for x in a]
    full_tracked['particle'] = a
    
    full_tracked =  full_tracked.sort_values('particle', ascending=True)
    
    full_tracked.to_csv(str(params.save_path+'__'+experiment_type+'__'+str(replicate)+'__full_tracked_.csv'), header=True, index=None, sep=',', mode='w')
    msd_df.to_csv(str(params.save_path+'__'+experiment_type+'__'+str(replicate)+'__MSD_.csv'), header=True, index=None, sep=',', mode='w')
    
    
def create_big_df(save_path):
    from glob import glob
    files = glob(str(save_path+'*.csv'))
    
    tracks_csv = []
    msd_csv = []
    df_new_full = []
    df_new_msd = []
    for filepath in files:
        if filepath.find('MSD')      != -1:
            msd_csv.append(filepath)
        elif filepath.find('full_tracked_')      != -1: 
            tracks_csv.append(filepath)
            
    for path in tracks_csv:
        df = pd.read_csv(str(path), low_memory=False, sep = ',') 
        df_new_full.append(df)
    df_new_full = pd.concat(df_new_full, ignore_index = True)

    for path in msd_csv:
        df = pd.read_csv(str(path), low_memory=False, sep = ',') 
        df_new_msd.append(df)
    df_new_msd = pd.concat(df_new_msd, ignore_index = True)

    df_new_full.to_csv(str(save_path+'all_tracked.csv'), header=True, index=None, sep=',', mode='w')
    df_new_msd.to_csv(str(save_path+'all_msd.csv'), header=True, index=None, sep=',', mode='w')

    
def create_list_of_vids(path, exp):
    from glob import glob
    files = glob(str(path+'*.tif'))
    files = sorted(files)
    relica_list = np.arange(len(files))
    type_list = [str(exp)] * len(files)
    return files,relica_list,type_list

def main(parampath = None, calibrate = True, **kwargs):
    if parampath is None:
        parampath = "params.yml" if os.path.exists("params.yml") else None
    params = Params(parampath, **kwargs)

    assert params.exp_type_folders is not None, "Please specify experiment folders"
    assert params.save_path is not None, "Please specify a save path"
    assert params.exp_types is not None, "Please specify experiment names"


    list_of_vids = []
    type_list = []
    replica_list = []

    for main_folder in range(len(params.exp_type_folders)):
        file_paths,relica_number,exp_type = create_list_of_vids(params.exp_type_folders[main_folder],params.exp_types[main_folder])
        list_of_vids.extend(file_paths)
        type_list.extend(exp_type)
        replica_list.extend(relica_number)

    if calibrate:
        calibration(params, video_location = list_of_vids[0])
    
    print(list_of_vids)

    for i in range(len(list_of_vids)):
        path = list_of_vids[i]
        lipase = type_list[i]
        replica = replica_list[i]
        runner_tracker(path, lipase, replica, params)
    create_big_df(params.save_path) # saving data


## Functions from stationary savinase / casein tracking

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

def plot_illumination_profile(photon_movie, gauss_meanmov, corrected_movie, save_path, counter):
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
        corrected_movie[ind],
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

def get_correction_profile(meanmov):
    from skimage.filters import gaussian
    gauss_meanmov = gaussian(meanmov, 30)
    gauss_meanmov = gauss_meanmov / gauss_meanmov.max()
    return gauss_meanmov

def Correct_illumination_profile(photon_movie, inplace = True):
    meanmov = np.mean(photon_movie, axis=0)
    gauss_meanmov = get_correction_profile(meanmov)
    if inplace:
        out = photon_movie
    else:
        out = None
    out = np.divide(photon_movie, gauss_meanmov, out = out)
    
    return out

class MockEtsFile:
    def __init__(self, filepath, channel = 0):
        from gpu_tracking import parse_ets
        self.arr = parse_ets(filepath)[0].astype("float32")
        self.shape = self.arr.shape
        
    def asarray(self, key):
        return self.arr[key]

class VideoChunker:
    def __init__(self,
     filepath,
     gb_limit = 1,
     dtype = None,
     n_chunks = None,
     return_frame_correction = True,
     offset = None,
     division = None,
     transform = None,
     auto_apply_transform = True,
     pad_video = False,
     ):
        if filepath.lower().endswith("tif") | filepath.lower().endswith("tif"):
            self.vid_file = tifffile.TiffFile(filepath).series[0]
        elif filepath.lower().endswith("ets"):
            self.vid_file = MockEtsFile(filepath)
        self.frames, self.height, self.width = self.vid_file.shape
        self.return_frame_correction = return_frame_correction
        self.offset = offset
        self.division = division
        self.transform = transform
        self.auto_apply_transform = auto_apply_transform
        self.mean_frame = None
        self.pad_video = pad_video

        if dtype is None:
            self.dtype = np.dtype(np.float32) 
        else:
            self.dtype = np.dtype(dtype)
        self.dtype_size = self.dtype.itemsize
        if n_chunks == None:
            self.byte_limit = gb_limit * (1<<30)
            self.n_chunks = np.ceil(self.frames / (self.byte_limit / (self.dtype_size * self.height * self.width))).astype(int)
        else:
            self.n_chunks = n_chunks
        self.frames_per_load = np.ceil(self.frames / self.n_chunks).astype(int)
        self.__iter__()
    
    def apply_transform(self, video):
        if self.offset is not None:
            video = np.subtract(video, self.offset, out = video)
        if self.division is not None:
            video = np.divide(video, self.division, out = video)
        if self.transform is not None:
            video = self.transform(video)
        return video
    
    def get_mean_frame(self):
        part_means = []
        weights = []
        auto_apply_mem = self.auto_apply_transform
        if self.transform is None:
            self.auto_apply_transform = False
        else:
            self.auto_apply_transform = True
        for frame_correction, vid_chunk in self:
            part_means.append(vid_chunk.mean(axis = 0))
            weights.append(vid_chunk.shape[0])
            del vid_chunk
        self.auto_apply_transform = auto_apply_mem
        self.mean_frame = np.average(part_means, axis = 0, weights = weights)
        if self.transform is None:
            self.apply_transform(self.mean_frame)

    def get_illumination_profile_across_chunks(self):
        if self.n_chunks > 1:
            if self.mean_frame is None:
                self.get_mean_frame()
            correction = get_correction_profile(self.mean_frame)
            if self.transform is None:
                self.division *= correction
            else:
                self.transform = lambda vid: np.divide(self.transform(vid), correction, out = vid)
        else:
            self.transform = lambda vid: Correct_illumination_profile(vid, inplace = True)

    def __iter__(self):
        self.chunk_idx = 0
        return self
    
    def __next__(self):
        if self.chunk_idx == self.n_chunks:
            raise StopIteration
        
        chunk = self.vid_file.asarray(
            key = slice(self.chunk_idx * self.frames_per_load, (self.chunk_idx+1)*self.frames_per_load))
        if self.pad_video:
            padded = np.full((chunk.shape[0], chunk.shape[1] + 1, chunk.shape[2] + 1), np.nan, dtype = self.dtype)
            padded[:, :-1, :-1] = chunk
            chunk = padded[:, :-1, :-1]
        else:
            chunk = chunk.astype(self.dtype)
        
        if self.auto_apply_transform:
            chunk = self.apply_transform(chunk)
            if self.n_chunks == 1:
                self.mean_frame = chunk.mean(axis = 0)
            
        frame_correction = self.chunk_idx * self.frames_per_load
        self.chunk_idx += 1
        if self.return_frame_correction:
            return frame_correction, chunk
        else:
            return chunk

def load_or_fit_random_pixels(no_particle_path, save_path, n_processes):
    no_particle_movie = image_loader_video(no_particle_path)
    movie_hash = hashlib.sha1(no_particle_movie).hexdigest()
    try:
        with open(save_path / "random_pix_fits.pkl", "rb") as file:
            Gs_mean, offs_mean, hash = pickle.load(file)
        if hash != movie_hash:
            raise ValueError
    except (FileNotFoundError, ValueError):
        n_random_pix = 400
        if n_processes > 0:
            pool = mp.Pool(n_processes)
        else:
            pool = None
        
        if pool is not None:
            pixfits = list(tqdm(pool.imap(partial(fit_random_pixel, no_particle_movie), range(n_random_pix)), total = n_random_pix))
        else:
            pixfits = [fit_random_pixel(no_particle_movie) for i in tqdm(range(n_random_pix))] #400 pixels to fit


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
    return Gs_mean, offs_mean

