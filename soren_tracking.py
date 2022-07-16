import numpy as np
import pandas as pd
import trackpy as tp
# import matplotlib  as mpl
import matplotlib.pyplot as plt
# from skimage.feature import peak_local_max
from scipy import ndimage
# from skimage.feature import blob_log
# from skimage import feature
# from scipy.stats.stats import pearsonr 
# import os
# import scipy
import scipy.ndimage as ndimage
# from skimage import measure
# from skimage.color import rgb2gray 
import probfit
import multiprocessing as mp
from contextlib import nullcontext
from functools import partial
  
#from pims import TiffStack
import time
# import matplotlib.patches 
from skimage import util, filters
# from skimage import morphology, util, filters
import time


mean_multiplier = 0.8

sep = 5                 #pixel seperation, ahould be close t search_range
object_size = 9         # 5 originally 
lip_int_size = 9        #originally 14 for first attemps
lip_BG_size = 60        # originally 50 for first attemps

# Tracking parameters
memory = 1              #frame
search_range = 5        # pixels

duration_filter = 0    # frames, originally 5

tracking_time = 0.036   #s --> maybe should change according to exposue 

pixel_size = 0.18333    #µm 60x is 18333 nm

n_processes = 8
n_processes_tp = n_processes
# n_processes_tp = 1


def image_loader_video(video):
    from skimage import io
    im = io.imread(video)
#    images_1 = TiffStack(video) 
    return np.asarray(im)

def crop_img(img):
    """
    Crop the image to select the region of interest
    """
    x_min = 0
    x_max = 1200
    y_min = 0
    y_max = 1200 
    return img[y_min:y_max,x_min:x_max]

def mp_msd(groupbby_tup, **msd_kwargs):
    return groupbby_tup[0], tp.msd(groupbby_tup[1], **msd_kwargs)

def crop_vid(video):
    """
    Crop the image to select the region of interest
    """
    x_min = 0
    x_max = 1200
    y_min = 0
    y_max = 1200 

    return video[:,y_min:y_max,x_min:x_max]

def preprocess_foam(img):
    """
    Apply image processing functions to return a binary image
    
    """
    #img = crop(img)
    # Apply thresholds
    
    adaptive_thresh = filters.threshold_local(img,91)
    idx = img > adaptive_thresh
    idx2 = img < adaptive_thresh
    img[idx] = 0
    img[idx2] = 201
    
    img = ndimage.binary_dilation(img)
    img = ndimage.binary_dilation(img).astype(img.dtype)
    img = ndimage.binary_dilation(img).astype(img.dtype) 
    img = ndimage.binary_dilation(img).astype(img.dtype)
    img = ndimage.binary_dilation(img).astype(img.dtype)
    

    return util.img_as_int(img)

#plt.imshow(preprocess_foam(img))
#pixel_val = preprocess_foam(img)

#print(pixel_val[20])
#list_zero = []

#for i in pixel_val[20].tolist():

    #if i == 0:
        
        #list_zero.append(i)
#print(len(list_zero))


def find_d_single_from_df(r):
    from probfit import BinnedLH, Chi2Regression, Extended,BinnedChi2,UnbinnedLH,gaussian
    import iminuit as Minuit
    r = np.asarray(r)
    def f1(r,D):
        if D> 0:
            return (r/(2.*D*tracking_time))*np.exp(-((r**2)/(4.*D*tracking_time))) # remember to update time
        else:
            return 0
    compdf1 = probfit.functor.AddPdfNorm(f1)
    
    ulh1 = UnbinnedLH(compdf1, r, extended=False)
    import iminuit
    m1 = iminuit.Minuit(ulh1, D=1., limit_D = (0,5),pedantic= False,print_level = 0)
    m1.migrad(ncall=10000)
    return m1.values['D']
     


def step_tracker(df):
    microns_per_pixel = pixel_size
    steps = []
    msd = []
    lag = []
    
    #x_temp = df['x'].astype(int)
    #y_temp = df['y'].astype(int)
    #pixel_list = pixel_val[y_temp,x_temp]
    
    #df['pixel_val'] = pixel_list
    
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
sqrt_BG_size = int(np.ceil(np.sqrt(lip_BG_size)))

def getinds2(smallmask, i_pos, j_pos, video):
    i_lower = int(i_pos) - sqrt_BG_size + 1
    j_lower = int(j_pos) - sqrt_BG_size + 1
    i_upper = int(i_pos) + sqrt_BG_size + 1
    j_upper = int(j_pos) + sqrt_BG_size + 1
    out_of_bounds = i_lower < 0 or j_lower < 0 or\
                    i_upper > video.shape[1] or\
                    j_upper > video.shape[2]
    if out_of_bounds:
        i_inds = slice(max(i_lower, 0),
                        min(i_upper, video.shape[1]))
        j_inds = slice(max(j_lower, 0),
                        min(j_upper, video.shape[2]))
        smallmask_i_lower = max(-i_lower, 0)
        smallmask_j_lower = max(-j_lower, 0)
        smallmask_i_upper = min(video.shape[1]-i_upper, 0)
        smallmask_j_upper = min(video.shape[2]-j_upper, 0)
        smallmask_i_upper = None if smallmask_i_upper == 0 else smallmask_i_upper
        smallmask_j_upper = None if smallmask_j_upper == 0 else smallmask_j_upper
        smallmaskslice = slice(smallmask_i_lower, smallmask_i_upper), slice(smallmask_j_lower, smallmask_j_upper)
        # print(smallmaskslice)
        smallmask = smallmask[smallmaskslice]
    else:
        i_inds = slice(int(i_pos) - sqrt_BG_size + 1, int(i_pos) + sqrt_BG_size + 1)
        j_inds = slice(int(j_pos) - sqrt_BG_size + 1, int(j_pos) + sqrt_BG_size + 1)
    # i_inds, j_inds = j_inds, i_inds
    return i_inds, j_inds, smallmask

def clean(i_pos, j_pos, video, frame):
    frame = int(frame )
    all_is, all_js = np.ogrid[-sqrt_BG_size-(i_pos%1)+1:+sqrt_BG_size-(i_pos%1)+1,
                    -sqrt_BG_size-(j_pos%1)+1:+sqrt_BG_size-(j_pos%1)+1]
    all_r2s = all_is**2 + all_js**2
    sig = all_r2s <= lip_int_size
    bg = all_r2s <= lip_BG_size
    bg = np.logical_xor(bg, sig)

    i_inds, j_inds, smallmask = getinds2(sig, i_pos, j_pos, video)
    out = np.sum(video[frame, i_inds, j_inds][smallmask])
    i_inds, j_inds, smallmask = getinds2(bg, i_pos, j_pos, video)
    out = (out, np.median(video[frame, i_inds, j_inds][smallmask]))
    return out

def share_video_with_subprocesses(raw_array, shape):
    global video
    print("new process")
    video = np.asarray(raw_array).reshape(shape)


def apply_numpy(row):
    # global video
    return clean(row[1], row[0], video, row[2])

def signal_extractor_no_pos(video, full_tracked, red_blue,roi_size,bg_size, pool = None):  # change so taht red initial is after appearance timing
    lip_int_size= roi_size
    lip_BG_size = bg_size
    def cmask(index, array, BG_size, int_size):
        a, b = index
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size   # to make a "gab" between BG and roi
    
        BG_mask = (x * x + y * y <= lip_BG_size)
        BG_mask = np.bitwise_xor(BG_mask, mask2)
        return (sum((array[mask]))), np.median(((array[BG_mask])))
    
    #print(full_tracked)
    full_tracked = full_tracked.sort_values(['frame'], ascending=True)
    

    size_maker = np.ones(video[0].shape)
    ind = 25, 25  # dont ask - leave it here, it just makes sure the below runs
    mask_size, BG_size = cmask(ind, size_maker, lip_BG_size, lip_int_size)
    mask_size = np.sum(mask_size)

    if pool is None:
        a = full_tracked.apply(lambda row: clean(row['y'], row['x'], video, row['frame']), axis=1)
    else:
        print("Using pool")
        a = pool.map(apply_numpy, full_tracked[['x', 'y', 'frame']].to_numpy())
    # a = df_extractor2(final_df, video)

    intensity = []
    bg = []
    for line in a:
        i, b = line
        bg.append(b)
        intensity.append(i)
    if red_blue == 'blue' or red_blue == 'Blue':
        full_tracked['blue_int'] = intensity
        full_tracked['blue_bg'] = bg
        full_tracked['blue_int_corrected'] = (full_tracked['blue_int']) - (full_tracked['blue_bg']*mask_size)
    elif red_blue == 'red' or red_blue == 'Red':
        full_tracked['red_int'] = intensity
        full_tracked['red_bg'] = bg
        full_tracked['red_int_corrected'] = (full_tracked['red_int']) - (full_tracked['red_bg']*mask_size)
    else:
        full_tracked['green_int'] = intensity
        full_tracked['green_bg'] = bg
        full_tracked['green_int_corrected'] = (full_tracked['green_int']) - (full_tracked['green_bg']*mask_size)

    return full_tracked



def tracker(videoinp, mean_multiplier, sep, replicate, save_path):
    # global video
    mean = np.mean(videoinp[0])
    #video = np.asarray(video) #dont use?
    #video_mean = np.median(video,axis = 0) #dont use 
    #video = video -video_mean # dont use 
    
    video = ndimage.gaussian_filter(videoinp, sigma=(0,0.5,0.5), order=0) #dont use 

    # ----- USE THIS WHEN AVG SUBSTRACKTING -----
    #mean_vid = np.mean(video,axis = 0) #finds avg video, use only when tracking in same layer in shor period of time 
    #sub_vid = video - mean_vid     
    #sub_vid =ndimage.gaussian_filter(sub_vid,sigma =(0,0.5,0.5), order =0) # smooting 
    
    #video = sub_vid
    
    full = tp.batch(video, object_size,invert=False, minmass =mean*mean_multiplier,
                    separation= sep, processes = n_processes_tp); # mac har problemer med multiprocessing, derfor processing = 1
                                                               # processes = 8 burde kunne køre på erda. (8 kerner)
    #check for subpixel accuracy
    tp.subpx_bias(full)
    plt.savefig(save_path+str(replicate)+'subpix.png')
    # plt.show()
    

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

            # results = pool.map(I, traj.groupby('particle'))
            # return results
            ids, msds = zip(*results)
        results = tp.motion.pandas_concat(msds, keys=ids)
        results = results.swaplevel(0, 1)[statistic].unstack()
        lagt = results.index.values.astype('float64')/float(fps)
        results.set_index(lagt, inplace=True)
        results.index.name = 'lag time [s]'
        return results
        
    if n_processes > 1:
        start = time.time()
        raw_array_video = mp.RawArray("d", video.flatten())
        end = time.time()
        print("Time to copy video: ", end-start)
        context_man = mp.get_context("spawn").Pool(n_processes, initializer = share_video_with_subprocesses, initargs = (raw_array_video, video.shape))
    else:
        context_man = nullcontext()

    with context_man as pool:
        full = signal_extractor_no_pos(video, full, 'red',lip_int_size,lip_BG_size, pool = pool)
        full_tracked = tp.link_df(full, search_range, memory=memory) #5 pixel search range, memory =2 
        full_tracked = tp.filter_stubs(full_tracked, duration_filter) # filter aour short stuff
        
        
        full_tracked = step_tracker(full_tracked)
        full_tracked['particle'] = full_tracked['particle'].transform(int)
        
        #full_tracked['particle'] = full_tracked['particle'].transform(str)
        #print(full_tracked.groupby('particle')['particle'])
        full_tracked['duration'] = full_tracked.groupby('particle')['particle'].transform(len)
        #full_tracked['frame'] = full_tracked['frame']+1 #made so there is no frame 0
        
        
        #full_tracked['particle'] = full_tracked['particle'].transform(int)
        #full_tracked = full_tracked.sort_values(['frame'], ascending=True) 
        #this should be used
        
        
        
        msd_df = mp_imsd(full_tracked, pixel_size, float(tracking_time),
        max_lagtime = full_tracked["duration"].max(), pool = pool)

    msd_df=msd_df.stack().reset_index()
    msd_df.columns = ['time', 'particle','msd']
    #msd_df =  full_tracked
    return full_tracked, msd_df 

def runner_tracker(video_location, experiment_type, save_path, replicate): 
    
    video = image_loader_video(video_location)
    #video = crop_vid(video)
    
    #video = video[:,500:1000,200:800] #crop video =video[:,y_min_cut:y_max_cut,x_min_cut:x_max_cut]
    
    full_tracked, msd_df = tracker(video,mean_multiplier, sep, replicate, save_path)    
    
    msd_df['experiment_type'] = str(experiment_type)
    msd_df['replicate'] = str(replicate)
    
    full_tracked['experiment_type'] = str(experiment_type)
    full_tracked['replicate'] = str(replicate)
    
    
    import uuid
    a  =full_tracked['particle'].tolist()
    z = uuid.uuid4().hex
    a = [str(str(x)+'_'+str(z)) for x in a]
    full_tracked['particle'] = a
    
    full_tracked =  full_tracked.sort_values('particle', ascending=True)
    
    full_tracked.to_csv(str(save_path+'__'+experiment_type+'__'+str(replicate)+'__full_tracked_.csv'), header=True, index=None, sep=',', mode='w')
    msd_df.to_csv(str(save_path+'__'+experiment_type+'__'+str(replicate)+'__MSD_.csv'), header=True, index=None, sep=',', mode='w')
    
    
def create_big_df(save_path):
    from glob import glob
    files = glob(str(save_path+'*.csv'))
    
    tracks_csv = []
    msd_csv = []
    df_new_full = pd.DataFrame()
    df_new_msd = pd.DataFrame()
    for filepath in files:
        if filepath.find('MSD')      != -1:
            msd_csv.append(filepath)
        elif filepath.find('full_tracked_')      != -1: 
            tracks_csv.append(filepath)
            
    for path in tracks_csv:
        df = pd.read_csv(str(path), low_memory=False, sep = ',') 
        df_new_full = df_new_full.append(df, ignore_index = True)
        
    for path in msd_csv:
        df = pd.read_csv(str(path), low_memory=False, sep = ',') 
        df_new_msd = df_new_msd.append(df, ignore_index = True)
    
    df_new_full.to_csv(str(save_path+'all_tracked.csv'), header=True, index=None, sep=',', mode='w')
    df_new_msd.to_csv(str(save_path+'all_msd.csv'), header=True, index=None, sep=',', mode='w')

    
def create_list_of_vids(path, exp):
    from glob import glob
    files = glob(str(path+'*.tif'))
    files = sorted(files)
    relica_list = np.arange(len(files))
    type_list = [str(exp)] * len(files)
    return files,relica_list,type_list