import pandas as pd
import numpy as np


DAT = pd.read_csv("../sample_output/full_tracked.csv")

def image_loader_video(video):
    from skimage import io
    im = io.imread(video)
#    images_1 = TiffStack(video) 
    return np.asarray(im)

VID = image_loader_video("../sample_vids/Process_8967_488_first5.tif")


lip_int_size = 9   #originally 14 for first attempts
lip_BG_size = 60
sqrt_BG_size = int(np.ceil(np.sqrt(lip_BG_size)))

def df_extractor2(row, video):

    b, a = row['x'], row['y'] #b,a
    frame = int(row['frame'])
    array = video[frame]
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    r2 = x * x + y * y
    # mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
    mask2 = r2 <= lip_int_size  # to make a "gab" between BG and roi
    BG_mask = (r2 <= lip_BG_size)
    BG_mask = np.bitwise_xor(BG_mask, mask2)
    
    # return mask2
    return np.sum((array[mask2])), np.median(((array[BG_mask]))) # added np in sum


def getinds(smallmask, i_pos, j_pos, video):
    i_inds, j_inds = np.where(smallmask)
    i_transform = int(i_pos) - sqrt_BG_size + 1
    j_transform = int(j_pos) - sqrt_BG_size + 1
    i_inds += i_transform
    j_inds += j_transform
    out_of_bounds = i_transform < 0 or j_transform < 0 or\
                    i_transform + len(smallmask)-1 >= video.shape[1] or\
                    j_transform + len(smallmask)-1 >= video.shape[2]
    if out_of_bounds:
        valid = (i_inds >= 0) & (j_inds >= 0) & (i_inds < video.shape[1]) & (j_inds < video.shape[2])
        i_inds = i_inds[valid]
        j_inds = j_inds[valid]
    return i_inds, j_inds 

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

def new(i_pos, j_pos, method = 1, frame = 0):
    all_is, all_js = np.ogrid[-sqrt_BG_size-(i_pos%1)+1:+sqrt_BG_size-(i_pos%1)+1,
                    -sqrt_BG_size-(j_pos%1)+1:+sqrt_BG_size-(j_pos%1)+1]
    all_r2s = all_is**2 + all_js**2
    sig = all_r2s <= lip_int_size
    bg = all_r2s <= lip_BG_size
    bg = np.logical_xor(bg, sig)

    def sort_it_out(method):
        if method == 1:
            out = np.sum(VID[(frame, *getinds(sig, i_pos, j_pos, VID))]), np.median(VID[(frame, *getinds(bg, i_pos, j_pos, VID))])
        elif method == 2:
            i_inds, j_inds, smallmask = getinds2(sig, i_pos, j_pos, VID)
            try:
                out = np.sum(VID[frame, i_inds, j_inds][smallmask])
            except IndexError as e:
                print(f"{frame = }")
                print(f"{i_inds = }")
                print(f"{j_inds = }")
                print(f"{smallmask = }")
                print(f"{smallmask.shape = }")
                raise e
            i_inds, j_inds, smallmask = getinds2(bg, i_pos, j_pos, VID)
            out = (out, np.median(VID[frame, i_inds, j_inds][smallmask]))
        return out
    out = sort_it_out(method)
    return out

def old(i_pos, j_pos):
    row = pd.Series({'x': i_pos, 'y': j_pos, "frame": 0})
    return df_extractor2(row, VID)


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

def apply_clean(row):
    return clean(row['y'], row['x'], VID, row['frame'])