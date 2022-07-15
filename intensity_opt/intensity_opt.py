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
    

    return np.sum((array[mask2])), np.median(((array[BG_mask]))) # added np in sum

