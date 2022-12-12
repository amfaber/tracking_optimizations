import numpy as np
import tifffile
from skimage.feature import blob_log, peak_local_max

log_space = np.load("testing/fft_version.npy")

idk = peak_local_max(log_space, threshold_abs = 0.1, exclude_border = (0, False, False))

img = tifffile.imread(r"testing/scuffed_blobs_1C.tif").astype("float32")

bah = blob_log(img, min_sigma = 10, max_sigma = 20, num_sigma = 10)

1+1