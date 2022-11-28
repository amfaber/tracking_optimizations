import numpy as np
import tifffile
from skimage.feature import blob_log


img = tifffile.imread(r"testing/scuffed_blobs_1C.tif").astype("float32")

blob_log(img, min_sigma = 10, max_sigma = 20, num_sigma = 10)