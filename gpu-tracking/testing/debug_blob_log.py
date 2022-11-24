import numpy as np
import tifffile
from skimage.feature import blob_log

img = tifffile.imread(r"scuffed_blobs_1C.tif")

blob_log(img)