import tifffile
import numpy as np
import psutil
print(psutil.Process().memory_info()[0] / (1 << 20))
idk = tifffile.TiffFile("/Users/amfaber/Documents/tracking_script/sample_vids/Process_8967_488.tif")
a = np.zeros((
    idk.series[0].shape[0], 
    # 10,
    idk.series[0].shape[1] + 1,
    idk.series[0].shape[2] + 1), dtype = np.float64)
print(psutil.Process().memory_info()[0] / (1 << 20))
# b = np.empty((idk.series[0].shape[0], idk.series[0].shape[1], idk.series[0].shape[2]), dtype = np.uint16)

# c = a[:, :-1, :-1]
# idk.filehandle.seek(idk.series[0].dataoffset)
# nbytes = idk.series[0].size*idk.series[0].dtype.itemsize

# c[:] = np.frombuffer(idk.filehandle.read(nbytes), dtype = np.uint16).reshape(c.shape).astype(np.float64)

print(psutil.Process().memory_info()[0] / (1 << 20))
