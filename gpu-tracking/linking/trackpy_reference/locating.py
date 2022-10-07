import trackpy as tp
import tifffile
import pickle
import time


if __name__ == "__main__":
    vid_path = "../../emily_tracking/sample_vids/s_20.tif"
    vid = tifffile.imread(vid_path)
    now = time.time()
    tracked = tp.batch(vid, 9, minmass = 1000)
    print(time.time() - now)
    with open("located.pkl", "wb") as file:
        pickle.dump(tracked, file)
    tracked[["x", "y", "frame"]].to_csv("located.csv", index = False)