import trackpy as tp
import tifffile
if __name__ == "__main__":
    vid = tifffile.imread("../../emily_tracking/sample_vids/s_20.tif")
    batch_result = tp.batch(vid, 9)
    import pickle
    
    with open("locations.pkl", "wb") as file:
        pickle.dump(batch_result, file)
