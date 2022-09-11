from skimage.filters import gaussian
import numpy as np
from pathlib import Path
from skimage import io
import pickle
from time import time

def image_loader_video(video):
    images_1 = io.imread(video)
    return np.asarray(images_1)

def Correct_illumination_profile(photon_movie):
    # save_path = Path(save_path)
    from skimage.filters import gaussian
    # import torch
    meanmov = np.mean(photon_movie, axis=0)
    gauss_meanmov = gaussian(meanmov, 30)
    # cuda = False
    # if cuda:
    #     cuda_gauss = torch.tensor(gauss_meanmov, requires_grad = False).to("cuda")
    #     cuda_gauss = cuda_gauss / cuda_gauss.max()
    #     scaled_movie = torch.tensor(photon_movie, requires_grad = False).to("cuda") / cuda_gauss
    #     scaled_movie = scaled_movie.cpu().numpy()
    # else:
    gauss_meanmov = gauss_meanmov / gauss_meanmov.max()
    scaled_movie = photon_movie / gauss_meanmov
    return scaled_movie

if __name__ == "__main__":
    start = time()
    folder = Path("sample_vids")
    video_path = str(folder / "s_20.tif")
    video = image_loader_video(video_path)
    # video = video[:200, :, :]
    with open("random_pix_fits.pkl", "rb") as file:
        Gs, offs, phots, pvals = pickle.load(file)
    data_type = np.float64
    photon_movie = (video.astype(data_type) - np.mean(offs[pvals > 0.01], dtype = data_type)) / np.mean(Gs[pvals > 0.01], dtype = data_type)
    corrected_mov = Correct_illumination_profile(photon_movie)
    print(time()-start)