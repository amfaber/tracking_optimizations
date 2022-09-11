from skimage.filters import gaussian
import numpy as np
from pathlib import Path
from skimage import io
import pickle

def image_loader_video(video):
    images_1 = io.imread(video)
    return np.asarray(images_1)

def Correct_illumination_profile(photon_movie):
    # save_path = Path(save_path)
    from skimage.filters import gaussian
    import torch
    meanmov = np.mean(photon_movie, axis=0)
    gauss_meanmov = gaussian(meanmov, 30)
    cuda_gauss = torch.tensor(gauss_meanmov, requires_grad = False).to("cuda")
    cuda_gauss = cuda_gauss / cuda_gauss.max()
    # gauss_meanmov = gauss_meanmov / gauss_meanmov.max()

    # scaled_movie = photon_movie / gauss_meanmov
    scaled_movie = torch.tensor(photon_movie, requires_grad = False).to("cuda") / cuda_gauss
    scaled_movie = scaled_movie.numpy()
    return scaled_movie

if __name__ == "__main__":
    folder = Path("sample_vids")
    video = str(folder / "s_20.tif")
    video_path = Path()
    video = image_loader_video(video)
    video = video[:200, :, :]
    with open("random_pix_fits.pkl", "rb") as file:
        Gs, offs, phots, pvals = pickle.load(file)
    photon_movie = (video - np.mean(offs[pvals > 0.01])) / np.mean(Gs[pvals > 0.01])
    corrected_mov = Correct_illumination_profile(photon_movie)