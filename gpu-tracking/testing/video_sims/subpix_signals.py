import numpy as np
from numbers import Number
def subpix_signals(pos, sigmas, picture = None, intensities = None, normalize = False):
    npart = pos.shape[0]
    ndim = pos.shape[1]
    if intensities is None:
        intensities = np.ones(npart)
    
    if isinstance(intensities, Number):
        intensities = np.full(npart, intensities)
    
    if intensities.ndim != ndim + 1:
        intensities = intensities.reshape([-1] + [1]*ndim)
    
    if isinstance(sigmas, Number):
        sigmas = np.full(ndim, sigmas)

    if normalize:
        intensities /= np.prod(sigmas)
    subpix_pos = pos % 1
    shift = subpix_pos > 0.5
    coord_inds = pos.astype(int) + shift
    subpix_pos[shift] -= 1
    rs = np.array(sigmas)*3
    rs = np.ceil(rs).astype(int)
    sqrt2pi = np.sqrt(2*np.pi)
    gauss = lambda r2: 1/(sqrt2pi)**ndim*np.exp(-r2/(2))
    
    # Create an open grid going from -r to r + 1 for ndim dimensions
    grids = [np.moveaxis(np.arange(-r, r + 1).reshape([-1] + [1]*(ndim - 1)), 0, dim) for dim, r in enumerate(rs)]
    
    # Insert a leading dimension that will represent the particles
    grids = [grid[None, ...] for grid in grids]

    # For each particle, find the distance between the pixels in the grid and the particle.
    grids = [grid - subpix_pos[:, dim].reshape([-1] + [1]*ndim) for dim, grid in enumerate(grids)]
    # Transform by sigma
    grids = [grid / sigma for grid, sigma in zip(grids, sigmas)]
    # r^2 in the transformed space
    r2 = sum(grid**2 for grid in grids)

    signals = gauss(r2)
    signals *= intensities

    # Possiblity of printing the integral of every signal to ensure normalization
    # print(signals.reshape(npart, -1).sum(1))
    
    if picture is None:
        vid_sizes = np.ceil(pos.max(axis = 0)).astype(int) + 1
        picture = np.zeros(vid_sizes)
    
    # Now all the signals are generated in their 0-centered coordinates. Time to insert them into the picture.
    # Some ugly bounds are calculated to ensure that pixels inserted at the edge of the picture don't
    # try to access indices outside the picture.
    clip_bounds = [np.array([0]*ndim), np.array(picture.shape)-1]
    for i, (part, pos) in enumerate(zip(signals, coord_inds)):
        # Bounds are shaped (2, ndim). Columns represent spatial coordinates
        bounds = np.tile(pos, (2, 1))
        # First are lower bounds, second row are upper bounds
        bounds[0, :] += -rs
        bounds[1, :] += rs + 1
        # Clipped makes sure we don't attempt to access elements outside the picture
        clipped = bounds.clip(clip_bounds[0], clip_bounds[1])
        # Change makes sure we also crop the signal to the same shape.
        change = clipped - bounds
        little_inds = np.array([(0,)*ndim, part.shape])
        little_inds += change
        picture[tuple(slice(*dim) for dim in clipped.T)] += part[tuple(slice(*dim) for dim in little_inds.T)]

    return picture

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time
    # from scipy.stats import multivariate_normal
    from matplotlib.animation import FuncAnimation
    vid_size = 200
    # pos = np.random.random((200, 3))*(vid_size-1)
    pos = np.random.normal(size = (1000, 3), scale = 1)
    picture = np.zeros((vid_size, )*pos.shape[1])
    ndim = pos.shape[1]
    # pos = pos/(np.sqrt((pos ** 2).sum(axis = 1)))
    poslen = np.sqrt((pos**2).sum(axis = 1))
    pos[poslen > 1] /= poslen[poslen > 1][:, None]
    pos = pos*(vid_size-1)*0.45 + np.array([vid_size]*ndim)*0.5
    # pos = np.array([199, 199, 0]).reshape(-1, 3)
    now = time()
    pic3d = subpix_signals(pos, [2, 2, 4], normalize = False, picture = picture)#, r_pic = True)
    # plt.imshow(pic3d)
    # plt.show()
    print(f"{time()-now}")
    pic = lambda i: pic3d[:, :, i]


    def gen_anim(frames, fov_dimx, fov_dimy, n_anim, frame_ms):
        fig, ax = plt.subplots(1,1,figsize=(12,8))
        im = ax.imshow(frames[:,:,0].T, cmap = "viridis", vmin = 0, vmax = 0.1)
        #ax.set_xlim(0, fov_dimx-1)
        #ax.set_ylim(0, fov_dimy-1)
        def animate(n_frame):
            return im.set_data(frames[:,:,n_frame].T)

        anim = FuncAnimation(fig, animate, frames=n_anim, interval=frame_ms)
        plt.show()
        # plt.close()
        return

    gen_anim(pic3d, 0, 0, vid_size, 10)