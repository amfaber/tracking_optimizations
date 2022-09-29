from cmath import rect
from signal import signal
from turtle import position
import numpy as np
import pandas as pd
from trackpy.utils import validate_tuple, default_pos_columns, guess_pos_columns, default_size_columns
import warnings
import torch
from trackpy.find import where_close
from time import time

def setup(vid_shape, diameter, device, minmass=None, maxsize=None, separation=None,
           noise_size=1, smoothing_size=None, threshold=None,
           percentile=64, topn=None, preprocess=True, max_iterations=10, sig_r2 = None,
           bg_r2 = None, gap_size2 = None):
    torch.set_grad_enabled(False)
    parameters = {}
    parameters["general"] = {}
    parameters["bandpass"] = {}
    parameters["grey_dilation"] = {}
    parameters["refinement"] = {}
    parameters["signal_extraction"] = {}

    # Validate parameters and set defaults.
    ndim = len(vid_shape) - 1
    # General settings

    diameter = validate_tuple(diameter, ndim)
    diameter = tuple([int(x) for x in diameter])
    if not np.all([x & 1 for x in diameter]):
        raise ValueError("Feature diameter must be an odd integer. Round up.")
    radius = tuple([x//2 for x in diameter])

    isotropic = np.all(radius[1:] == radius[:-1])
    if (not isotropic) and (maxsize is not None):
        raise ValueError("Filtering by size is not available for anisotropic "
                         "features.")
    if minmass is None:
        minmass = 0
    
    pos_columns = default_pos_columns(ndim)

    parameters["general"]["shape"] = vid_shape
    parameters["general"]["ndim"] = ndim
    parameters["general"]["pos_columns"] = pos_columns
    parameters["general"]["minmass"] = minmass
    parameters["general"]["maxsize"] = minmass
    parameters["general"]["preprocess"] = preprocess
    parameters["general"]["topn"] = topn

    # bandpass
    threshold = 0
    noise_size = validate_tuple(noise_size, ndim)
    if smoothing_size is None:
        smoothing_size = diameter
    else:
        smoothing_size = validate_tuple(smoothing_size, ndim)
    if np.any([x >= y for (x, y) in zip(noise_size, smoothing_size)]):
        raise ValueError("The smoothing length scale must be larger than " +
                         "the noise length scale.")
    truncate = 4
    padder = (*(np.array(smoothing_size) // 2),)*2
    filter = torch.full(smoothing_size, 1/np.product(smoothing_size), device = device).reshape(1, 1, *smoothing_size)
    gauss_filter = make_gaussian_filter(noise_size, truncate, device)

    parameters["bandpass"]["gauss_filter"] = gauss_filter
    parameters["bandpass"]["filter"] = filter
    parameters["bandpass"]["threshold"] = threshold
    parameters["bandpass"]["padder"] = padder

    # grey dilation
    if separation is None:
        separation = tuple([x + 1 for x in diameter])
    else:
        separation = validate_tuple(separation, ndim)
    
    size = [int(2 * s / np.sqrt(ndim)) for s in separation]
    shape = torch.tensor(vid_shape[2:], device = device)
    padding = (*(np.array(size) // 2).astype(int),)
    margin = torch.tensor([max(rad, sep // 2 - 1, sm // 2) for (rad, sep, sm) in
                    zip(radius, separation, smoothing_size)], device = device)

    parameters["grey_dilation"]["size"] = size
    parameters["grey_dilation"]["shape"] = shape
    parameters["grey_dilation"]["padding"] = padding
    parameters["grey_dilation"]["margin"] = margin
    parameters["grey_dilation"]["percentile"] = percentile
    parameters["grey_dilation"]["ndim"] = ndim

    # refinement
    radii = torch.tensor(radius, device = device)
    shift_thresh = 0.6
    shift_thresh = torch.tensor(shift_thresh, device = device)
    zero_centered_grids = [torch.moveaxis(torch.arange(-r, r + 1, device = device).reshape([-1] + [1]*(ndim-1)), 0, i) for i, r in enumerate(radii)]
    r2 = sum([grid**2/r**2 for grid, r in zip(zero_centered_grids, radii)])
    mask = r2 <= 1
    upper_bound = torch.tensor(vid_shape[1:], device = device) - 1 - radii
    shift_thresh = 0.6

    nan = torch.tensor(float("nan"), device = device)
    parameters["refinement"]["radii"] = radii
    parameters["refinement"]["circle_mask"] = mask
    parameters["refinement"]["upper_bound"] = upper_bound
    parameters["refinement"]["zero_centered_grids"] = zero_centered_grids
    parameters["refinement"]["max_iterations"] = max_iterations
    parameters["refinement"]["shift_thresh"] = shift_thresh
    parameters["refinement"]["ndim"] = ndim
    parameters["refinement"]["nan"] = nan

    # Signal extraction
    if sig_r2 is None:
        sig_r2 = diameter[0] / 2
        # sig_r2 = (torch.tensor(diameter) / 2) ** 2
    else:
        # sig_r2 = validate_tuple(sig_r2, ndim)
        # sig_r2 = torch.tensor(sig_r2)
        pass
    
    if bg_r2 is None:
        bg_r2 = 4 * sig_r2
    else:
        # bg_r2 = validate_tuple(bg_r2, ndim)
        # bg_r2 = torch.tensor(bg_r2)
        pass

    if gap_size2 is None:
        #(r + k*r)**2 = r2 + (k*r)**2 + 2*r*(k*r) = r2 + k**2 * r2 + 2*k*r2 = r2 + r2 * (k**2 + 2*k)
        factor_increase = 1.1
        gap_size2 = sig_r2 * (factor_increase**2 + 2*factor_increase)
    
    # r_ceils = torch.ceil(np.sqrt(torch.tensor(bg_r2))).to(int)
    # signal_grids = [torch.moveaxis(torch.arange(-r_ceil + 1, r_ceil + 1, device = device).reshape([-1] + [1]*(ndim)), 0, dim + 1) for dim, r_ceil in enumerate(r_ceils)]
    r_ceil = torch.ceil(np.sqrt(torch.tensor(bg_r2))).to(int)
    signal_grids = [torch.moveaxis(torch.arange(-r_ceil + 1, r_ceil + 1, device = device).reshape([-1] + [1]*ndim), 0, dim + 1) for dim in range(ndim)]

    parameters["signal_extraction"]["sig_r2"] = sig_r2
    parameters["signal_extraction"]["bg_r2"] = bg_r2
    parameters["signal_extraction"]["gap_size2"] = gap_size2
    parameters["signal_extraction"]["signal_grids"] = signal_grids
    parameters["signal_extraction"]["nan"] = nan

    
    return parameters



def minibatch(tensor, factor = 5):
    
    step = np.ceil(tensor.shape[0] / factor).astype(int)
    start = 0
    end = step
    while start < tensor.shape[0]:
        yield start, tensor[start:end]
        start += step
        end += step

def locate(raw_video, diameter, minmass=None, maxsize=None, separation=None,
           noise_size=1, smoothing_size=None, threshold=None,
           percentile=64, topn=None, preprocess=True, max_iterations=10, padded_vid = None, params = None):
    
    params = setup(vid_shape = raw_video.shape, diameter = diameter, device = raw_video.device, minmass=minmass,
            maxsize=maxsize, separation=separation, noise_size=noise_size, smoothing_size=smoothing_size,
            threshold=threshold, percentile=percentile, topn=topn, preprocess=preprocess,
            max_iterations=max_iterations)
    gparams = params["general"]
    refined_coords = []
    for frame_start, video in minibatch(raw_video, factor = 20):
        if gparams["preprocess"]:
            video = bandpass(video, **params["bandpass"])
        
        candidates = grey_dilation(video, **params["grey_dilation"])

        batch_coords = _refine(raw_video, video, candidates, frame_start, **params["refinement"])
        refined_coords.append(batch_coords)
        frames = torch.arange(frame_start, frame_start + video.shape[0])
        part_inds = batch_coords[:, 0:3]
        signals = fast_signal_extraction(raw_video, frames, frame_start, part_inds, **params["signal_extraction"])

    refined_coords = torch.concat(refined_coords)
    
    return refined_coords
    # return refined_coords
    if len(refined_coords) == 0:
        return refined_coords

    # Flat peaks return multiple nearby maxima. Eliminate duplicates.
    if np.all(np.greater(separation, 0)):
        all_to_drop = []
        offset = 0
        for frame, data in refined_coords.groupby("frame"):
            to_drop = where_close(data[pos_columns], separation,
                                data['mass'])
            to_drop += offset
            offset += len(data)
            all_to_drop.append(to_drop)
        all_to_drop = np.concatenate(all_to_drop)
        refined_coords.drop(all_to_drop, axis=0, inplace=True)
        refined_coords.reset_index(drop=True, inplace=True)

    # mass and signal values has to be corrected due to the rescaling
    # raw_mass was obtained from raw video; size and ecc are scale-independent
    # refined_coords['mass'] /= scale_factor
    # if 'signal' in refined_coords:
    #     refined_coords['signal'] /= scale_factor

    # Filter on mass and size, if set.
    condition = refined_coords['mass'] > minmass
    if maxsize is not None:
        condition &= refined_coords['size'] < maxsize
    if not condition.all():  # apply the filter
        # making a copy to avoid SettingWithCopyWarning
        refined_coords = refined_coords.loc[condition].copy()

    if len(refined_coords) == 0:
        warnings.warn("No maxima survived mass- and size-based filtering. "
                      "Be advised that the mass computation was changed from "
                      "version 0.2.4 to 0.3.0 and from 0.3.3 to 0.4.0. "
                      "See the documentation and the convenience functions "
                      "'minmass_v03_change' and 'minmass_v04_change'.")
        return refined_coords

    if topn is not None and len(refined_coords) > topn:
        # go through numpy for easy pandas backwards compatibility
        mass = refined_coords['mass'].values
        if topn == 1:
            # special case for high performance and correct shape
            refined_coords = refined_coords.iloc[[np.argmax(mass)]]
        else:
            refined_coords = refined_coords.iloc[np.argsort(mass)[-topn:]]

    # If this is a pims Frame object, it has a frame number.
    # Tag it on; this is helpful for parallelization.
    if hasattr(raw_video, 'frame_no') and raw_video.frame_no is not None:
        refined_coords['frame'] = int(raw_video.frame_no)
    return refined_coords

def bandpass(video, padder, filter, gauss_filter, threshold):
    
    video = torch.unsqueeze(video, 1)
    
    
    padded_video = torch.nn.functional.pad(video, padder, "replicate")
    background = torch.nn.functional.conv2d(padded_video, filter)
    del padded_video
    result = torch.nn.functional.conv2d(video, gauss_filter, padding = "same")
    del video
    result -= background
    del background
    result = torch.where(result >= threshold, result, 0)
    result = result.squeeze(dim = 1)
    return result

def make_gaussian_filter(lshort, truncate, device):
    # now = time()
    lws = [int(truncate * lsh + 0.5) for lsh in lshort]
    grids = [torch.moveaxis(torch.arange(-lw, lw + 1, device = device)[(slice(None), ) + (None,)*(len(lshort)-1)], 0, i) for i, lw in enumerate(lws)]
    r2 = sum([grid**2 for grid in grids])
    #TODO skewed gaussian support
    res = torch.exp(r2/(-2*lshort[0]**2))
    res /= res.sum()
    res = res.reshape(1, 1, *res.shape)
    # print(f"making gaussian {time()-now}")
    return res

def convert_to_int(video):
    vid_max = video.max()
    factor = 255 / vid_max
    output = (video * vid_max).to(torch.uint8)
    return factor, output

def grey_dilation(video, margin, percentile, size, shape, padding, ndim):
    # Find the largest box that fits inside the ellipse given by separation
    video = torch.unsqueeze(video, 1)
    nframes = video.shape[0]
    threshold = video.reshape(nframes, -1).quantile(percentile / 100, dim = -1).reshape([-1] + [1]*(ndim))
    
    dilation = torch.nn.functional.max_pool2d(video, size, stride = 1, padding = padding)
    dilation = torch.squeeze(dilation, dim = 1)
    video = torch.squeeze(video, dim = 1)
    if dilation.shape != video.shape:
        dilation = dilation[(slice(None), ) + (slice(1, None), )*ndim]
    
    maxima = (video == dilation) & (video > threshold)
    candidates = torch.nonzero(maxima)

    if torch.sum(maxima) == 0:
        warnings.warn("video contains no local maxima.", UserWarning)
        return np.empty((0, ndim))

    pos = candidates[:, 1:]
    # Do not accept peaks near the edges.
    near_edge = ((pos < margin) | (pos > (shape - margin - 1))).any(axis = 1)
    pos = pos[~near_edge]
    # exclude_channel = torch.tensor([True, False] + [True]*ndim)
    candidates = candidates[~near_edge]#[:, exclude_channel]
    if len(pos) == 0:
        warnings.warn("All local maxima were in the margins.", UserWarning)
        return np.empty((0, ndim))


    return candidates



def _refine(raw_video, video, candidates, frame_start, max_iterations,
            shift_thresh, zero_centered_grids, radii, upper_bound, circle_mask, ndim, nan):
    
    device = video.device
    # Columns:
    # {0: "y", 1: "x", 2: "mass", 3: "signal"}

    pos = candidates[:, 1:]
    frames = candidates[:, 0].reshape(-1, 1, 1)
    N = candidates.shape[0]
    results = torch.full((N, 4), nan, dtype = torch.float32, device = device)
    not_done = torch.arange(pos.shape[0], device = device)

    for _ in range(max_iterations):
        spatial_inds = tuple(zero_centered_grids[dim] + pos[:, dim].reshape([-1] + [1]*ndim) for dim in range(ndim))
        all_inds = (frames, *spatial_inds)
        neighborhoods = video[all_inds]
        neighborhoods = torch.multiply(neighborhoods, circle_mask, out = neighborhoods)
        masses = neighborhoods.sum(axis = (1, 2))
        cm = torch.stack([(neighborhoods * grid).sum(axis = (1, 2)) / masses for grid in zero_centered_grids], axis = 1)
        if (masses == 0).sum() > 0:
            cm[masses == 0] = torch.tensor([0, 0], device = device, dtype = cm.dtype)
        # off_center = cm - radii
        back_shift = cm < -shift_thresh
        forward_shift = cm > shift_thresh
        keep_iterating = (back_shift | forward_shift).any(axis = 1)
        leaving_iteration = ~keep_iterating
        if (leaving_iteration).sum() != 0:
            newly_done = not_done[leaving_iteration]
            results[newly_done, :2] = cm[leaving_iteration] + pos[leaving_iteration]
            results[newly_done, 2] = masses[leaving_iteration]
            results[newly_done, 3] = neighborhoods[leaving_iteration].reshape(leaving_iteration.sum(), -1).max(dim = -1)[0]
        

        if keep_iterating.sum() == 0:
            break
        pos[back_shift] -= 1
        pos[forward_shift] += 1
        # # pass
        pos = pos[keep_iterating, :].clip(radii, upper_bound)
        frames = frames[keep_iterating, :, :]
        not_done = not_done[keep_iterating]

    if not keep_iterating.sum() == 0:
        print(f"{keep_iterating.sum() / len(not_done) * 100}% of particles didn't converge")
        rest = not_done[keep_iterating]
        results[rest, :2] = cm[keep_iterating] + pos[keep_iterating]
        results[rest, 2] = masses[keep_iterating]
        results[rest, 3] = neighborhoods[keep_iterating].max()

    frames = candidates[:, 0].reshape(-1, 1, 1) + frame_start

    # final_pixels = torch.round(final_coords).to(int)
    # raw_masses = (raw_video[frames, make_inds_for_dim(0, final_pixels), make_inds_for_dim(1, final_pixels)] * mask).sum(axis = (1,2))

    return torch.column_stack((torch.squeeze(frames), results))

def fast_signal_extraction(raw_video, raw_frames, frame_start, particle_inds, bg_r2, sig_r2, gap_size2, signal_grids, nan, return_means = False):
    #device = raw_video.device
    ndim = raw_video.ndim - 1
    frames = particle_inds[:, 0].to(int).reshape([-1] + [1]*ndim) - frame_start
    positions = particle_inds[:, 1:3]
    sub_pix = positions % 1
    pix = positions.to(int)
    to_pad = (0, 1)*ndim
    padded_video = torch.nn.functional.pad(raw_video[raw_frames], to_pad, value = nan)
    # difference_for_dimension = lambda dim: torch.moveaxis(torch.arange(
    #     -r_ceil + 1, r_ceil + 1, device = device) - sub_pix[:, dim].reshape(-1, 1, 1), -1, dim + 1)
    # inds_for_dimension = lambda dim: torch.moveaxis((
    #     torch.arange(-r_ceil + 1, r_ceil + 1, device = device)) + pix[:, dim].reshape(-1, 1, 1), -1, dim+1).clip(
    #         -1, padded_video.shape[dim + 1] - 1)

    diffs = [grid - sub_pix[:, dim].reshape([-1] + [1]*ndim) for dim, grid in enumerate(signal_grids)]

    r2 = sum(diff**2 for diff in diffs)
    grids = [(grid - pix[:, dim].reshape([-1] + [1]*ndim)).clip(-1, raw_video.shape[dim + 1]) for dim, grid in enumerate(signal_grids)]
    
    rectangles = padded_video[(frames, *grids)]
    del padded_video

    sigmasks = r2 <= sig_r2
    if gap_size2 != 0:
        bg_inner_mask = r2 <= sig_r2 + gap_size2
    else:
        bg_inner_mask = sigmasks

    bgmasks = torch.logical_xor(r2 <= bg_r2, bg_inner_mask)
    
    #nan = torch.tensor(float("nan"), dtype = torch.float32).to(device)
    
    signals = torch.where(sigmasks, rectangles, nan).reshape(positions.shape[0], -1)
    backgrounds = torch.where(bgmasks, rectangles, nan).reshape(positions.shape[0], -1)
    del rectangles

    sig_sums = torch.nansum(signals, axis = 1)
    bg_medians = torch.nanmedian(backgrounds, axis = 1)[0]
    sig_sizes = torch.nansum(sigmasks.reshape(positions.shape[0], -1), axis = 1)
    out = [sig_sums, bg_medians, sig_sizes]
    if return_means:
        sig_means = sig_sums / sig_sizes
        bg_sums = torch.nansum(backgrounds, axis = 1)
        bg_sizes = torch.nansum(bgmasks.reshape(positions.shape[0], -1), axis = 1)
        bg_means = bg_sums / bg_sizes
        out.extend([sig_means, bg_means])
    out = torch.column_stack(out)
    
    return out





def refine_com(raw_video, video, radius, coords, max_iterations=10,
               shift_thresh=0.6,
               pos_columns=None, padded_video = None, params = None):
    
    if isinstance(coords, pd.DataFrame):
        if pos_columns is None:
            pos_columns = guess_pos_columns(coords)
        index = coords.index
        coords = coords[pos_columns].values
    else:
        index = None
    ndim = video.ndim - 1
    radius = validate_tuple(radius, ndim)
    
    if pos_columns is None:
        pos_columns = default_pos_columns(ndim)
    columns = pos_columns + ['mass'] + ["signal"]
    # if characterize:
    #     isotropic = radius[1:] == radius[:-1]
    #     columns += default_size_columns(ndim, isotropic) + \
    #         ['ecc', 'signal', 'raw_mass']

    if len(coords) == 0:
        return pd.DataFrame(columns=columns)

    refined = refine_com_arr(raw_video, video, radius, coords,
                             max_iterations=max_iterations, shift_thresh=shift_thresh)
    if padded_video is not None:
        test = fast_signal_extraction(refined[:, 1:3], refined[:, 0], radius[0], padded_video, params)
    columns = ["frame"] + columns 
    return pd.DataFrame(refined.cpu().numpy(), columns=columns, index=index).astype({"frame": int})





def refine_com_arr(raw_video, video, radius, coords, max_iterations=10,
                   shift_thresh=0.6, characterize=True):
    if max_iterations <= 0:
        warnings.warn("max_iterations has to be larger than 0. setting it to 1.")
        max_iterations = 1
    ndim = raw_video.ndim - 1
    if ndim + 1 != coords.shape[1]:
        raise ValueError("The video has a different number of dimensions than "
                         "the coordinate array.")

    # ensure that radius is tuple of integers, for direct calls to refine_com_arr()
    radius = validate_tuple(radius, ndim)
    # Main loop will be performed in separate function.
    # In here, coord is an integer. Make a copy, will not modify inplace.
    if coords.is_floating_point():
        coords = torch.round(coords).to(int)
    results = _refine(raw_video, video, radius, coords, max_iterations,
                          shift_thresh)

    return results