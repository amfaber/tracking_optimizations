import numpy as np
import pandas as pd
from trackpy.utils import validate_tuple, default_pos_columns, guess_pos_columns, default_size_columns
import warnings
import torch
from trackpy.find import where_close

# class MiniBatcher:
#     def __init__(self, tensor, factor = 1):
#         self.tensor = tensor
#         self.i = 0
#         self.factor = factor
#         self.step = np.ceil(self.tensor.shape[0] / factor).astype(int)

#     def __iter__(self):
#         self.i = 0
#         return self

#     def __next__(self):
#         start = self.i*self.step
#         end = (self.i+1)*self.step
#         self.i += 1
#         return self.tensor[start:end]
# PYTORCH_ENABLE_MPS_FALLBACK=1
# import os
# print(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"])
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
    
    torch.set_grad_enabled(False)
    device = raw_video.device
    # Validate parameters and set defaults.
    raw_video = torch.squeeze(raw_video)
    shape = raw_video.shape
    ndim = raw_video.ndim - 1

    diameter = validate_tuple(diameter, ndim)
    diameter = tuple([int(x) for x in diameter])
    if not np.all([x & 1 for x in diameter]):
        raise ValueError("Feature diameter must be an odd integer. Round up.")
    radius = tuple([x//2 for x in diameter])

    isotropic = np.all(radius[1:] == radius[:-1])
    if (not isotropic) and (maxsize is not None):
        raise ValueError("Filtering by size is not available for anisotropic "
                         "features.")

    is_float_video = raw_video.is_floating_point()

    if separation is None:
        separation = tuple([x + 1 for x in diameter])
    else:
        separation = validate_tuple(separation, ndim)

    if smoothing_size is None:
        smoothing_size = diameter
    else:
        smoothing_size = validate_tuple(smoothing_size, ndim)

    noise_size = validate_tuple(noise_size, ndim)

    if minmass is None:
        minmass = 0

    # Check whether the video looks suspiciously like a color video.
    if 3 in shape or 4 in shape:
        dim = raw_video.ndim - 1
        warnings.warn("I am interpreting the video as {}-dimensional. "
                      "If it is actually a {}-dimensional color video, "
                      "convert it to grayscale first.".format(dim, dim-1))

    if threshold is None:
        if is_float_video:
            threshold = 1/255.
        else:
            threshold = 1

    # Determine `video`: the video to find the local maxima on.
    if preprocess:
        video = bandpass(raw_video, noise_size, smoothing_size, threshold)
    else:
        video = raw_video

    # For optimal performance, performance, coerce the video dtype to integer.
    if is_float_video:  # For float videos, assume bitdepth of 8.
        dtype = np.uint8
    else:   # For integer videos, take original dtype
        dtype = raw_video.dtype
    # Normalize_to_int does nothing if video is already of integer type.
    # scale_factor, video = convert_to_int(video)

    pos_columns = default_pos_columns(ndim)

    # Find local maxima.
    # Define zone of exclusion at edges of video, avoiding
    #   - Features with incomplete video data ("radius")
    #   - Extended particles that cannot be explored during subpixel
    #       refinement ("separation")
    #   - Invalid output of the bandpass step ("smoothing_size")
    margin = torch.tensor([max(rad, sep // 2 - 1, sm // 2) for (rad, sep, sm) in
                    zip(radius, separation, smoothing_size)], device = device)
    # Find features with minimum separation distance of `separation`. This
    # excludes detection of small features close to large, bright features
    # using the `maxsize` argument.
    candidates = grey_dilation(video, separation, percentile, margin)
    # Refine their locations and characterize mass, size, etc.

    refined_coords = refine_com(raw_video, video, radius, candidates,
                                max_iterations=max_iterations, padded_video = padded_vid, params = params)
    
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

def bandpass(video, lshort, llong, threshold=None, truncate=4):
    ndim = video.ndim - 1
    lshort = validate_tuple(lshort, ndim)
    llong = validate_tuple(llong, ndim)
    if np.any([x >= y for (x, y) in zip(lshort, llong)]):
        raise ValueError("The smoothing length scale must be larger than " +
                         "the noise length scale.")
    if threshold is None:
        if video.is_floating_point():
            threshold = 1/255.
        else:
            threshold = 1
    
    device = video.device
    # nframes = video.shape[0]
    video = torch.unsqueeze(video, 1)
    # video = video.reshape(nframes, 1, *video.shape[1:])
    filter = torch.full(llong, 1/np.product(llong), device = device).reshape(1, 1, *llong)
    gauss_filter = make_gaussian_filter(lshort, truncate, device)
    result = []
    for frame_start, batch in minibatch(video):
        padder = (*(np.array(llong) // 2),)*2
        padded_video = torch.nn.functional.pad(batch, padder, "replicate")
        background = torch.nn.functional.conv2d(padded_video, filter)
        batch_result = torch.nn.functional.conv2d(batch, gauss_filter, padding = "same")
        batch_result -= background
        batch_result = torch.where(batch_result >= threshold, batch_result, 0)
        # batch_result = batch_result.squeeze()
        result.append(batch_result)
    result = torch.concat(result)
    result = result.squeeze(dim = 1)
    return result

def make_gaussian_filter(lshort, truncate, device):
    lws = [int(truncate * lsh + 0.5) for lsh in lshort]
    grids = [torch.moveaxis(torch.arange(-lw, lw + 1, device = device)[(slice(None), ) + (None,)*(len(lshort)-1)], 0, i) for i, lw in enumerate(lws)]
    r2 = sum([grid**2 for grid in grids])
    #TODO skewed gaussian support
    res = torch.exp(r2/(-2*lshort[0]**2))
    res /= res.sum()
    res = res.reshape(1, 1, *res.shape)
    return res

def convert_to_int(video):
    vid_max = video.max()
    factor = 255 / vid_max
    output = (video * vid_max).to(torch.uint8)
    return factor, output

def grey_dilation(video, separation, percentile=64, margin=None):
    ndim = video.ndim - 1
    # video = video.reshape(video.shape[0], 1, *video.shape[1:])
    video = torch.unsqueeze(video, 1)
    
    device = video.device

    separation = validate_tuple(separation, ndim)
    if margin is None:
        margin = tuple([int(s / 2) for s in separation])

    # Compute a threshold based on percentile.
    

    # Find the largest box that fits inside the ellipse given by separation
    size = [int(2 * s / np.sqrt(ndim)) for s in separation]
    candidates = []

    if device.type == "mps":
        mean = video.mean(axis = 0).ravel()
        mean = mean.to("cpu").numpy()
        
        threshold = torch.tensor(np.percentile(mean, percentile, overwrite_input = True).astype(np.float32),
             device = device)
        
    for frame_start, batch in minibatch(video):
        nframes = batch.shape[0]
        if device.type != "mps":
            threshold = batch.reshape(nframes, -1).quantile(percentile / 100, dim = -1).reshape([-1] + [1]*(ndim))
        padding = (*(np.array(size) // 2).astype(int),)
        dilation = torch.nn.functional.max_pool2d(batch, size, stride = 1, padding = padding)
        dilation = torch.squeeze(dilation, dim = 1)
        batch = torch.squeeze(batch, dim = 1)
        if dilation.shape != batch.shape:
            dilation = dilation[(slice(None), ) + (slice(1, None), )*ndim]

        maxima = (batch == dilation) & (batch > threshold)
        if device.type == "mps":
            np_out = np.stack(maxima.cpu().numpy().nonzero(), axis = 1)
            out = torch.tensor(np_out, device = device)
        else:
            out = torch.nonzero(maxima)
        out[:, 0] += frame_start
        candidates.append(out)
    candidates = torch.concat(candidates)

    if torch.sum(maxima) == 0:
        warnings.warn("video contains no local maxima.", UserWarning)
        return np.empty((0, ndim))

    pos = candidates[:, 1:]
    # Do not accept peaks near the edges.
    shape = torch.tensor(video.shape[2:], device = device)
    near_edge = ((pos < margin) | (pos > (shape - margin - 1))).any(axis = 1)
    pos = pos[~near_edge]
    # exclude_channel = torch.tensor([True, False] + [True]*ndim)
    candidates = candidates[~near_edge]#[:, exclude_channel]
    if len(pos) == 0:
        warnings.warn("All local maxima were in the margins.", UserWarning)
        return np.empty((0, ndim))

    # Remove local maxima that are too close to each other

    return candidates



def _refine(raw_video, video, radius, candidates, max_iterations,
            shift_thresh, minmass = None):
    
    ndim = video.ndim - 1
    N = candidates.shape[0]
    device = video.device
    radii = torch.tensor(radius, device = device)
    shift_thresh = torch.tensor(shift_thresh, device = device)
    zero_centered_grids = [torch.moveaxis(torch.arange(-r, r + 1, device = device)[(slice(None), ) + (None,)*(ndim - 1)], 0, i) for i, r in enumerate(radii)]
    r2 = sum([grid**2/r**2 for grid, r in zip(zero_centered_grids, radii)])
    mask = r2 <= 1
    
    # grids = [torch.moveaxis(
    # torch.arange(-radius[dim], radius[dim] + 1, device = device)[:, None], 0, dim) 
            # for dim in range(ndim)]
    upper_bound = torch.tensor(video.shape[1:], device = device) - 1 - radii

    nan = torch.tensor(float("nan"), device = device)
    
    # Columns:
    # {0: "y", 1: "x", 2: "mass", 3: "signal"}
    results = torch.full((N, 4), nan, dtype = torch.float32, device = device)


    make_inds_for_dim = lambda dim, pos: zero_centered_grids[dim] + pos[:, dim].reshape([-1] + [1]*ndim)

    for i, (index_offset, batch_candidates) in enumerate(minibatch(candidates)):

        pos = batch_candidates[:, 1:]
        frames = batch_candidates[:, 0].reshape(-1, 1, 1)

        not_done = torch.arange(index_offset, index_offset + pos.shape[0], device = device)

        for _ in range(max_iterations):
            neighborhoods = video[frames, make_inds_for_dim(0, pos), make_inds_for_dim(1, pos)]
            neighborhoods = torch.multiply(neighborhoods, mask, out = neighborhoods)
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

    frames = candidates[:, 0].reshape(-1, 1, 1)
    if minmass is not None:
        pass

    # final_pixels = torch.round(final_coords).to(int)
    # raw_masses = (raw_video[frames, make_inds_for_dim(0, final_pixels), make_inds_for_dim(1, final_pixels)] * mask).sum(axis = (1,2))

    return torch.column_stack((torch.squeeze(frames), results))

def fast_signal_extraction(positions, frames, r, padded_video, params, return_means = False):
    # if backend is None:
    #     try:
    #         import torch
            
    #         backend = torch
    #     except ModuleNotFoundError:
    #         try:
    #             import bottleneck
    #             backend = bottleneck
    #         except ModuleNotFoundError:
    #             backend = np
        
    # if backend.__name__ == "torch":
    #     using_torch = True
    #     ap = backend
    #     if device is None:
    #         device = "cuda" if ap.cuda.is_available() else "cpu"
    #     if not isinstance(padded_video, ap.Tensor):
    #         padded_video = ap.tensor(padded_video).to(device)
    # else:
    #     using_torch = False
    #     ap = np
    device = padded_video.device
    sub_pix = positions % 1
    pix = positions.to(int)
    frames = frames.to(int)

    r_ceil = np.ceil(r).astype(int)
    difference_for_dimension = lambda dim: torch.moveaxis(torch.arange(
        -r_ceil + 1, r_ceil + 1, device = device) - sub_pix[:, dim].reshape(-1, 1, 1), -1, dim + 1)
    xs = difference_for_dimension(0)
    ys = difference_for_dimension(1)
    # if using_torch:
    r2 = xs**2 + ys**2
    inds_for_dimension = lambda dim: torch.moveaxis((
        torch.arange(-r_ceil + 1, r_ceil + 1, device = device)) + pix[:, dim].reshape(-1, 1, 1), -1, dim+1).clip(
            -1, padded_video.shape[dim + 1] - 1)
    
    xinds = inds_for_dimension(0)
    yinds = inds_for_dimension(1)

    rectangles = padded_video[frames[:, None, None], xinds, yinds]

    sigmasks = r2 <= params.lip_int_size
    if params.gap_size != 0:
        bg_inner = r2 <= params.lip_int_size + params.gap_size
    else:
        bg_inner = sigmasks

    bgmasks = torch.logical_xor(r2 <= r**2, bg_inner)
    
    # if using_torch:
    nan = torch.tensor(float("nan"), dtype = torch.float64).to(device)
    # else:
        # nan = ap.nan
    
    signals = torch.where(sigmasks, rectangles, nan).reshape(positions.shape[0], -1)
    backgrounds = torch.where(bgmasks, rectangles, nan).reshape(positions.shape[0], -1)

    sig_sums = torch.nansum(signals, axis = 1)
    bg_medians = torch.nanmedian(backgrounds, axis = 1)
    sig_sizes = torch.nansum(sigmasks.reshape(positions.shape[0], -1), axis = 1)
    out = [sig_sums, bg_medians, sig_sizes]
    if return_means:
        sig_means = sig_sums / sig_sizes
        # if using_torch:
        bg_sums = torch.nansum(backgrounds, axis = 1)
        bg_sizes = torch.nansum(bgmasks.reshape(positions.shape[0], -1), axis = 1)
        bg_means = bg_sums / bg_sizes
        # else:
        #     bg_means = backend.nansum(backgrounds, axis = 1)
        out.extend([sig_means, bg_means])
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