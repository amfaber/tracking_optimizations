import numpy as np
from scipy import ndimage as ndi
from trackpy.preprocessing import bandpass, invert_image


def binary_mask(radius, ndim):
    "Elliptical mask in a rectangular array"
    radius = validate_tuple(radius, ndim)
    points = [np.arange(-rad, rad + 1) for rad in radius]
    if len(radius) > 1:
        coords = np.array(np.meshgrid(*points, indexing="ij"))
    else:
        coords = np.array([points[0]])
    r = [(coord / rad) ** 2 for (coord, rad) in zip(coords, radius)]
    return sum(r) <= 1


def validate_tuple(value, ndim):
    if not hasattr(value, "__iter__"):
        return (value,) * ndim
    if len(value) == ndim:
        return tuple(value)
    raise ValueError("List length should have same length as image dimensions.")


def trackpy_preproccessing(
    raw_image,
    diameter,
    noise_size,
    smoothing_size=None,
    threshold=None,
    invert=False,
    maxsize=None,
    separation=None,
    minmass=None,
):
    preprocess = True
    raw_image = np.squeeze(raw_image)
    shape = raw_image.shape
    ndim = len(shape)

    diameter = validate_tuple(diameter, ndim)
    diameter = tuple([int(x) for x in diameter])
    if not np.all([x & 1 for x in diameter]):
        raise ValueError("Feature diameter must be an odd integer. Round up.")
    radius = tuple([x // 2 for x in diameter])

    isotropic = np.all(radius[1:] == radius[:-1])
    if (not isotropic) and (maxsize is not None):
        raise ValueError(
            "Filtering by size is not available for anisotropic " "features."
        )

    is_float_image = not np.issubdtype(raw_image.dtype, np.integer)

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

    # Check whether the image looks suspiciously like a color image.
    if 3 in shape or 4 in shape:
        dim = raw_image.ndim
        warnings.warn(
            "I am interpreting the image as {}-dimensional. "
            "If it is actually a {}-dimensional color image, "
            "convert it to grayscale first.".format(dim, dim - 1)
        )

    if threshold is None:
        if is_float_image:
            threshold = 1 / 255.0
        else:
            threshold = 1

    # Invert the image if necessary
    if invert:
        raw_image = invert_image(raw_image)

    # Determine `image`: the image to find the local maxima on.
    if preprocess:
        image = bandpass(raw_image, noise_size, smoothing_size, threshold)
    else:
        image = raw_image

    return image


def calc_minmass(
    first_frame,
    particles_per_frame,
    snr,
    diameter,
    preprocess=False,
    noise_size=1,
    smoothing_size=None,
    threshold=None,
):
    if preprocess:
        first_frame = trackpy_preproccessing(
            first_frame, diameter, noise_size, smoothing_size, threshold
        )
    dimensions = first_frame.shape[0:2]
    area = dimensions[0] * dimensions[1]
    # calculate percentile based on estimated particles per frame
    percentile = (1 - (particles_per_frame / area)) * 100
    video = first_frame
    signal = np.percentile(video, percentile)
    video_wo_signal = video[video < signal]
    video_signal = video[video > signal]
    signal_positions = np.where(video > signal)
    noisevid = np.std(video)
    noise = np.std(video_wo_signal)
    meanvideo_wo_signal = np.mean(video_wo_signal)
    # print(
    #     "Noise after signal removal: ", noise, "Noise before signal removal: ", noisevid
    # )
    image = video
    ndim = image.ndim
    diameter = validate_tuple(diameter, ndim)
    diameter = tuple([int(x) for x in diameter])
    radius = tuple([x // 2 for x in diameter])
    mask = binary_mask(radius, ndim).astype(np.uint8)
    coords = np.array(signal_positions).T
    N = coords.shape[0]
    final_coords = np.empty_like(coords, dtype=np.float64)
    mass = np.empty(N, dtype=np.float64)
    mean = np.empty(N, dtype=np.float64)
    peak_signal = np.empty(N, dtype=np.float64)
    mass_to_signal = np.empty(N, dtype=np.float64)
    mean_ratio = np.empty(N, dtype=np.float64)
    confirmed_signals = []
    for feat, coord in enumerate(coords):
        try:
            # Define the circular neighborhood of (x, y).
            rect = tuple([slice(c - r, c + r + 1) for c, r in zip(coord, radius)])
            neighborhood = mask * image[rect]
            mass[feat] = neighborhood.sum()
            # print(neighborhood)
            neighborhood_area = np.size(neighborhood)
            mean[feat] = neighborhood[neighborhood != 0].mean()
            center = ndi.center_of_mass(neighborhood)
            peak_signal[feat] = neighborhood.max()
            mean_ratio[feat] = mean[feat] / meanvideo_wo_signal

            if (
                peak_signal[feat] / noise > snr
                and mean[feat] > 1.0 * np.mean(video_wo_signal)
                and mean[feat] / noise > snr
            ):
                confirmed_signals.append(mass[feat])
        except ValueError:
            print("Feature {} is outside image by diameter.".format(feat))
            pass

    if len(confirmed_signals) == 0:
        raise ValueError("No particles found")
    return np.min(confirmed_signals), np.array(signal_positions), image
