from .gpu_tracking import batch_rust 
from .gpu_tracking import batch_file_rust
from .gpu_tracking import link_rust
import pandas as pd

def batch(
    video,
    diameter,
    minmass = None,
    maxsize = None,
    separation = None,
    noise_size = None,
    smoothing_size = None,
    threshold = None,
    invert = None,
    percentile = None,
    topn = None,
    preprocess = None,
    max_iterations = None,
    characterize = None,
    filter_close = None,
    search_range = None,
    memory = None,
    sig_radius = None,
    bg_radius = None,
    gap_radius = None,
    points_to_characterize = None
    ):

    if isinstance(points_to_characterize, pd.DataFrame):
        points_to_characterize = points_to_characterize[["frame", "y", "x"]].to_numpy()
    
    if points_to_characterize is not None:
        points_to_characterize = points_to_characterize.astype("float32")

    arr, columns = batch_rust(
        video,
        diameter,
        minmass,
        maxsize,
        separation,
        noise_size,
        smoothing_size,
        threshold,
        invert,
        percentile,
        topn,
        preprocess,
        max_iterations,
        characterize,
        filter_close,
        search_range,
        memory,
        sig_radius,
        bg_radius,
        gap_radius,
        points_to_characterize,
    )
    columns = {name: typ for name, typ in columns}
    return pd.DataFrame(arr, columns = columns).astype(columns)

def batch_file(
    path,
    diameter,
    channel = None,
    minmass = None,
    maxsize = None,
    separation = None,
    noise_size = None,
    smoothing_size = None,
    threshold = None,
    invert = None,
    percentile = None,
    topn = None,
    preprocess = None,
    max_iterations = None,
    characterize = None,
    filter_close = None,
    search_range = None,
    memory = None,
    sig_radius = None,
    bg_radius = None,
    gap_radius = None,
    points_to_characterize = None
    ):
    if isinstance(points_to_characterize, pd.DataFrame):
        points_to_characterize = points_to_characterize[["frame", "y", "x"]].to_numpy()
    
    if points_to_characterize is not None:
        points_to_characterize = points_to_characterize.astype("float32")

    arr, columns = batch_file_rust(
        path,
        diameter,
        channel,
        minmass,
        maxsize,
        separation,
        noise_size,
        smoothing_size,
        threshold,
        invert,
        percentile,
        topn,
        preprocess,
        max_iterations,
        characterize,
        filter_close,
        search_range,
        memory,
        sig_radius,
        bg_radius,
        gap_radius,
        points_to_characterize,
    )
    
    columns = {name: typ for name, typ in columns}
    return pd.DataFrame(arr, columns = columns).astype(columns)

def link(to_link, search_range, memory):
    if isinstance(to_link, pd.DataFrame):
        to_link = to_link[["frame", "y", "x"]].to_numpy()

    result = link_rust(to_link, search_range, memory)

    if isinstance(to_link, pd.DataFrame):
        output = to_link.copy()
        output["particle"] = result
    else:
        output = result

    return output

# def characterize_points(
#     points,
#     video,
#     diameter,
#     channel = None,
#     minmass = None,
#     maxsize = None,
#     separation = None,
#     noise_size = None,
#     smoothing_size = None,
#     threshold = None,
#     invert = None,
#     percentile = None,
#     topn = None,
#     preprocess = None,
#     max_iterations = None,
#     characterize = None,
#     filter_close = None,
#     search_range = None,
#     memory = None,
#     sig_radius = None,
#     bg_radius = None,
#     gap_radius = None,
#     ):

#     arr, columns = batch_file_rust(
#         points,
#         video,
#         diameter,
#         channel,
#         minmass,
#         maxsize,
#         separation,
#         noise_size,
#         smoothing_size,
#         threshold,
#         invert,
#         percentile,
#         topn,
#         preprocess,
#         max_iterations,
#         characterize,
#         filter_close,
#         search_range,
#         memory,
#         sig_radius,
#         bg_radius,
#         gap_radius,
#     )
#     columns = {name: typ for name, typ in columns}
#     return pd.DataFrame(arr, columns = columns).astype(columns)
