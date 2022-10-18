from .gpu_tracking import batch as _batch
from .gpu_tracking import link
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
    cpu_processed = None,
    sig_radius = None,
    bg_radius = None,
    gap_radius = None,
    ):

    arr, columns = _batch(
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
        cpu_processed,
        sig_radius,
        bg_radius,
        gap_radius,
    )
    columns = {name: typ for name, typ in columns}
    return pd.DataFrame(arr, columns = columns).astype(columns)