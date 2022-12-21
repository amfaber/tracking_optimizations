# from .gpu_tracking import batch_rust 
# from .gpu_tracking import batch_file_rust
# from .gpu_tracking import link_rust
# from .gpu_tracking import batch_log
# from .gpu_tracking import batch_file_log
from .gpu_tracking import *
import pandas as pd

def batch(
    video,
    diameter,
    **kwargs
    ):
    try:
        points_to_characterize = kwargs["points_to_characterize"]
        if isinstance(points_to_characterize, pd.DataFrame):
            points_to_characterize = points_to_characterize[["frame", "y", "x"]].to_numpy()
        kwargs["points_to_characterize"] = points_to_characterize.astype("float32")

    except KeyError:
        pass
    arr, columns = batch_rust(
        video,
        diameter,
        **kwargs
    )
    columns = {name: typ for name, typ in columns}
    return pd.DataFrame(arr, columns = columns).astype(columns)


def batch_file(
    path,
    diameter,
    **kwargs
    ):
    try:
        points_to_characterize = kwargs["points_to_characterize"]
        if isinstance(points_to_characterize, pd.DataFrame):
            points_to_characterize = points_to_characterize[["frame", "y", "x"]].to_numpy()
        kwargs["points_to_characterize"] = points_to_characterize.astype("float32")
    except KeyError:
        pass
    
    arr, columns = batch_file_rust(
        path,
        diameter,
        **kwargs
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

def LoG(video, min_r, max_r, **kwargs):
    arr, columns = batch_log(video, min_radius = min_r, max_radius = max_r, **kwargs)

    columns = {name: typ for name, typ in columns}
    return pd.DataFrame(arr, columns = columns).astype(columns)

def LoG_file(path, min_r, max_r, **kwargs):
    arr, columns = batch_file_log(path, min_radius = min_r, max_radius = max_r, **kwargs)

    columns = {name: typ for name, typ in columns}
    return pd.DataFrame(arr, columns = columns).astype(columns)

def load(path, ets_channel = 0, keys = None):
    extension = path.split(".")[1].lower()
    if extension == "tif" | extension == "tiff":
        import tifffile
        video = tifffile.imread(path, keys)
    elif extension == "ets":
        if keys is not None:
            keys = list(keys)
            video = parse_ets_with_keys(path, keys, ets_channel)
        else:
            video = parse_ets(path)[ets_channel]
    else:
        raise ValueError("Unrecognized file format. Recognized formats: tiff, ets")
    return video

def annotate(video, tracked_df, figax = None, r = None, frame = 0, imshow_kw = {}, circle_kw = {}, subplot_kw = {}):
    import matplotlib.pyplot as plt

    circle_kw = {"fill": False, **circle_kw}

    subset_df = tracked_df[tracked_df["frame"] == frame]
    
    if r is None and not "r" in subset_df:
        r = 5
        print(f"Using default r of {r}")
    else:
        r = None
    if figax is None:
        fig, ax = plt.subplots(**subplot_kw)
    else:
        fig, ax = figax
    ax.imshow(video[frame], **imshow_kw)

    for row in tracked_df.iterrows():
        if r is None:
            inner_r = row["r"]
        else:
            inner_r = r
        x, y = row["x"], row["y"]
        ax.add_patch(plt.Circle((x, y), inner_r, **circle_kw))
    return (fig, ax)
        