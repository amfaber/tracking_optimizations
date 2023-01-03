from gpu_tracking import *
import pandas as pd
import uuid

def batch(
    video,
    diameter,
    **kwargs
    ):
    try:
        points_to_characterize = kwargs["points_to_characterize"]
        if isinstance(points_to_characterize, pd.DataFrame):
            if "frame" in points_to_characterize.columns:
                points_to_characterize = points_to_characterize[["frame", "y", "x"]].to_numpy()
            else:
                points_to_characterize = points_to_characterize[["y", "x"]].to_numpy()
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
            if "frame" in points_to_characterize.columns:
                points_to_characterize = points_to_characterize[["frame", "y", "x"]].to_numpy()
            else:
                points_to_characterize = points_to_characterize[["y", "x"]].to_numpy()
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
    if extension == "tif" or extension == "tiff":
        import tifffile
        video = tifffile.imread(path, key = keys)
    elif extension == "ets":
        if keys is not None:
            keys = list(keys)
            video = parse_ets_with_keys(path, keys, ets_channel)
        else:
            video = parse_ets(path)[ets_channel]
    else:
        raise ValueError("Unrecognized file format. Recognized formats: tiff, ets")
    return video

def annotate_image(image, tracked_df, figax = None, r = None, frame = None, imshow_kw = {}, circle_kw = {}, subplot_kw = {}):
    import matplotlib.pyplot as plt

    circle_kw = {"fill": False, **circle_kw}
    if frame is not None:
        subset_df = tracked_df[tracked_df["frame"] == frame]
    else:
        subset_df = tracked_df
    
    if r is None and not "r" in subset_df:
        r = 5
        print(f"Using default r of {r}")
    if figax is None:
        fig, ax = plt.subplots(**subplot_kw)
    else:
        fig, ax = figax
    ax.imshow(image, **imshow_kw)

    for _idx, row in subset_df.iterrows():
        if r is None:
            inner_r = row["r"]
        else:
            inner_r = r
        x, y = row["x"], row["y"]
        ax.add_patch(plt.Circle((x, y), inner_r, **circle_kw))
    return (fig, ax)

def annotate_image_plotly(image, tracked_df, figax = None, r = None, frame = None, imshow_kw = {}, circle_kw = {}, subplot_kw = {}):
    from plotly import express as px
    
    if frame is not None:
        subset_df = tracked_df[tracked_df["frame"] == frame]
    else:
        subset_df = tracked_df
    
    if r is None and "r" not in subset_df:
        r = 5
        print(f"Using default r of {r}")
    fig = px.imshow(image, color_continuous_scale = "viridis", **imshow_kw)

    for _idx, row in subset_df.iterrows():
        if r is None:
            inner_r = row["r"]
        else:
            inner_r = r
        x, y = row["x"], row["y"]
        fig.add_shape(
            type = "circle", xref = "x", yref = "y",
            x0 = x - inner_r, y0 = y - inner_r,
            x1 = x + inner_r, y1 = y + inner_r,
            line_color = "black", line_width = 1,
        )
    return fig

def annotate_video(video, tracked_df, frame = 0, **kwargs):
    image = video[frame]
    return annotate_image(image, tracked_df, frame = frame, **kwargs)

def annotate_file(path, tracked_df, ets_channel = 0, frame = 0, **kwargs):
    image = load(path, ets_channel = ets_channel, keys = [frame])[0]
    return annotate_image(image, tracked_df, frame = frame, **kwargs)

def unique_particle_ids(df, column = "particle"):
    id = uuid.uuid4()
    df[column] = df[column].as_type("str") + id
    
