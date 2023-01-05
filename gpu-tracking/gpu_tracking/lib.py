from gpu_tracking import *
import pandas as pd
import numpy as np
import uuid

def batch(
    video_or_path,
    diameter,
    **kwargs
    ):
    if isinstance(video_or_path, np.ndarray):
        video = video_or_path
        if video.ndim == 2:
            video = video[None, ...]    
        arr, columns = batch_rust(
            video,
            diameter,
            **kwargs
        )
    else:
        path = str(video_or_path)
        arr, columns = batch_file_rust(
            path,
            diameter,
            **kwargs
        )

    columns = {name: typ for name, typ in columns}
    return pd.DataFrame(arr, columns = columns).astype(columns)

def characterize_points(
    video_or_path,
    points_to_characterize,
    diameter = None,
    has_frames = None,
    has_r = None,
    **kwargs,
):
    if isinstance(points_to_characterize, pd.DataFrame):
        cols = ["y", "x"]
        if "frame" in points_to_characterize.columns:
            cols = ["frame"] + cols
            if has_frames is None:
                has_frames = True
        if "r" in points_to_characterize.columns:
            cols = cols + ["r"]
            if has_r is None:
                has_r = True

        if has_frames is None:
            has_frames = False
        if has_r is None:
            has_r = False
        points_arr = points_to_characterize[cols].to_numpy().astype("float32")

    if diameter is None:
        if has_r:
            diameter = 2*int(points_arr[:, -1].max() + 0.5) + 1
        else:
            raise ValueError("Diameter needs to be specified if the supplied points don't have associated radiuses")
            
    if isinstance(video_or_path, np.ndarray):
        video = video_or_path
        if video.ndim == 2:
            video = video[None, ...]
        arr, columns = characterize_rust(
            video,
            points_arr,
            has_frames,
            has_r,
            diameter,
            **kwargs
        )
    else:
        path = str(video_or_path)
        arr, columns = characterize_file_rust(
            path,
            points_arr,
            has_frames,
            has_r,
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

def LoG(video_or_path, min_r, max_r, **kwargs):
    if isinstance(video_or_path, np.ndarray):
        video = video_or_path
        if video.ndim == 2:
            video = video[None, ...]
        arr, columns = log_rust(
            video,
            min_r,
            max_r,
            **kwargs
        )
    else:
        path = str(video_or_path)
        arr, columns = log_file_rust(
            path,
            min_r,
            max_r,
            **kwargs
        )

    columns = {name: typ for name, typ in columns}
    return pd.DataFrame(arr, columns = columns).astype(columns)

# def load(path, ets_channel = 0, keys = None):
#     extension = path.split(".")[1].lower()
#     if extension == "tif" or extension == "tiff":
#         import tifffile
#         video = tifffile.imread(path, key = keys)
#     elif extension == "ets":
#         if keys is not None:
#             keys = list(keys)
#             video = parse_ets_with_keys(path, keys, ets_channel)
#         else:
#             video = parse_ets(path)[ets_channel]
#     else:
#         raise ValueError("Unrecognized file format. Recognized formats: tiff, ets")
#     return video

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

def annotate_image_plotly(image, tracked_df, r = None, frame = None, imshow_kw = {}, circle_color = "white", color_scale = "viridis", circle_kw = {}):
    from plotly import express as px
    
    if frame is not None:
        subset_df = tracked_df[tracked_df["frame"] == frame]
    else:
        subset_df = tracked_df
    
    if r is None and "r" not in subset_df:
        r = 5
        print(f"Using default r of {r}")
    fig = px.imshow(image, color_continuous_scale = color_scale, **imshow_kw)

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
            line_color = circle_color, line_width = 1, **circle_kw
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
    
