from .gpu_tracking import batch_rust 
from .gpu_tracking import batch_file_rust
from .gpu_tracking import link_rust
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
