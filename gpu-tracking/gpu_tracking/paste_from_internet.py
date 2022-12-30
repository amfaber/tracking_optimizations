
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Import Modules

# +
import os
import sys

import pandas as pd

import plotly.graph_objs as go

# #############################################################################
# from local_methods import get_sliders_init_dict, get_slider_step_i, get_layout
# -

# # Methods

# +
# import plotly.graph_objs as go


def get_layout(
    duration_long=1000 * 2,
    duration_short=800 * 2,
    ):
    """
    """
    # | - get_layout

    # | - updatemenus
    updatemenus = [
        {
            'buttons': [
                {
                    'args': [
                        None,
                        {
                            'frame': {
                                'duration': duration_long,  # TEMP_DURATION
                                'redraw': False,
                                },
                            'fromcurrent': True,
                            'transition': {
                                'duration': duration_short,  # TEMP_DURATION
                                'easing': 'quadratic-in-out',
                                }
                            }
                        ],
                    'label': 'Play',
                    'method': 'animate'
                    },
                {
                    'args': [
                        [None],
                        {
                            'frame': {
                                'duration': 0,
                                'redraw': False,
                                },
                            'mode': 'immediate',
                            'transition': {'duration': 0},
                            }
                        ],
                    'label': 'Pause',
                    'method': 'animate'
                    }
                ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
            }
        ]

    #__|

    layout = go.Layout(
        # title='Material Discovery Training',
        showlegend=False,
        font=dict(
            # family='Courier New, monospace',

            family='Arial',
            size=20,
            color='black'
            ),

        xaxis={
            'title': 'X-axis Title',

            # 'range': [0 - 5, len(models_list[0]) + 5],
            # 'autorange': False,
            'autorange': True,

            'showgrid': False,
            'zeroline': False,
            'showline': True,
            'ticks': '',
            'showticklabels': True,
            'mirror': True,
            'linecolor': 'black',

            },

        yaxis={
            'title': 'Y-axis Title',

            # 'range': [global_y_min, global_y_max],
            # 'range': [-1.5, 2.4],
            # 'autorange': False,
            'autorange': True,
            'fixedrange': False,

            'showgrid': False,
            'zeroline': True,
            'showline': True,
            'ticks': '',
            'showticklabels': True,
            'mirror': True,
            'linecolor': 'black',

            },

        updatemenus=updatemenus,
        )

    return(layout)
    #__|

    
def get_sliders_init_dict(duration_short):
    """
    """
    # | - get_sliders_init_dict
    sliders_dict = {
        # 'active': 0,
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Loop #:',
            'visible': True,
            'xanchor': 'right'
            },
        'transition': {
            'duration': duration_short,  # TEMP_DURATION
            'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.8,
        'x': 0.1,
        'y': 0,
        'steps': [],
        }

    return(sliders_dict)
    #__|


def get_slider_step_i(i_cnt, duration_short):
    """
    """
    #| - get_slider_step_i
    slider_step_i = {
        'args': [
            [str(i_cnt)],
            {
                'frame': {
                    'duration': duration_short,  # TEMP_DURATION
                    'redraw': False},
                'mode': 'immediate',
                'transition': {
                    'duration': duration_short,  # TEMP_DURATION
                    },
                },
            ],
        'label': str(i_cnt),
        'method': 'animate',
        }

    return(slider_step_i)
    #__|



# -

# # Script Inputs

duration_long = 1000 * 3
duration_short = 800 * 3

# # Read Data

# +
dict_data = {
    ('0', 'name'): {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    ('0', 'x'): {0: 1, 1: 2, 2: 3, 3: 4},
    ('0', 'y'): {0: 1, 1: 2, 2: 3, 3: 4},
    ('0', 'color'): {0: 'red', 1: 'blue', 2: 'green', 3: 'black'},
    ('1', 'name'): {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    ('1', 'x'): {0: 2, 1: 1, 2: 3, 3: 4},
    ('1', 'y'): {0: 2, 1: 1, 2: 3, 3: 4},
    ('1', 'color'): {0: 'red', 1: 'blue', 2: 'green', 3: 'black'},
    ('2', 'name'): {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    ('2', 'x'): {0: 4, 1: 3, 2: 1, 3: 2},
    ('2', 'y'): {0: 4.0, 1: 3.0, 2: 1.0, 3: 0.8},
    ('2', 'color'): {0: 'red', 1: 'blue', 2: 'green', 3: 'black'},
    ('3', 'name'): {0: 'a', 1: 'b', 2: 'd', 3: 'c'},
    ('3', 'x'): {0: 4, 1: 3, 2: 1, 3: -1},
    ('3', 'y'): {0: 4.0, 1: 3.0, 2: 0.8, 3: 1.0},
    ('3', 'color'): {0: 'red', 1: 'blue', 2: 'green', 3: 'black'},
    }


df_data = pd.DataFrame(dict_data)

# +
# df_data = pd.read_csv("data.csv", header=[0, 1])

time_series_indices = list(df_data.columns.levels[0])

df_list = []
for i in time_series_indices:
    df_i = df_data[i]
    df_i = df_i.set_index("name")
    # df_i = df_i.sort_index()
    df_list.append(df_i)


# -

# # methods | get_traces

def get_traces(df):
    trace = go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers",
        marker=dict(
            symbol="circle",
            color=df["color"],
            size=18),
        )
    data = [trace]

    return(data)


# +
layout_anim = get_layout(
    duration_long=duration_long,
    duration_short=duration_short)
sliders_dict = get_sliders_init_dict(duration_short)

frames = []; data = []
for i_cnt, df_i in enumerate(df_list):
    traces_i = get_traces(df_i)


    # #####################################################
    # i_cnt = 0
    layout_i = go.Layout(

        annotations=[
            go.layout.Annotation(
                x=1,
                y=1,
                xref="x",
                yref="y",
                text="TEMP_" + str(i_cnt).zfill(3),
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-40
                )
            ],

        )
    # #####################################################



    # #####################################################
    if i_cnt == 0:
        data.extend(traces_i)
    data_i = []
    data_i.extend(traces_i)
    # frame_i = go.Frame(data=data_i, name=str(i_cnt))
    frame_i = go.Frame(data=data_i, name=str(i_cnt), layout=layout_i)
    frames.append(frame_i)
    slider_step_i = get_slider_step_i(i_cnt, duration_short)
    sliders_dict['steps'].append(slider_step_i)

# +
layout_anim["showlegend"] = True

fig = go.Figure(
    data=data,
    layout=layout_anim,
    frames=frames)
fig['layout']['sliders'] = [sliders_dict]

fig.show()