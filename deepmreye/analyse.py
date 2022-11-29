import os
import numpy as np
import pandas as pd
import warnings

import plotly.graph_objs as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------------
# --------------------------VISUALIZATIONS----------------------------------------
# --------------------------------------------------------------------------------

def visualise_input_data(X, y, color="rgb(0, 150, 175)", cut_at=151, bg_color="rgb(247,247,247)", ylim=[-6, 6], num_functionals=78):
    # Prepare data for plotting
    # For visualisation we use a downsampled and padded version of X and split the eye balls.
    X_right = np.pad(X[:, 0:25, ..., 0], ((0, 0), (0, 5), (1, 2), (0, 0)))
    X_left = np.pad(X[:, 25::, ..., 0], ((0, 0), (6, 2), (1, 2), (0, 0)))
    X_concat = np.concatenate((X_left, X_right), axis=1)
    X_vis = X_concat[0:cut_at, ...]

    # Output can contain NaN which is taken care of during model training. We just remove NaN here for visualization purposes
    vis_x = pd.Series(np.median(y[0:cut_at, ..., 0], axis=1)).ffill().values
    vis_y = pd.Series(np.median(y[0:cut_at, ..., 1], axis=1)).ffill().values

    # Create interactive plot to visualize example participant
    fig = make_subplots(rows=2, cols=4, horizontal_spacing=0.01, vertical_spacing=0.15,
                        specs=[[{"rowspan": 2, "colspan": 2}, None, {"colspan": 2}, None], [None, None, {"colspan": 2}, None]])
    fig.add_trace(go.Scatter(x=np.arange(0, len(vis_x)), y=vis_x, mode='lines', line_color='rgb(0,0,0)', opacity=0.5, line_width=3), row=1, col=3)
    fig.add_trace(go.Scatter(x=np.arange(0, len(vis_y)), y=vis_y, mode='lines', line_color='rgb(0,0,0)', opacity=0.5, line_width=3), row=2, col=3)

    # Plot input signal together with split output signal (X & Y)
    for i in range(0, X_vis.shape[0]):
        fig.add_trace(go.Scatter(x=[i], y=[vis_x[i]], mode='markers', marker=dict(size=12, color=color, line=dict(width=2, color='DarkSlateGrey')), visible=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=[i], y=[vis_y[i]], mode='markers', marker=dict(size=12, color=color, line=dict(width=2, color='DarkSlateGrey')), visible=False), row=2, col=3)
        this_z = np.mean(X_vis[i, ..., :], axis=-1).transpose()
        if bg_color == "rgb(247,247,247)":
            colorscale = 'RdBu'
        elif bg_color == "rgb(255,255,255)":
            colorscale = 'RdGy'
        fig.add_heatmap(z=this_z, colorscale=colorscale, visible=False, showscale=False, zmid=0, row=1, col=1, name="TR: {}".format(i))  # colorbar=dict(x=0.45, y=0.5, thickness=10, len=0.5))
    fig.data[num_functionals*3+2].visible = True  # Some arithmetic;
    fig.data[num_functionals*3+2+1].visible = True  # connects to the active value below;
    fig.data[num_functionals*3+2+2].visible = True  # should be abstracted as a variable;

    # Add slider for changing TR
    steps, stepcount = [], 0
    for i in range(2, len(fig.data) - 2, 3):
        step = dict(method="update", label="TR: {}".format(stepcount), args=[{"visible": [True, True] + [False] * (len(fig.data) - 2)}])
        step["args"][0]["visible"][i] = True
        step["args"][0]["visible"][i+1] = True
        step["args"][0]["visible"][i+2] = True
        steps.append(step)
        stepcount += 1
    sliders = [dict(active=num_functionals, currentvalue={"prefix": "TR: ", "visible": False}, pad={"t": 40}, steps=steps)]

    # Add arrows
    annotations = [dict(x=11, y=8, xref="x", yref="y", text="Left eye", font=(dict(size=20)), showarrow=True, arrowhead=5, ax=-40, ay=80),
                   dict(x=46, y=8, xref="x", yref="y", text="Right eye", font=(dict(size=20)), showarrow=True, arrowhead=5, ax=40, ay=80),
                   dict(x=0.18, y=1.08, xref='paper', yref='paper', text="<b>Normalized MR-Signal</b>", font=(dict(size=20)), showarrow=False),
                   dict(x=0.78, y=1.08, xref='paper', yref='paper', text="<b>Gaze position</b>", font=(dict(size=20)), showarrow=False)]

    # Update layout and axes descriptions
    fig.update_layout(sliders=sliders, showlegend=False, margin=dict(t=70, l=20, b=20, r=20), autosize=False, width=1600, height=600,
                      plot_bgcolor=bg_color, paper_bgcolor=bg_color, annotations=annotations)
    fig.update_xaxes(showticklabels=False, row=1, col=1)  # title=dict(text='Eyeball signal', font=dict(size=50), standoff=0))
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(range=ylim, row=1, col=3, ticksuffix='°', title=dict(text='X', standoff=0, font=dict(size=20)))
    fig.update_yaxes(range=ylim, row=2, col=3, ticksuffix='°', title=dict(text='Y', standoff=0, font=dict(size=20)))
    fig.update_xaxes(range=[-2, 150+2], row=1, col=3, title=dict(text='Functional Volume (TR)', standoff=16, font=dict(size=20)))
    fig.update_xaxes(range=[-2, 150+2], row=2, col=3)
    fig.update_layout()

    return fig


def visualise_predictions_click(evaluation, scores, color="rgb(0, 150, 175)", bg_color="rgb(247,247,247)"):
    # Prepare data for plotting
    all_scores = []
    for _, item in scores.items():
        all_scores.append(item.values)
    all_scores = np.array(all_scores)
    to_plot = np.concatenate((all_scores[..., 2], all_scores[..., 5]), axis=0)  # Pearson Mean & R2 Mean
    x = ['Pearson'] * all_scores[..., 2].shape[0] + ['R^2-Score'] * all_scores[..., 5].shape[0]
    participants = list(evaluation.keys())
    participants = participants * 2  # Repeat once for each statistic

    fig = go.FigureWidget(make_subplots(rows=2, cols=4, horizontal_spacing=0.01, vertical_spacing=0.15, shared_xaxes='columns',
                                        specs=[[{"rowspan": 2, "colspan": 2}, None, {"colspan": 2}, None], [None, None, {"colspan": 2}, None]]))

    fig.add_trace(go.Box(y=to_plot[:, 0], marker_size=12, x=x, boxpoints='all', pointpos=0, marker=dict(opacity=0.65, color=color, line=dict(color='rgb(0,0,0)', width=2)),
                         name='Default', line=dict(color='rgb(0,0,0)'), fillcolor="rgb(180, 180, 180)"
                         ), row=1, col=1)
    fig.add_trace(go.Box(y=to_plot[:, 1], marker_size=12, x=x, boxpoints='all', pointpos=0, marker=dict(opacity=0.65, color=color, line=dict(color='rgb(0,0,0)', width=2)),
                         name='Default subTR', line=dict(color='rgb(0,0,0)'), fillcolor="rgb(180, 180, 180)"
                         ), row=1, col=1)
    fig.add_trace(go.Box(y=to_plot[:, 2], marker_size=12, x=x, boxpoints='all', pointpos=0, marker=dict(opacity=0.65, color=color, line=dict(color='rgb(0,0,0)', width=2)),
                         name='Refined', line=dict(color='rgb(0,0,0)'), fillcolor="rgb(180, 180, 180)"
                         ), row=1, col=1)
    fig.add_trace(go.Box(y=to_plot[:, 3], marker_size=12, x=x, boxpoints='all', pointpos=0, marker=dict(opacity=0.65, color=color, line=dict(color='rgb(0,0,0)', width=2)),
                         name='Refined subTR', line=dict(color='rgb(0,0,0)'), fillcolor="rgb(180, 180, 180)"
                         ), row=1, col=1)

    this_real = np.median(evaluation[participants[0]]['real_y'], axis=1)
    this_pred = np.median(evaluation[participants[0]]['pred_y'], axis=1)
    fig.add_trace(go.Scatter(x=np.arange(0, len(this_real[:, 0])), y=this_real[:, 0], mode='lines', line_color='rgb(0,0,0)', opacity=0.5, line_width=3), row=1, col=3)
    fig.add_trace(go.Scatter(x=np.arange(0, len(this_pred[:, 0])), y=this_pred[:, 0], mode='lines', line_color=color, opacity=0.85, line_width=3), row=1, col=3)
    fig.add_trace(go.Scatter(x=np.arange(0, len(this_real[:, 1])), y=this_real[:, 1], mode='lines', line_color='rgb(0,0,0)', opacity=0.5, line_width=3), row=2, col=3)
    fig.add_trace(go.Scatter(x=np.arange(0, len(this_pred[:, 1])), y=this_pred[:, 1], mode='lines', line_color=color, opacity=0.85, line_width=3), row=2, col=3)

    # create our callback function
    def update_point(trace, points, selector):
        if points.point_inds:
            this_participant = participants[points.point_inds[0]]
            this_real = np.median(evaluation[this_participant]['real_y'], axis=1)
            this_pred = np.median(evaluation[this_participant]['pred_y'], axis=1)

            with fig.batch_update():
                all_scatterplots[0].x = np.arange(0, len(this_real[:, 0]))
                all_scatterplots[0].y = this_real[:, 0]
                all_scatterplots[1].x = np.arange(0, len(this_pred[:, 0]))
                all_scatterplots[1].y = this_pred[:, 0]

                all_scatterplots[2].x = np.arange(0, len(this_real[:, 1]))
                all_scatterplots[2].y = this_real[:, 1]
                all_scatterplots[3].x = np.arange(0, len(this_pred[:, 1]))
                all_scatterplots[3].y = this_pred[:, 1]
            fig.update_xaxes(range=[-2, 150+2], row=1, col=3, title=dict(text='Input Volume (TR)', standoff=16, font=dict(size=20)))
            fig.update_xaxes(range=[-2, 150+2], row=2, col=3)
    all_boxplots = fig.data[0:4]
    all_scatterplots = fig.data[4::]
    for bp in all_boxplots:
        bp.on_click(update_point)

    annotations = [dict(x=0.14, y=1.08, xref='paper', yref='paper', text="<b>Model Performance across participants</b>", font=(dict(size=20)), showarrow=False),
                   dict(x=0.855, y=1.08, xref='paper', yref='paper', text="<b>Predicted vs. True gaze position</b>", font=(dict(size=20)), showarrow=False)]

    fig.update_layout(showlegend=False, margin=dict(t=70, l=50, b=50, r=50), plot_bgcolor=bg_color, paper_bgcolor=bg_color,
                      boxmode='group', autosize=False, width=1600, height=650, annotations=annotations)
    fig.update_yaxes(range=[-1, 1], row=1, col=1)
    fig.update_xaxes(tickfont=dict(size=20), row=1, col=1)
    fig.update_yaxes(range=[-6, 6], row=1, col=3, ticksuffix='°', title=dict(text='X', standoff=0, font=dict(size=20)))
    fig.update_yaxes(range=[-6, 6], row=2, col=3, ticksuffix='°', title=dict(text='Y', standoff=0, font=dict(size=20)))
    fig.update_xaxes(range=[-2, 150+2], row=1, col=3, title=dict(text='Functional Volume (TR)', standoff=16, font=dict(size=20)))
    fig.update_xaxes(range=[-2, 150+2], row=2, col=3)

    return fig


def visualise_predictions_slider(evaluation, scores, color="rgb(0, 150, 175)", bg_color="rgb(247,247,247)", line_color="rgb(240,240,240)", ylim=[-6, 6], subTR=False):
    # Prepare data for plotting
    all_scores = []
    for _, item in scores.items():
        all_scores.append(item.values)
    all_scores = np.array(all_scores)
    to_plot = np.concatenate((all_scores[..., 2], all_scores[..., 5]), axis=0)  # Pearson Mean & R2 Mean
    x = ['Pearson'] * all_scores[..., 2].shape[0] + ['R^2-Score'] * all_scores[..., 5].shape[0]
    participants = list(evaluation.keys())
    hover_texts = []
    for subj in participants * 2:
        this_sub = os.path.splitext(os.path.basename(subj))[0]
        hover_texts.append('participant {}'.format(this_sub))
    # participants = participants * 4

    fig = go.FigureWidget(make_subplots(rows=2, cols=4, horizontal_spacing=0.05, vertical_spacing=0.15, shared_xaxes='columns',
                                        specs=[[{"rowspan": 2, "colspan": 2}, None, {"colspan": 2}, None], [None, None, {"colspan": 2}, None]]))

    fig.add_trace(go.Box(y=to_plot[:, 0], marker_size=12, x=x, boxpoints='all', pointpos=0, marker=dict(opacity=0.65, color=color, line=dict(color='rgb(0,0,0)', width=2)),
                         name='Default', text=hover_texts, line=dict(color='rgb(0,0,0)'), fillcolor="rgb(180, 180, 180)"
                         ), row=1, col=1)
    fig.add_trace(go.Box(y=to_plot[:, 1], marker_size=12, x=x, boxpoints='all', pointpos=0, marker=dict(opacity=0.65, color=color, line=dict(color='rgb(0,0,0)', width=2)),
                         name='Default subTR', text=hover_texts, line=dict(color='rgb(0,0,0)'), fillcolor="rgb(180, 180, 180)"
                         ), row=1, col=1)
    fig.add_trace(go.Box(y=to_plot[:, 2], marker_size=12, x=x, boxpoints='all', pointpos=0, marker=dict(opacity=0.65, color=color, line=dict(color='rgb(0,0,0)', width=2)),
                         name='Refined', text=hover_texts, line=dict(color='rgb(0,0,0)'), fillcolor="rgb(180, 180, 180)"
                         ), row=1, col=1)
    fig.add_trace(go.Box(y=to_plot[:, 3], marker_size=12, x=x, boxpoints='all', pointpos=0, marker=dict(opacity=0.65, color=color, line=dict(color='rgb(0,0,0)', width=2)),
                         name='Refined subTR', text=hover_texts, line=dict(color='rgb(0,0,0)'), fillcolor="rgb(180, 180, 180)"
                         ), row=1, col=1)

    # Plot input signal together with split output signal (X & Y)
    for key, item in evaluation.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if subTR:
                this_real = np.reshape(item['real_y'], (item['real_y'].shape[0] * item['real_y'].shape[1], -1))
                this_pred = np.reshape(item['pred_y'], (item['pred_y'].shape[0] * item['pred_y'].shape[1], -1))          
            else:
                this_real = np.nanmedian(item['real_y'], axis=1)
                this_pred = np.nanmedian(item['pred_y'], axis=1)
        this_sub = os.path.splitext(os.path.basename(key))[0]

        fig.add_trace(go.Scatter(x=np.arange(0, len(this_real[:, 0])), y=this_real[:, 0], mode='lines', visible=False,
                                 line_color='rgb(0,0,0)', opacity=0.5, line_width=3, name=this_sub), row=1, col=3)
        fig.add_trace(go.Scatter(x=np.arange(0, len(this_pred[:, 0])), y=this_pred[:, 0], mode='lines', visible=False,
                                 line_color=color, opacity=0.85, line_width=3, name=this_sub), row=1, col=3)
        fig.add_trace(go.Scatter(x=np.arange(0, len(this_real[:, 1])), y=this_real[:, 1], mode='lines', visible=False,
                                 line_color='rgb(0,0,0)', opacity=0.5, line_width=3, name=this_sub), row=2, col=3)
        fig.add_trace(go.Scatter(x=np.arange(0, len(this_pred[:, 1])), y=this_pred[:, 1], mode='lines', visible=False,
                                 line_color=color, opacity=0.85, line_width=3, name=this_sub), row=2, col=3)
    for i in range(0, 4):
        fig.data[4+i].visible = True

    # Add slider for changing participant
    steps, stepcount = [], 0
    for i in range(4, len(fig.data), 4):
        # print('Subs {}, i = {}, sub i-4: {}'.format(participants, i, participants[i-4]))
        this_sub = os.path.splitext(os.path.basename(participants[stepcount]))[0]
        step = dict(method="update", label="{}".format(this_sub), args=[{"visible": 4 * [True] + [False] * (len(fig.data) - 4)}])
        for j in range(0, 4):
            step["args"][0]["visible"][i+j] = True
        steps.append(step)
        stepcount += 1
    sliders = [dict(active=0, currentvalue={"prefix": "", "visible": False}, pad={"t": 70, "b": 10}, steps=steps)]  # len=0.95, x=0.05)]

    annotations = [dict(x=0.10, y=1.08, xref='paper', yref='paper', text="<b>Model Performance across participants</b>", font=(dict(size=20)), showarrow=False),
                   dict(x=0.855, y=1.08, xref='paper', yref='paper', text="<b>Predicted vs. True gaze position</b>", font=(dict(size=20)), showarrow=False)]

    fig.update_layout(showlegend=False, margin=dict(t=70, l=50, b=50, r=50), plot_bgcolor=bg_color, paper_bgcolor=bg_color,
                      boxmode='group', autosize=False, width=1600, height=650, annotations=annotations, sliders=sliders)

    fig.update_yaxes(range=[-1.1, 1.1], linecolor=line_color, zerolinecolor=line_color, gridcolor=line_color, row=1, col=1)
    fig.update_xaxes(tickfont=dict(size=20), row=1, col=1)
    fig.update_yaxes(range=ylim, row=1, col=3, linecolor=line_color, zerolinecolor=line_color, gridcolor=line_color, ticksuffix='°', title=dict(text='X', standoff=0, font=dict(size=20)))
    fig.update_yaxes(range=ylim, row=2, col=3, linecolor=line_color, zerolinecolor=line_color, gridcolor=line_color, ticksuffix='°', title=dict(text='Y', standoff=0, font=dict(size=20)))
    
    x_range = 150
    x_start = -2
    if subTR:
        x_start = 10000
        x_range *= item['real_y'].shape[1]
    fig.update_xaxes(range=[x_start, x_range+x_start], row=1, col=3, title=dict(text='Functional Volume (TR)', standoff=16, font=dict(size=20)))
    fig.update_xaxes(range=[x_start, x_range+x_start], row=2, col=3)

    return fig
