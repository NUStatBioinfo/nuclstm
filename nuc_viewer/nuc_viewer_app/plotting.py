from bokeh.plotting import figure
from bokeh.models import Range1d, Legend
from bokeh.models.tools import PanTool, BoxSelectTool, BoxZoomTool
from bokeh.palettes import Category10
import numpy as np


def base_position_plot(df, features, start, end=None, y_max=1.0,
                       seq_len=500, plot_width=900, plot_height=600):
    """
    Create a bokeh plot with base position on the x-axis, with multiple lines in it
    and a legend on the right-hand side.

    :param df: pandas.DataFrame
    :param features: list of str, features to include in plot. Must be columns in df.
    :param start: int left-hand base position
    :param end: int right-hand base position; defaults to start + seq_len
    :param y_max: float y-axis maximum. Minimum is 0, as most of this nucleosome data is nonnegative.
    :param seq_len: int length of sequence to view, starting from start
    :param plot_width: int width of plot, in pixels
    :param plot_height: int height of plot, in pixels
    :return: a bokeh.plotting.figure.Figure
    """
    if not end:
        end = start + seq_len

    # identify where desired base locations are in the data supplied.
    x_idx = np.where((df['pos'] >= start) & (df['pos'] <= end))[0].tolist()
    x = df.iloc[x_idx]['pos']

    # etc overhead.
    n_feat = len(features)
    if n_feat < 3:
        cols = ['#EF1428', '#45E320'][0:n_feat]
    else:
        cols = Category10[n_feat]

    # establish plot, set initial view, configure bokeh plotting tools.
    p = figure(plot_width=plot_width
               , plot_height=plot_height
               , x_axis_label='genome position')
    p.y_range = Range1d(0, y_max)
    p.x_range = Range1d(min(x), min((start + seq_len), max(x)))
    for t in [PanTool, BoxSelectTool, BoxZoomTool]:
        p.add_tools(t(dimensions='width'))

    # add lines to plot
    lines = []
    for i in range(n_feat):
        if features[i] in ['nucleosome', 'nucleosome_padded']:
            col = 'blue'
        else:
            col = cols[i % 10]

        lines.append(p.line(x
                            , y=df.iloc[x_idx][features[i]].values.tolist()
                            , color=col
                            , line_width=1.25))

    # configure legend.
    legend = Legend(items=[(features[i], [lines[i]]) for i in range(n_feat)]
                    , location=(0, int(plot_height / 2)))
    p.add_layout(legend, 'right')

    return p


def bokeh_lines(x, ys, labels, x_label, y_label, plot_height=450, plot_width=700):
    """
    Plot a set of lines on a single Bokeh plot.

    :param x: list or np.array that will comprise the x-axis
    :param ys: list of list or np.array the features to plot against x
    :param labels: list of str names of the features being plotted against x. Must have same length as ys.
    :param x_label: str x-axis label
    :param y_label: y-axis label
    :param plot_width: int width of plot, in pixels
    :param plot_height: int height of plot, in pixels
    :return: a bokeh.plotting.figure.Figure
    :return:
    """
    n_feat = len(ys)

    if n_feat != len(labels):
        raise ValueError('len(ys) != len(labels)')

    if n_feat < 3:
        cols = ['#EF1428', '#45E320'][0:n_feat]
    else:
        cols = Category10[n_feat]

    # establish plot.
    p = figure(plot_width=plot_width
               , plot_height=plot_height
               , x_axis_label=x_label
               , y_axis_label=y_label)

    # add lines to plot
    lines = []
    for i in range(n_feat):
        col = cols[i % 10]

        lines.append(p.line(x
                            , y=ys[i]
                            , color=col
                            , line_width=1.25))

    # configure legend.
    legend = Legend(items=[(labels[i], [lines[i]]) for i in range(n_feat)]
                    , location=(0, int(plot_height / 2)))
    p.add_layout(legend, 'right')

    return p
