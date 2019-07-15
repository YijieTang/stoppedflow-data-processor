from __future__ import print_function
import matplotlib
import pandas as pd
import os
import math
import numpy as np
from itertools import cycle, islice

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import to_rgb, to_hex, to_rgba

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from functools import reduce
import re
from datetime import datetime


class AdjustablePlot:
    MAX_LINES = 100
    
    def quickload_df(df):
        colorsCycle = list(islice(cycle(['black', 'red', 'green', 'orange', 'blue']), None, AdjustablePlot.MAX_LINES))
        colors = [colorsCycle[i] for i in range(len(df.columns))]
        axis = plt.figure().gca()
        legends = [str(i+1) for i in range(len(df.columns))]
        scales = [1] * len(df.columns)
        shifts = [0] * len(df.columns)
        return AdjustablePlot(axis, df, colors, legends, scales, shifts)
    
    
    def __init__(self, axis, df, colors, legends, scales, shifts, xlabel='', ylabel=''):
        self.axis = axis
        self.df = df
        self.colors = colors
        self.legends = legends
        self.scales = scales
        self.shifts = shifts
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.existingLines = 0
        
    
    def plot(self):
        df = self.df
        shifts = self.shifts
        scales = self.scales
        axis = self.axis
        colors = self.colors
        legends = self.legends
        xlabel = self.xlabel
        ylabel = self.ylabel
        
        existingLines = len(axis.lines) # record number of lines before plot
        self.existingLines = existingLines
        
        for i in range(len(df.columns)):
            shift = shifts[i]
            scale = scales[i]
            df_adjust = df.iloc[:,[i]] * scale + shift
            df_adjust.plot.line(
                ax=axis,  # plot in the defined axis
                xlim=(float(df.index[0]), float(df.index[-1])),
                legend=None
            )
        
        for i in range(0, len(df.columns)):
            line = axis.lines[i + existingLines]
            if i < len(colors):
                line.set_color(colors[i])
            if i < len(legends):
                line.set_label(legends[i])
                
        # set labels and legends
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        handles, _ = axis.get_legend_handles_labels()
        axis.legend().set_draggable(True)
        
    
    def plot_interactive_buttons(self):
        def change_color(axis, line, color):
            line.set_color(color)
            axis.legend(loc=axis.get_legend()._loc_real).set_draggable(True)

        def change_data(line, df, shift, scale):
            axis = line.axes
            index = axis.lines.index(line)
            origin = df.iloc[:,[index]]
            line.set_ydata(origin*scale + shift)
            axis.relim()
            axis.autoscale_view(True, True, True)

        def change_legend(axis, line, legend):
            line.set_label(legend)
            axis.legend(loc=axis.get_legend()._loc_real).set_draggable(True)

        def change_linewidth(line, width):
            line.set_linewidth(width)
            axis = line.axes
            axis.legend(loc=axis.get_legend()._loc_real).set_draggable(True)
            
        def change_marker(line, marker):
            line.set_marker(marker)
            axis = line.axes
            axis.legend(loc=axis.get_legend()._loc_real).set_draggable(True)
            
        def change_markersize(line, markersize):
            line.set_markersize(markersize)
            axis = line.axes
            axis.legend(loc=axis.get_legend()._loc_real).set_draggable(True)
        # do not plot interactive buttons if there are too many curves

        df = self.df
        shifts = self.shifts
        scales = self.scales
        axis = self.axis
        legends = self.legends
        colors = self.colors

        for i in range(len(df.columns)):
            shift = shifts[i]
            scale = scales[i]
            legend = legends[i]
            color = colors[i]
            line = axis.lines[i + self.existingLines]
            
            markerToText = matplotlib.markers.MarkerStyle.markers
            textToMarker = {}
            for key,value in markerToText.items():
                textToMarker[value] = key
            
            color_widget = widgets.ColorPicker(
                concise=True, description=' ', value=to_hex(color), continuous_update=True)
            width_widget = widgets.FloatText(value=line.get_linewidth(), continuous_update=True)
            marker_widget = widgets.Dropdown(options=textToMarker, value=textToMarker['nothing'])
            msize_widget = widgets.FloatText(value=line.get_markersize(), continuous_update=True)
            legend_widget = widgets.Text(value=legend, placeholder='Legend')
            shift_widget = widgets.FloatText(value=shift, step=0.01, readout_format='.2f', continuous_update=True)
            scale_widget = widgets.FloatText(value=scale, readout_format='.2f', continuous_update=True)

            color_widget.layout.width = '5%'
            width_widget.layout.width = '15%'
            marker_widget.layout.width = '15%'
            msize_widget.layout.width = '15%'
            legend_widget.layout.width = '15%'
            shift_widget.layout.width = '15%'
            scale_widget.layout.width = '15%'

            w_color = interactive(change_color, axis=fixed(axis), line=fixed(line), color=color_widget)
            w_width = interactive(change_linewidth, line=fixed(line), width=width_widget)
            w_marker = interactive(change_marker, line=fixed(line), marker=marker_widget)
            w_msize = interactive(change_markersize, line=fixed(line), markersize=msize_widget)
            w_legend = interactive(change_legend, axis=fixed(axis), line=fixed(line), legend=legend_widget)
            w_data = interactive(change_data, line=fixed(line), df=fixed(df), shift=shift_widget, scale=scale_widget)
            add = lambda a,b:a+b
            childs = reduce(add, map(lambda w:list(w.children), 
                                     [w_color, w_width, w_marker, w_msize, w_legend, w_data]))
            hbox = widgets.HBox(tuple(childs))
            display(hbox)
            
            
def read_single_uv_to_df(filename, encoding='utf-16'):
    return pd.read_csv(filename, encoding=encoding, index_col=0).iloc[:, :-1]


def read_multiple_uv_to_df(filenames, encoding='utf-16'):
    dfs = []
    for filename in filenames:
        dfs.append(read_single_uv_to_df(filename, encoding=encoding))
    return pd.concat(dfs, axis=1)           


def color_range(c1, c2, n):
    rgba1 = to_rgba(c1)
    rgba2 = to_rgba(c2)
    result = [(rgba1[0] + (rgba2[0] - rgba1[0]) / n * i, 
               rgba1[1] + (rgba2[1] - rgba1[1]) / n * i, 
               rgba1[2] + (rgba2[2] - rgba1[2]) / n * i, 
               rgba1[3] + (rgba2[3] - rgba1[3]) / n * i ) for i in range(n) ]
    return result


def shift_to_align_wavelength(df, wavelength):
    index = np.argmin(np.abs(df.index.values - wavelength))
    return -df.iloc[index, :].values


def get_index_of_closest_x_value(df, x):
    return np.argmin(np.abs(df.index.values - x))
