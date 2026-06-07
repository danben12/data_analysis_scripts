import pandas as pd
import numpy as np
import math
from bokeh.plotting import figure
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from bokeh.layouts import column, row
from bokeh.io import output_file, show
from bokeh.models import Select, CustomJS, Legend, ColumnDataSource, LegendItem, HoverTool, TapTool, Label
from statsmodels.stats.multitest import multipletests
from scipy.stats import permutation_test
from bokeh.palettes import Category20
from bokeh.palettes import Category10
import statsmodels.api as sm




def find_droplet_location(df):
    circle_radius = 13200/2
    circle_center_x = 13200 / 2
    circle_center_y = 13200 / 2
    df['distance_to_center'] = np.sqrt((df['X'] - circle_center_x) ** 2 + (df['Y'] - circle_center_y) ** 2)
    df['is_inside_circle'] = df['distance_to_center'] <= 5000
    return df[df['is_inside_circle']].reset_index(drop=True)

def read_data():
    Tk().withdraw()
    file_path = askopenfilename()
    if file_path:
        return pd.read_csv(file_path, encoding='ISO-8859-1')
    else:
        return None



def Volume_distribution(df, time=24):
    p = figure(title='Distribution of Population Fraction by Droplet Volume',
               x_axis_label='log10(Volume)', y_axis_label='Fraction of Population (%)',
               output_backend="webgl",width=800, height=600,x_range=(3,8))
    renderers = []
    labels = []
    colors=Category10[10]
    color_index=0
    for slice in ['C1 - Control','C2- 30ug/ml Amp (X3 MIC)','C2 - 3XMIC 18 ug/ml Gentamicin','C6 - Polymixin B 45ug ml 30XMIC']:
        subset = df[df['Slice'] == slice]
        subset = subset[(subset['time'] == 0) | (subset['time'] == time)]
        subset['bottom_bin'] = np.log10(subset['Volume']).apply(math.floor)
        subset['top_bin'] = np.log10(subset['Volume']).apply(math.ceil)
        total_start = subset.loc[subset['time'] == 0, 'Count'].sum()
        total_end = subset.loc[subset['time'] == time, 'Count'].sum()
        grouped = subset.groupby(['bottom_bin', 'top_bin', 'time'])['Count'].sum()
        grouped = grouped.unstack().fillna(0)
        grouped['start fraction'] = grouped[0] / total_start * 100
        grouped['end fraction'] = grouped[time] / total_end * 100
        grouped['delta'] = grouped['end fraction'] - grouped['start fraction']
        grouped = grouped.reset_index()
        grouped['bin center'] = (grouped['bottom_bin'] + grouped['top_bin']) / 2
        r = p.line(grouped['bin center'], grouped['delta'], line_width=3, alpha=0.8, color=colors[color_index % len(colors)])
        color_index+=1
        renderers.append(r)
        labels.append((slice[4:], [r]))
    legend = Legend(items=labels)
    legend.click_policy = 'hide'
    p.add_layout(legend, 'right')
    return p


def fold_change(df):
    min_fc = -10
    p = figure(
        x_axis_type='log',
        y_axis_type='linear',
        x_axis_label='Volume (μm³)',
        # y_axis_label="Log2 biomass\nfold change",
        output_backend="canvas",
        width=2400,
        height=1600,
        y_range=(-6, 6)
    )
    baseline = p.line(
        [min(df['Volume']), max(df['Volume'])],
        [0, 0],
        color='black',
        line_dash='dashdot',
        line_width=6
    )

    p.xaxis.axis_label_text_font_style = "normal"
    p.yaxis.axis_label_text_font_style = "normal"

    # --- Explicitly set all font colors to black ---
    p.xaxis.axis_label_text_color = "black"
    p.yaxis.axis_label_text_color = "black"
    p.xaxis.major_label_text_color = "black"
    p.yaxis.major_label_text_color = "black"

    p.xaxis.axis_label_text_font_size = "96pt"  # x-axis label
    p.yaxis.axis_label_text_font_size = "96pt"  # y-axis label
    p.xaxis.major_label_text_font_size = "86pt"  # x-axis ticks
    p.yaxis.major_label_text_font_size = "86pt"  # y-axis ticks
    p.xaxis.axis_line_width = 4
    p.yaxis.axis_line_width = 4
    p.xaxis.major_tick_line_width = 4
    p.yaxis.major_tick_line_width = 4
    p.xaxis.minor_tick_line_width = 4
    p.yaxis.minor_tick_line_width = 4
    p.xaxis.major_tick_out = 20
    p.yaxis.major_tick_out = 20
    p.xaxis.minor_tick_out = 10
    p.yaxis.minor_tick_out = 10

    p.yaxis.ticker.desired_num_ticks = 5
    p.yaxis.ticker.num_minor_ticks = 2

    colors = Category10[10]
    color_idx = 0
    legend_items = []

    for Well in ['C5','C8','C6','C4']:
        #Amp - 'C1','C6','C3','C2'
        #Gen - 'C1','C4','C8','C2'
        #Pol - 'C5','C8','C6','C4'
        fold_change_list = []
        volume_list = []
        subset = df[df['Well'] == Well]

        if subset.empty:
            continue

        slice_name = subset['Slice'].iloc[0]
        for drop in subset['Droplet'].unique():
            droplet_data = subset[subset['Droplet'] == drop]
            if droplet_data.empty or droplet_data['Count'].iloc[0] == 0:
                continue

            last_mean = droplet_data['Count'].iloc[-4:].mean()
            if last_mean == 0:
                fc = np.nan
            else:
                fc = last_mean / droplet_data['Count'].iloc[0]

            fold_change_list.append(fc)
            volume_list.append(droplet_data['Volume'].iloc[0])

        if len(volume_list) == 0:
            continue

        n = min(len(volume_list), len(fold_change_list))
        volume_list = volume_list[:n]
        fold_change_list = fold_change_list[:n]

        fold_change_arr = np.log2(fold_change_list)
        fold_change_arr = np.where(np.isnan(fold_change_arr), min_fc, fold_change_arr)

        data = pd.DataFrame({'Volume': volume_list, 'fold change': fold_change_arr})
        data = data.sort_values(by='Volume').reset_index(drop=True)
        sub_df = data[data['fold change'] > min_fc].reset_index(drop=True)
        if sub_df.empty:
            continue

        # Moving average and SE
        sub_df['moving average'] = sub_df['fold change'].rolling(window=100, min_periods=1).mean()
        sub_df['std'] = sub_df['fold change'].rolling(window=100, min_periods=1).std()
        # sub_df['count'] = sub_df['fold change'].rolling(window=100, min_periods=1).count()
        # sub_df['SE'] = sub_df['std'] / np.sqrt(sub_df['count'])
        sub_df['upper'] = sub_df['moving average'] + sub_df['std']
        sub_df['lower'] = sub_df['moving average'] - sub_df['std']

        sub_source = ColumnDataSource(sub_df)

        # Add shaded area (±SE)
        varea_renderer = p.varea(
            x='Volume', y1='lower', y2='upper', source=sub_source,
            fill_color=colors[color_idx % len(colors)], fill_alpha=0.2
        )

        # Add moving average line
        line_renderer = p.line(
            'Volume', 'moving average', source=sub_source,
            line_width=6, color=colors[color_idx % len(colors)]
        )

        if Well == 'C5':
            slice_name = 'Control'
        if Well == 'C8':
            slice_name = 'Low concentration'
        if Well == 'C6':
            slice_name = 'Medium concentration'
        if Well == 'C4':
            slice_name = 'High concentration'
        legend_items.append(LegendItem(label=f'{slice_name}',
                                       renderers=[line_renderer, varea_renderer]))
        color_idx += 1

    legend_items.append(LegendItem(label='Baseline (0)',
                                   renderers=[baseline]))

    # Custom legend combining both line + shaded area
    legend = Legend(items=legend_items, location='top_right', click_policy='hide')

    # --- Set legend fonts to black ---
    legend.label_text_font_size = "40pt"
    legend.label_text_color = "black"

    legend.glyph_width = 80
    legend.glyph_height = 40
    legend.spacing = 20
    legend.padding = 20
    # p.add_layout(legend, 'right')
    return p




data=read_data()
data=find_droplet_location(data)
data=data[data['log_Volume'] >= 3]
p=fold_change(data)
output_file("a.html")
show(p)



