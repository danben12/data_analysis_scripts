import math
import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from bokeh.io import output_file
from bokeh.models import Div, HoverTool, TapTool, ColorBar, CheckboxGroup, LogTicker, LinearColorMapper, \
    BasicTicker, Legend, LegendItem, BoxAnnotation, Span, Whisker, Label, Spacer, \
    ColumnDataSource, CDSView, Select, CustomJS, GlyphRenderer,FixedTicker
from bokeh.models.formatters import CustomJSTickFormatter
from bokeh.layouts import column, row
from bokeh.palettes import Category20, RGB,RdBu, bokeh, Viridis256
from bokeh.plotting import figure, show, markers
from bokeh.transform import linear_cmap, jitter,dodge
from scipy.stats import linregress
from matplotlib import cm
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests
from scipy.stats import gaussian_kde
from bokeh.palettes import linear_palette
import colorcet as cc

FIG_WIDTH = 800
FIG_HEIGHT = 600
FIG_BACKEND = "webgl"
SCATTER_SIZE = 6
LINE_WIDTH = 2
LINE_WIDTH_EMPHASIS = 3
LINE_WIDTH_META = 2
LINE_WIDTH_BOX = 1.5
LINE_WIDTH_BRACKET = 1
LINE_WIDTH_BRACKET_LEG = 1.5
VIOLIN_MAX_WIDTH = 80
JITTER_WIDTH = 35


def apply_standard_figure_style(p):
    """Match bottom dashboard figures: 800x600, webgl, black axis labels."""
    p.width = FIG_WIDTH
    p.height = FIG_HEIGHT
    for axis in (p.xaxis, p.yaxis):
        axis.axis_label_text_color = "black"
        axis.major_label_text_color = "black"
        axis.axis_label_text_font_style = "normal"


def standard_figure(**kwargs):
    kwargs.setdefault("output_backend", FIG_BACKEND)
    kwargs.setdefault("width", FIG_WIDTH)
    kwargs.setdefault("height", FIG_HEIGHT)
    p = figure(**kwargs)
    apply_standard_figure_style(p)
    return p


def read_data():
    Tk().withdraw()
    file_path = askopenfilename()
    if file_path:
        return pd.read_csv(file_path, encoding='ISO-8859-1')
    else:
        return None
def split_data_to_chips():
    data=read_data()
    chips = {slice_id: df.reset_index(drop=True) for slice_id, df in data.groupby('Slice')}
    return chips
def initial_stats(data):
    filtered_chips = {slice_id: df[df['time'] == 0].reset_index(drop=True) for slice_id, df in data.items()}
    return filtered_chips
def get_slice(data, slice_id):
    slice_data = data[slice_id]
    experiment_time = slice_data['time'].max()
    time_steps = np.diff(sorted(slice_data['time'].unique()))[0]
    chip = {droplet_id: df.reset_index(drop=True) for droplet_id, df in slice_data.groupby('Droplet')}
    return chip,experiment_time,time_steps

def stats_box(df,time, max_step,chip_name):
    volume =np.log10(df['Volume'].sum())
    mean=np.log10(df['Volume'].mean())
    std=np.log10(df['Volume'].std())
    bacteria_pool = df['Count'].sum()
    chip_density = np.log10(bacteria_pool / 10**volume)

    stats_text = (f"Chip: {chip_name}<br>"
                  f"Total droplets volume: 10<sup>{volume:.2f}</sup><br>"
                  f"Actual Droplets Mean Size: 10<sup>{mean:.2f}</sup><br>"
                  f"Actual Droplets Standard Deviation: 10<sup>{std:.2f}</sup><br>"
                  f"Number of bacteria: {bacteria_pool}<br>"
                  f"Chip Density: 10<sup>{chip_density:.2f}</sup><br>"
                  f"Time: {time}<br>"
                  f"Time Step: {max_step}<br>")
    stats_div = Div(
        text=stats_text,
        width=500,
        height=300
    )
    stats_div.styles = {
        'text-align': 'left',
        'margin': '10px auto',
        'font-size': '12pt',
        'font-family': 'Arial, sans-serif',
        'color': 'black',
        'background-color': 'lightgray',
        'border': '1px solid black',
        'padding': '20px',
        'box-shadow': '5px 5px 5px 0px lightgray',
        'border-radius': '10px',
        'line-height': '1.5em',
        'font-weight': 'bold',
        'white-space': 'pre-wrap',
        'word-wrap': 'break-word',
        'overflow-wrap': 'break-word',
        'text-overflow': 'ellipsis',
        'hyphens': 'auto'
    }
    return column(stats_div)

def droplet_histogram(df):
    bins = np.logspace(3, 8, num=16)
    hist = standard_figure(x_axis_type='log',
                           x_axis_label='Volume',
                           y_axis_label='Frequency')

    hist_data = np.histogram(df['Volume'], bins=bins)
    hist_data_occupied = np.histogram(df[df['Count'] > 0]['Volume'], bins=bins)
    # Create a ColumnDataSource from histogram data
    source = ColumnDataSource(data=dict(
        top=hist_data[0],
        bottom=np.zeros_like(hist_data[0]),
        left=bins[:-1],
        right=bins[1:],
        top_occupied=hist_data_occupied[0]
    ))
    view = CDSView()
    hist.quad(top='top', bottom='bottom', left='left', right='right',
              color='gray', alpha=0.5, legend_label='Total Droplets', source=source, view=view)
    hist.quad(top='top_occupied', bottom='bottom', left='left', right='right',
              color='blue', alpha=0.1, legend_label='Occupied Droplets', source=source, view=view)
    hist.legend.label_text_color = "black"
    droplet_num = int(df['Volume'].count())
    occupied_droplets = int(df[df['Count'] > 0]['Volume'].count())
    occupancy_rate = occupied_droplets / droplet_num
    stats_text = f"Droplets count: {droplet_num}<br>Occupied Droplets: {occupied_droplets}<br>Occupancy Rate: {occupancy_rate:.2%}"
    stats_div = Div(text=stats_text, width=400, height=100)
    stats_div.styles = {'text-align': 'center', 'margin': '10px auto', 'font-size': '12pt',
                        'font-family': 'Arial, sans-serif', 'color': 'black', 'background-color': 'lightgray',
                        'border': '1px solid black', 'padding': '10px', 'box-shadow': '5px 5px 5px 0px lightgray',
                        'border-radius': '10px', 'line-height': '1.5em', 'font-weight': 'bold',
                        'white-space': 'pre-wrap', 'word-wrap': 'break-word', 'overflow-wrap': 'break-word',
                        'text-overflow': 'ellipsis', 'hyphens': 'auto'}
    combined_plot = column(hist, stats_div)
    return combined_plot


def N0_Vs_Volume(df, Vc):
    source = ColumnDataSource(df)
    view = CDSView()
    scatter = standard_figure(x_axis_type='log',
                              y_axis_type='log',
                              x_axis_label='Volume',
                              y_axis_label='N0')
    scatter_renderer = scatter.scatter('Volume', 'Count', source=source, view=view, color='gray', alpha=1,size=SCATTER_SIZE,
                                       legend_label='N0 vs. Volume')
    hover = HoverTool(tooltips=[('Volume', '@Volume'), ('Count', '@Count'), ('Droplet ID', '@Droplet')],
                      renderers=[scatter_renderer])
    scatter.add_tools(hover)
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)
    scatter.legend.label_text_color = "black"
    filtered_df = df[df['Count'] > 0]
    filtered_df = filtered_df[filtered_df['Volume'] >= Vc]
    x = np.log10(filtered_df['Volume'])
    y = np.log10(filtered_df['Count'])
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_values = np.linspace(min(df['Volume']), max(df['Volume']), 100)
    y_values = 10 ** (intercept + slope * np.log10(x_values))
    stats_text = f'y = {slope:.2f}x + {intercept:.2f}<br>R² value: {r_value ** 2:.2f}'
    scatter.line(x_values, y_values, color='red', legend_label='Linear Regression', line_width=LINE_WIDTH)
    stats_div = Div(text=stats_text, width=400, height=100)
    stats_div.styles = {'text-align': 'center', 'margin': '10px auto', 'font-size': '12pt',
                        'font-family': 'Arial, sans-serif', 'color': 'black', 'background-color': 'lightgray',
                        'border': '1px solid black', 'padding': '10px', 'box-shadow': '5px 5px 5px 0px lightgray',
                        'border-radius': '10px', 'line-height': '1.5em', 'font-weight': 'bold',
                        'white-space': 'pre-wrap', 'word-wrap': 'break-word', 'overflow-wrap': 'break-word',
                        'text-overflow': 'ellipsis', 'hyphens': 'auto'}
    combined_plot = column(scatter, stats_div)
    return combined_plot


def Initial_Density_Vs_Volume(df, initial_density):
    df['initial density'] = df['Count'] / df['Volume']
    source = ColumnDataSource(df)
    view = CDSView()
    scatter = standard_figure(x_axis_type='log',
                              y_axis_type='log',
                              x_axis_label='Volume (μm³)',
                              y_axis_label='Initial Density (pixels/μm³)')
    scatter_renderer = scatter.scatter('Volume', 'initial density', source=source, view=view, color='gray', alpha=1,size=SCATTER_SIZE)
    hover = HoverTool(
        tooltips=[('Volume', '@Volume'), ('Initial Density', '@{initial density}'), ('Droplet ID', '@Droplet')],
        renderers=[scatter_renderer])
    scatter.add_tools(hover)
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)
    filtered_sorted_df = df[df['initial density'] > 0].sort_values(by='Volume').reset_index()
    log_density = np.log10(filtered_sorted_df['initial density'])
    rolling_mean = log_density.rolling(window=100,min_periods=1).mean()
    scatter.line(filtered_sorted_df['Volume'], 10 ** rolling_mean, color='red',line_width=LINE_WIDTH)
    scatter.line([min(df['Volume']), max(df['Volume'])], [initial_density, initial_density], color='black',line_width=LINE_WIDTH)
    convergence_window = 2
    tolerance = 0.05
    differences = np.abs(1 - (10 ** rolling_mean / initial_density))
    for i in range(len(differences) - convergence_window):
        window_mean_diff = differences.iloc[i:i + convergence_window].mean()
        if window_mean_diff <= tolerance:
            closest_index = i + convergence_window // 2
            break
        else:
            closest_index = differences.idxmin()
    closest_point = filtered_sorted_df.loc[closest_index]
    closest_volume = closest_point['Volume']
    vline = Span(location=closest_volume, dimension='height', line_color='black', line_dash='dashed', line_width=LINE_WIDTH)
    scatter.add_layout(vline)
    scatter.renderers.append(vline)
    invisible_line = scatter.line([0], [0], color='black', line_dash='dashed', line_width=LINE_WIDTH)
    legend = Legend(items=[
        LegendItem(label='Initial density vs. Volume', renderers=[scatter.renderers[0]]),
        LegendItem(label='Rolling Mean', renderers=[scatter.renderers[1]]),
        LegendItem(label='Initial Density', renderers=[scatter.renderers[2]]),
        LegendItem(label='Vc', renderers=[invisible_line])
    ], location='top_right')
    legend.label_text_color = "black"
    scatter.add_layout(legend)
    return scatter, closest_volume


def Distribution_comparison(dic, time):
    combined_df = pd.concat(dic.values(), ignore_index=True)
    combined_df = combined_df[combined_df['Count'] >= 0]

    df_start = combined_df[combined_df['time'] == 0]
    df_end = combined_df[combined_df['time'] == time]

    vals_start = np.log10(df_start['Volume'].values)
    weights_start = df_start['Count'].values
    vals_end = np.log10(df_end['Volume'].values)
    weights_end = df_end['Count'].values

    min_x = min(vals_start.min(), vals_end.min())
    max_x = max(vals_start.max(), vals_end.max())
    x_grid = np.linspace(min_x, max_x, 200)

    kde_start = gaussian_kde(vals_start, weights=weights_start)
    kde_end = gaussian_kde(vals_end, weights=weights_end)
    y_start = kde_start(x_grid)
    y_end = kde_end(x_grid)

    p = standard_figure(
        x_axis_label='Volume (μm³)',
        y_range=(0, 1),
        x_axis_type="log"
    )

    # --- Custom Tick Formatter adjusted for actual log values ---
    p.xaxis.formatter = CustomJSTickFormatter(code="""
        const superscripts = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
            '-': '⁻'
        };
        // Get the exponent (e.g., log10(1000) = 3) and round to avoid float errors
        const log_val = Math.round(Math.log10(tick));
        const t_str = log_val.toString();
        let exp = "";
        for (let i = 0; i < t_str.length; i++) {
            exp += superscripts[t_str[i]];
        }
        return "10" + exp;
    """)

    p.yaxis.ticker.desired_num_ticks = 5
    p.yaxis.ticker.num_minor_ticks = 2

    # Transform x_grid back to linear space (10^x) for plotting on the log axis
    x_grid_plot = 10 ** x_grid

    # Added legend_labels back in so the p.legend calls below don't throw an error
    p.varea(x=x_grid_plot, y1=0, y2=y_start, fill_color="grey", fill_alpha=0.3,legend_label='Start (t = 0 h)')
    p.line(x=x_grid_plot, y=y_start, line_color="black", line_width=LINE_WIDTH, line_dash="dashed",legend_label='Start (t = 0 h)')
    p.varea(x=x_grid_plot, y1=0, y2=y_end, fill_color="black", fill_alpha=0.5,legend_label='End (t = 24 h)')
    p.line(x=x_grid_plot, y=y_end, line_color="black", line_width=LINE_WIDTH,legend_label='End (t = 24 h)')

    p.legend.label_text_color = "black"
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.background_fill_color=None
    p.legend.border_line_color=None
    p.outline_line_color = None
    p.border_fill_color = None
    p.background_fill_color = None
    return p

def Fraction_in_each_bin(dic, time):
    combined_df = pd.concat(dic.values(), ignore_index=True)
    combined_df = combined_df[(combined_df['time'] == 0) | (combined_df['time'] == time)]
    combined_df['bottom_bin'] = np.log10(combined_df['Volume']).apply(math.floor)
    combined_df['top_bin'] = np.log10(combined_df['Volume']).apply(math.ceil)
    combined_df['bottom_bin'] = np.where(combined_df['bottom_bin'] >= 6, 6, combined_df['bottom_bin'])
    combined_df['top_bin'] = np.where(combined_df['top_bin'] >= 6, 6, combined_df['top_bin'])
    start_total = combined_df[combined_df['time'] == 0]['Count'].sum()
    end_total = combined_df[combined_df['time'] == time]['Count'].sum()
    ratio=end_total/start_total
    bins = combined_df.groupby(['bottom_bin', 'top_bin', 'time'])['Count'].sum().unstack().fillna(0)
    bins['start fraction'] = bins[0] / start_total * 100
    bins['end fraction'] = bins[time] / end_total * 100
    bin_labels = [
        "> 6" if int(start) == 6 and int(end) == 6 else f"{int(start)}–{int(end)}"
        for start, end in bins.index
    ]
    p = standard_figure(
        title='Fraction of Population in Each Bin at Start and End of the experiment',
        x_range=bin_labels,
        y_axis_label='Fraction of Population (%)',
        x_axis_label='Volume Bins (log10 μm³)',
        y_range=(0, 100)
    )
    bar_width = 0.4
    p.vbar(
        x=dodge('x', -bar_width / 2, range=p.x_range),
        top='start fraction',
        width=bar_width,
        source={'x': bin_labels, 'start fraction': bins['start fraction']},
        color='gray',
        legend_label='Start'
    )
    p.vbar(
        x=dodge('x', bar_width / 2, range=p.x_range),
        top='end fraction',
        width=bar_width,
        source={'x': bin_labels, 'end fraction': bins['end fraction']},
        color='black',
        legend_label='End'
    )
    for i, label in enumerate(bin_labels):
        start_val = bins['start fraction'].iloc[i]
        end_val = bins['end fraction'].iloc[i]
        delta = end_val - start_val
        sign = "+" if delta >= 0 else "–"
        text = f"{sign}{abs(delta):.1f}%"
        y_pos = max(start_val, end_val) + 3  # offset slightly above the taller bar
        p.add_layout(Label(x=i+0.5, y=y_pos, text=text, text_align='center', text_font_size='10pt', text_font_style='bold'))
    if ratio >= 1:
        ratio_text = f"Metapopulation growth: +{(ratio - 1) * 100:.1f}%"
    else:
        ratio_text = f"Metapopulation decline: –{(1 - ratio) * 100:.1f}%"

    p.add_layout(Label(
        x=len(bin_labels) / 2,
        y=95,
        text=ratio_text,
        text_align='center',
        text_font_size='12pt',
        text_font_style='bold',
        text_color='darkred' if ratio < 1 else 'darkgreen'
    ))
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    return p


def fold_change(data_dict, Vc):
    fold_change = np.array([])
    Volume = np.array([])
    droplet_id = np.array([])
    google_drive_url = np.array([])  # Array to store Google Drive URLs
    min_fc = -10
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            if value['Count'].iloc[-4:].mean() == 0:
                fold_change = np.append(fold_change, np.nan)
                Volume = np.append(Volume, value['Volume'].iloc[0])
                droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
                google_drive_url = np.append(google_drive_url, value['Google Drive Link'].iloc[0])  # Add URL
            else:
                fold_change = np.append(fold_change, value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])
                Volume = np.append(Volume, value['Volume'].iloc[0])
                droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
                google_drive_url = np.append(google_drive_url, value['Google Drive Link'].iloc[0])  # Add URL
    fold_change = np.log2(fold_change)
    fold_change = np.where(np.isnan(fold_change), min_fc, fold_change)
    df = pd.DataFrame(
        {'Volume': Volume, 'fold change': fold_change, 'Droplet': droplet_id, 'Google Drive Link': google_drive_url})
    df = df.sort_values(by='Volume').reset_index(drop=True)
    sub_df = df[df['fold change'] > min_fc].reset_index(drop=True)
    sub_df['moving average'] = sub_df['fold change'].rolling(window=100, min_periods=1).mean()
    source = ColumnDataSource(df)
    sub_source = ColumnDataSource(sub_df)
    view = CDSView()
    sub_view = CDSView()
    scatter = standard_figure(x_axis_type='log', y_axis_type='linear',
                              x_axis_label='Volume (μm³)',
                              y_range=(-6, 8))
    scatter.yaxis.minor_tick_line_color = None
    scatter.yaxis.ticker = [-4, -2, 0, 2, 4, 6]

    scatter.scatter('Volume', 'fold change', source=source, view=view, color='gray', alpha=1, size=SCATTER_SIZE)
    scatter.line('Volume', 'moving average', source=sub_source, view=sub_view, color='black', line_width=LINE_WIDTH)

    total_initial_counts = sum(value['Count'].iloc[0] for value in data_dict.values() if value['Count'].iloc[0] != 0)
    total_final_counts = sum(
        value['Count'].iloc[-4:].mean() for value in data_dict.values() if value['Count'].iloc[0] != 0)
    metapopulation_fold_change = np.log2(total_final_counts / total_initial_counts)

    scatter.line([min(df['Volume']), max(df['Volume'])], [metapopulation_fold_change, metapopulation_fold_change],
                 color='black', line_dash='dashed', line_width=LINE_WIDTH)
    # baseline = scatter.line(
    #     [min(df['Volume']), max(df['Volume'])],
    #     [0, 0],
    #     color='black',
    #     line_dash='dashdot',
    #     line_width=3
    # )
    # vc_line = scatter.line([Vc,Vc],[-12,12],color='black',line_dash='dashed',line_width=3)

    hover = HoverTool(tooltips=[('Volume', '@Volume'), ('Fold Change', '@{fold change}'), ('Droplet ID', '@Droplet')],
                      renderers=[scatter.renderers[0]])
    scatter.add_tools(hover)

    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)

    legend = Legend(items=[
        LegendItem(label='Fold change (FC)', renderers=[scatter.renderers[0]]),
        LegendItem(label='FC moving average', renderers=[scatter.renderers[1]]),
        LegendItem(label='Metapopulation FC', renderers=[scatter.renderers[2]]),
        # LegendItem(label='Vc', renderers=[vc_line]),
        # LegendItem(label='FC Baseline (0)', renderers=[baseline])
    ], location='top_right')

    legend.label_text_color = "black"
    scatter.outline_line_color = None
    scatter.border_fill_color = None
    scatter.background_fill_color = None
    return scatter

def growth_curves(dict):
    valid_droplets = []
    for key, value in dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            valid_droplets.append(value)
    df = pd.concat(valid_droplets, ignore_index=True)
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)
    df['lower bin'] = np.where(df['lower bin']>=6, 6, df['lower bin'])
    df['upper bin'] = np.where(df['upper bin']>=6, 6, df['upper bin'])
    grouped = df.groupby(['lower bin', 'upper bin', 'time'])
    counts = grouped.size().reset_index(name='sample_count')
    means = grouped['Count'].mean().reset_index(name='mean')
    stds = grouped['Count'].std().reset_index(name='std')
    result = pd.merge(counts, means, on=['lower bin', 'upper bin', 'time'])
    result = pd.merge(result, stds, on=['lower bin', 'upper bin', 'time'])
    result['SE'] = result['std'] / np.sqrt(result['sample_count'])
    result['mean + se'] = result['mean'] + result['SE']
    result['mean - se'] = result['mean'] - result['SE']
    unique_bins = result.groupby(['lower bin', 'upper bin']).ngroups
    high_contrast_color_map=[cc.CET_D1[0], cc.CET_D1[80],cc.CET_D1[180], cc.CET_D1[230]]
    palette = linear_palette(high_contrast_color_map, unique_bins)
    p1 = standard_figure(title='Growth Curves', x_axis_label='Time', y_axis_label='Mean Count')
    p2 = standard_figure(x_axis_label='Time (h)',
                         y_axis_label='population Mean',
                         y_axis_type='log')
    p2.yaxis.ticker.desired_num_ticks = 5
    legend_items_1, legend_items_2 = [], []
    for (color, ((lower_bin, upper_bin), group)) in zip(palette, result.groupby(['lower bin', 'upper bin'])):
        source = ColumnDataSource(group)
        line_1 = p1.line('time', 'mean', source=source, color=color, line_width=LINE_WIDTH)
        varea_1 = p1.varea(x='time', y1='mean - se', y2='mean + se', source=source, color=color, alpha=0.2)
        line_2 = p2.line('time', 'mean', source=source, color=color, line_width=LINE_WIDTH)
        varea_2 = p2.varea(x='time', y1='mean - se', y2='mean + se', source=source, color=color, alpha=0.2)
        if lower_bin==6:
            label=f'Bin > {lower_bin}'
        else:
            label=f'Bin {lower_bin}-{upper_bin}'
        legend_items_1.append(LegendItem(label=label, renderers=[line_1, varea_1]))
        legend_items_2.append(LegendItem(label=label, renderers=[line_2, varea_2]))
    meta = df.groupby('time')['Count'].agg(['sum', 'std', 'count']).reset_index()
    meta['SE'] = meta['std'] / np.sqrt(meta['count'])
    meta['sum + SE'] = meta['sum'] + meta['SE']
    meta['sum - SE'] = meta['sum'] - meta['SE']

    # Add metapopulation line to p1
    meta_source = ColumnDataSource(meta)
    meta_linear_line=p1.line('time', 'sum', source=meta_source, color='black', line_width=LINE_WIDTH_META,line_dash='dashed')
    meta_linear_SE=p1.varea(x='time', y1='sum - SE', y2='sum + SE', source=meta_source, color='black', alpha=0.15)
    legend_item_1 = LegendItem(label='Metapopulation', renderers=[meta_linear_line,meta_linear_SE])
    legend_items_1.append(legend_item_1)
    meta_log_line=p2.line('time', 'sum', source=meta_source, color='black', line_width=LINE_WIDTH_META,line_dash='dashed')
    meta_log_SE=p2.varea(x='time', y1='sum - SE', y2='sum + SE', source=meta_source, color='black', alpha=0.15)
    legend_item_2 = LegendItem(label='Metapopulation', renderers=[meta_log_line,meta_log_SE])
    legend_items_2.append(legend_item_2)
    legend_1 = Legend(items=legend_items_1, location='top_right')
    legend_2 = Legend(items=legend_items_2, location='top_right')
    p1.add_layout(legend_1, 'right')
    p1.legend.click_policy = 'hide'
    p2.add_layout(legend_2, 'right')
    p2.legend.click_policy = 'hide'
    return row(p1,p2)


def normalize_growth_curves(data_dict):
    valid_droplets = []
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            valid_droplets.append(value)
    df = pd.concat(valid_droplets, ignore_index=True)
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)
    df['lower bin'] = np.where(df['lower bin'] >= 6, 6, df['lower bin'])
    df['upper bin'] = np.where(df['upper bin'] >= 6, 6, df['upper bin'])
    grouped = df.groupby(['lower bin', 'upper bin', 'time'])
    counts = grouped.size().reset_index(name='sample_count')
    means = grouped['Count'].mean().reset_index(name='mean')
    stds = grouped['Count'].std().reset_index(name='std')
    result = pd.merge(counts, means, on=['lower bin', 'upper bin', 'time'])
    result = pd.merge(result, stds, on=['lower bin', 'upper bin', 'time'])
    result['SE'] = result['std'] / np.sqrt(result['sample_count'])
    result['max_mean'] = result.groupby(['lower bin', 'upper bin'])['mean'].transform('max')
    result['normalized_mean'] = result['mean'] / result['max_mean']
    result['normalized_SE'] = result['SE'] / result['max_mean']
    result['mean + SE'] = result['normalized_mean'] + (result['normalized_SE'])
    result['mean - SE'] = result['normalized_mean'] - (result['normalized_SE'])
    unique_bins = result.groupby(['lower bin', 'upper bin']).ngroups
    high_contrast_color_map=[cc.CET_D1[0], cc.CET_D1[80],cc.CET_D1[180], cc.CET_D1[230]]
    palette = linear_palette(high_contrast_color_map, unique_bins)
    p1 = standard_figure(title='Normalized to Max Growth Curves by Droplets Volume Bin',
                         x_axis_label='Time (h)',
                         y_axis_label='Normalized population size in Volume bin (pixels)')
    p2 = standard_figure(x_axis_label='Time (h)',
                         y_axis_label='Relative population size (Nₜ / N₀)',
                         y_axis_type='log')
    legend_items_1 = []
    legend_items_2 = []
    for (color, ((lower_bin, upper_bin), group)) in zip(palette, result.groupby(['lower bin', 'upper bin'])):
        source = ColumnDataSource(group)
        line_1 = p1.line('time', 'normalized_mean', source=source, color=color, line_width=LINE_WIDTH)
        varea_1 = p1.varea(x='time', y1='mean - SE', y2='mean + SE', source=source, color=color,
                            alpha=0.2)
        line_2 = p2.line('time', 'normalized_mean', source=source, color=color, line_width=LINE_WIDTH)
        varea_2 = p2.varea(x='time', y1='mean - SE', y2='mean + SE', source=source, color=color,
                            alpha=0.2)
        legend_item_1 = LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[line_1, varea_1])
        legend_items_1.append(legend_item_1)
        legend_item_2 = LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[line_2, varea_2])
        legend_items_2.append(legend_item_2)
    legend_1 = Legend(items=legend_items_1, location='top_right')
    legend_2 = Legend(items=legend_items_2, location='top_right')
    p1.add_layout(legend_1, 'right')
    p1.legend.click_policy = 'hide'
    p2.add_layout(legend_2, 'right')
    p2.legend.click_policy = 'hide'
    return row(p1, p2)


def normalize_growth_curves_first_timepoint(data_dict):
    valid_droplets = []
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            valid_droplets.append(value)
    df = pd.concat(valid_droplets, ignore_index=True)
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)
    df['lower bin'] = np.where(df['lower bin'] >= 6, 6, df['lower bin'])
    df['upper bin'] = np.where(df['upper bin'] >= 6, 6, df['upper bin'])
    grouped = df.groupby(['lower bin', 'upper bin', 'time'])
    counts = grouped.size().reset_index(name='sample_count')
    means = grouped['Count'].mean().reset_index(name='mean')
    stds = grouped['Count'].std().reset_index(name='std')
    result = pd.merge(counts, means, on=['lower bin', 'upper bin', 'time'])
    result = pd.merge(result, stds, on=['lower bin', 'upper bin', 'time'])
    result['SE'] = result['std'] / np.sqrt(result['sample_count'])
    first_means = result[result['time'] == 0][['lower bin', 'upper bin', 'mean']].rename(columns={'mean': 'first_mean'})
    result = result.merge(first_means, on=['lower bin', 'upper bin'], how='left')
    result['normalized_mean'] = result['mean'] / result['first_mean']
    result['normalized_SE'] = result['SE'] / result['first_mean']
    result['mean + SE'] = result['normalized_mean'] + (result['normalized_SE'])
    result['mean - SE'] = result['normalized_mean'] - (result['normalized_SE'])
    unique_bins = result.groupby(['lower bin', 'upper bin']).ngroups
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230]]
    palette = linear_palette(high_contrast_color_map, unique_bins)

    p1 = standard_figure(x_axis_label='Time (h)')
    p2 = standard_figure(x_axis_label='Time', y_axis_type='log')
    p1.xaxis.minor_tick_line_color = None
    p1.yaxis.ticker.desired_num_ticks = 5
    p1.yaxis.ticker.num_minor_ticks = 2

    legend_items_1 = []
    legend_items_2 = []

    # Helper dictionary to convert numbers to unicode superscripts
    sup_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
               '-': '⁻'}

    def to_sup(val):
        return ''.join(sup_map.get(char, char) for char in str(val))

    for (color, ((lower_bin, upper_bin), group)) in zip(palette, result.groupby(['lower bin', 'upper bin'])):
        source = ColumnDataSource(group)
        line_1 = p1.line('time', 'normalized_mean', source=source, color=color, line_width=LINE_WIDTH_EMPHASIS)
        varea_1 = p1.varea(x='time', y1='mean - SE', y2='mean + SE', source=source, color=color,
                                       alpha=0.2)
        line_2 = p2.line('time', 'normalized_mean', source=source, color=color, line_width=LINE_WIDTH_EMPHASIS)
        varea_2 = p2.varea(x='time', y1='mean - SE', y2='mean + SE', source=source, color=color,
                                       alpha=0.2)

                    # New label formatting using superscripts
        if lower_bin == 6:
            label = f'> 10{to_sup(lower_bin)}'
        else:
            label = f'10{to_sup(lower_bin)} - 10{to_sup(upper_bin)}'

        legend_item_1 = LegendItem(label=label, renderers=[line_1, varea_1])
        legend_items_1.append(legend_item_1)
        legend_item_2 = LegendItem(label=label, renderers=[line_2, varea_2])
        legend_items_2.append(legend_item_2)

    # Compute metapopulation mean and SE
    meta = df.groupby('time')['Count'].agg(['mean', 'std', 'count']).reset_index()
    meta['SE'] = meta['std'] / np.sqrt(meta['count'])
    meta_first_mean = meta.loc[meta['time'] == 0, 'mean'].values[0]
    meta['normalized_mean'] = meta['mean'] / meta_first_mean
    meta['normalized_SE'] = meta['SE'] / meta_first_mean
    meta['mean + SE'] = meta['normalized_mean'] + meta['normalized_SE']
    meta['mean - SE'] = meta['normalized_mean'] - meta['normalized_SE']

    # Add metapopulation line to p1
    meta_source = ColumnDataSource(meta)
    meta_linear_line = p1.line('time', 'normalized_mean', source=meta_source, color='black', line_width=LINE_WIDTH_META,
                               line_dash='dashed')
    meta_linear_SE = p1.varea(x='time', y1='mean - SE', y2='mean + SE', source=meta_source, color='black', alpha=0.15)
    legend_item_1 = LegendItem(label='Metapopulation', renderers=[meta_linear_line, meta_linear_SE])
    legend_items_1.append(legend_item_1)

    meta_log_line = p2.line('time', 'normalized_mean', source=meta_source, color='black', line_width=LINE_WIDTH_META,
                            line_dash='dashed')
    meta_log_SE = p2.varea(x='time', y1='mean - SE', y2='mean + SE', source=meta_source, color='black', alpha=0.15)
    legend_item_2 = LegendItem(label='Metapopulation', renderers=[meta_log_line, meta_log_SE])
    legend_items_2.append(legend_item_2)

    # # Add titles to the legends
    legend_1 = Legend(items=legend_items_1, location='top_right', title='Droplet size')
    legend_2 = Legend(items=legend_items_2, location='top_right', title='Droplet size')

    for leg in [legend_1, legend_2]:
        leg.title_text_font_style = "bold"
        leg.title_text_color = "black"
        leg.label_text_color = "black"
    p1.outline_line_color = None
    p1.border_fill_color = None
    p1.background_fill_color = None
    # p1.add_layout(legend_1, 'right')
    # p1.legend.click_policy = 'hide'
    # p2.add_layout(legend_2, 'right')
    # p2.legend.click_policy = 'hide'

    custom_ticks = [0, 12, 24, 50, 74]
    p1.xaxis.ticker = FixedTicker(ticks=custom_ticks)
    return row(p1)
    # return row(p1, p2)



def last_4_hours_average(chip, volume):
    last_4_hours = {droplet_id: df[df['time'] > 20].reset_index(drop=True) for droplet_id, df in chip.items()}
    average_counts = np.array([df['Count'].mean() for df in last_4_hours.values()])
    min_average_count = 0.1
    average_counts = np.where(average_counts == 0, min_average_count, average_counts)
    droplet_sizes = [df['Volume'].iloc[0] for df in chip.values()]
    droplet_ids = [df['Droplet'].iloc[0] for df in chip.values()]
    google_drive_urls = [df['Google Drive Link'].iloc[0] for df in chip.values()]  # Add URL
    data = pd.DataFrame({'Volume': droplet_sizes, 'Average Count': average_counts, 'Droplet': droplet_ids, 'Google Drive Link': google_drive_urls})
    data = data.sort_values(by='Volume').reset_index(drop=True)
    data_before = data[data['Volume'] <= volume]
    data_after = data[data['Volume'] > volume]
    data_before = data_before[data_before['Average Count'] > data_before['Average Count'].min()]
    data_after = data_after[data_after['Average Count'] > data_after['Average Count'].min()]
    if not data_before.empty:
        slope_before, intercept_before, r_value_before, _, _ = linregress(np.log10(data_before['Volume']), np.log10(data_before['Average Count']))
        x_values_before = np.linspace(data_before['Volume'].min(), volume, 100)
        y_values_before = 10 ** (intercept_before + slope_before * np.log10(x_values_before))
    else:
        x_values_before, y_values_before = np.array([]), np.array([])
        slope_before, r_squared_before = None, None

    if not data_after.empty:
        slope_after, intercept_after, r_value_after, _, _ = linregress(np.log10(data_after['Volume']), np.log10(data_after['Average Count']))
        x_values_after = np.linspace(volume, data_after['Volume'].max(), 100)
        y_values_after = 10 ** (intercept_after + slope_after * np.log10(x_values_after))
    else:
        x_values_after, y_values_after = np.array([]), np.array([])
        slope_after, r_squared_after = None, None

    source = ColumnDataSource(data)
    view = CDSView()
    scatter = standard_figure(title='Average Number of Bacteria in Last 4 Hours vs. Droplet Size',
                              x_axis_type='log',
                              y_axis_type='log',
                              x_axis_label='Volume',
                              y_axis_label='Average Count')
    scatter.scatter('Volume', 'Average Count', source=source, view=view, color='gray', alpha=1, size=SCATTER_SIZE)
    regression_before_renderer = None
    regression_after_renderer = None
    if x_values_before.any() and y_values_before.any():
        regression_before_renderer = scatter.line(x_values_before, y_values_before, color='red', line_width=LINE_WIDTH)
        label_x_before = (x_values_before[0] + x_values_before[-1]) / 2
        label_y_before = 10 ** (intercept_before + slope_before * np.log10(label_x_before))
        label_before = Label(x=label_x_before, y=label_y_before, text=f'Slope: {slope_before:.2f}', text_color='red')
        scatter.renderers.append(label_before)
        scatter.add_layout(label_before)
    if x_values_after.any() and y_values_after.any():
        regression_after_renderer = scatter.line(x_values_after, y_values_after, color='blue', line_width=LINE_WIDTH)
        label_x_after = (x_values_after[0] + x_values_after[-1]) / 2
        label_y_after = 10 ** (intercept_after + slope_after * np.log10(label_x_after))
        label_after = Label(x=label_x_after, y=label_y_after, text=f'Slope: {slope_after:.2f}', text_color='blue')
        scatter.renderers.append(label_after)
        scatter.add_layout(label_after)
    hover = HoverTool(tooltips=[('Volume', '@Volume'), ('Average Count', '@{Average Count}'), ('Droplet ID', '@Droplet')],
                      renderers=[scatter.renderers[0]])
    scatter.add_tools(hover)
    vline = Span(location=volume, dimension='height', line_color='blue', line_dash='dashed', line_width=LINE_WIDTH)
    scatter.add_layout(vline)
    scatter.renderers.append(vline)
    invisible_line = scatter.line([0], [0], color='blue', line_dash='dashed', line_width=LINE_WIDTH)
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)
    legend_items = [
        LegendItem(label='Average Count', renderers=[scatter.renderers[0]]),
        LegendItem(label='Vc', renderers=[invisible_line])
    ]
    if regression_before_renderer:
        legend_items.append(LegendItem(label='Regression Before', renderers=[regression_before_renderer]))
    if regression_after_renderer:
        legend_items.append(LegendItem(label='Regression After', renderers=[regression_after_renderer]))
    legend = Legend(items=legend_items, location='top_right')
    scatter.add_layout(legend, 'right')
    return scatter

def find_droplet_location(df):
    circle_radius = 13200/2
    circle_center_x = 13200 / 2
    circle_center_y = 13200 / 2
    df['distance_to_center'] = np.sqrt((df['X'] - circle_center_x) ** 2 + (df['Y'] - circle_center_y) ** 2)
    df['is_inside_circle'] = df['distance_to_center'] <= 5000
    return df[df['is_inside_circle']].reset_index(drop=True)


def death_rate_by_bins(dict):
    valid_droplets = []
    for key, value in dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            valid_droplets.append(value)
    df = pd.concat(valid_droplets, ignore_index=True)
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)
    grouped = df.groupby(['lower bin', 'upper bin', 'time'])['Count'].sum().reset_index(name='Count')
    mask = grouped['Count'] > 0
    grouped['log_count'] = grouped[mask]['Count'].apply(np.log)
    window_size = 4
    grouped['slope'] = grouped.groupby(['lower bin', 'upper bin'])['log_count'].transform(
        lambda x: x.rolling(window_size).apply(lambda y: linregress(range(window_size), y)[0]))
    grouped['standard_error'] = grouped.groupby(['lower bin', 'upper bin'])['log_count'].transform(
        lambda x: x.rolling(window_size).apply(lambda y: linregress(range(window_size), y)[4]))
    grouped['slope - standard_error'] = grouped['slope'] - grouped['standard_error']
    grouped['slope + standard_error'] = grouped['slope'] + grouped['standard_error']

    # Calculate metapopulation death rate
    metapopulation = df.groupby('time')['Count'].sum().reset_index(name='metapopulation')
    metapopulation['log_metapopulation'] = np.log(metapopulation['metapopulation'])
    metapopulation['slope'] = metapopulation['log_metapopulation'].rolling(window=window_size).apply(
        lambda x: linregress(range(window_size), x)[0])
    p = standard_figure(title='Death Rate by Bins', x_axis_label='Time', y_axis_label='Slope')
    colors = Category20[20]
    color_index = 0
    legend_items = []
    for (lower_bin, upper_bin), group in grouped.groupby(['lower bin', 'upper bin']):
        source = ColumnDataSource(group)
        view = CDSView()
        line = p.line('time', 'slope', source=source, view=view, color=colors[color_index], line_width=LINE_WIDTH)
        varea = p.varea(x='time', y1='slope - standard_error', y2='slope + standard_error', source=source,
                        color=colors[color_index], alpha=0.2)
        legend_item = LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[line, varea])
        legend_items.append(legend_item)
        color_index = (color_index + 1) % len(colors)

    # Add metapopulation death rate line
    metapopulation_source = ColumnDataSource(metapopulation)
    metapopulation_line = p.line('time', 'slope', source=metapopulation_source, line_width=LINE_WIDTH_EMPHASIS, color='black')
    legend_items.append(LegendItem(label='Metapopulation Death Rate', renderers=[metapopulation_line]))

    legend = Legend(items=legend_items, location='top_right')
    p.add_layout(legend, 'right')
    p.legend.click_policy = 'hide'
    return p


def death_rate_by_droplets(data_dict, chip):
    volumes = []
    max_death_rate = []
    droplet_ids = []
    google_drive_urls = []
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            window_size = 4
            mask = value['Count'] > 0
            value['log_count'] = value[mask]['Count'].apply(np.log)
            value['slope'] = value['log_count'].rolling(window_size).apply(
                lambda x: linregress(range(window_size), x)[0])
            volumes.append(value['Volume'].iloc[0])
            if chip == 'C1 - Control' or chip == 'C8 - Control' or chip == 'C4 - spec 25 ug/ml (0.33x MIC)' or chip == 'C5 - spec 25 ug/ml (0.33x MIC)' or chip=='C4 Control':
                max_death_rate.append(
                    value['slope'].max() if value['slope'].max() > 0 and value['slope'].max() < 1 else np.nan)
            else:
                max_death_rate.append(value['slope'].min())
            droplet_ids.append(key)
            google_drive_urls.append(value['Google Drive Link'].iloc[0])

    df = pd.DataFrame({
        'Volume': np.log10(volumes),
        'Linear_Volume': volumes,  # Added for plotting on the log axis
        'Slope': max_death_rate,
        'Droplet': droplet_ids,
        'Google Drive Link': google_drive_urls
    })

    df['upper bin'] = df['Volume'].apply(math.ceil)
    df['lower bin'] = df['Volume'].apply(math.floor)
    grouped = df.groupby(['lower bin', 'upper bin'])

    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]
    color_map = {group: high_contrast_color_map[i % len(high_contrast_color_map)] for i, group in
                 enumerate(grouped.groups.keys())}
    df['color'] = df.apply(lambda row: color_map[(row['lower bin'], row['upper bin'])], axis=1)

    valid_droplets = [value for key, value in data_dict.items() if value['Count'].iloc[0] != 0]
    metapopulation_df = pd.concat(valid_droplets, ignore_index=True)
    metapopulation = metapopulation_df.groupby('time')['Count'].sum().reset_index(name='metapopulation')
    window_size = 4
    metapopulation['log_metapopulation'] = np.log(metapopulation['metapopulation'])
    metapopulation['slope'] = metapopulation['log_metapopulation'].rolling(window=window_size).apply(
        lambda x: linregress(range(window_size), x)[0]
    )

    if chip == 'C1 - Control' or chip == 'C5 - Control' or chip=='C4 Control':
        mean_death_rate = metapopulation['slope'].max()
        title = 'Maximal slope by Droplets'
    else:
        mean_death_rate = metapopulation['slope'].min()
        title = 'Minimal slope by Droplets'

    p = standard_figure(
        x_axis_label='Volume (µm³)',
        y_axis_label='Growth rate (h⁻¹)',
        x_axis_type="log"
    )

    # Custom Formatter adjusted for log axis ticks
    p.xaxis.formatter = CustomJSTickFormatter(code="""
        const superscripts = {
            '0': '⁰','1': '¹','2': '²','3': '³','4': '⁴',
            '5': '⁵','6': '⁶','7': '⁷','8': '⁸','9': '⁹',
            '-': '⁻'
        };
        const log_val = Math.round(Math.log10(tick));
        const t_str = log_val.toString();
        let exp = "";
        for (let i = 0; i < t_str.length; i++) {
            exp += superscripts[t_str[i]];
        }
        return "10" + exp;
    """)

    p.yaxis.ticker.desired_num_ticks = 5
    p.yaxis.ticker.num_minor_ticks = 2

    permutations_data = pd.DataFrame(columns=['first compared bin', 'second compared bin', 'p-value'])
    lower_bins = sorted(df['lower bin'].unique())

    for k in range(len(lower_bins) - 1):
        i = lower_bins[k]
        j = lower_bins[k + 1]
        data1 = df[df['lower bin'] == i]['Slope']
        data2 = df[df['lower bin'] == j]['Slope']
        if len(data1) >= 2 and len(data2) >= 2:
            p_value = permutation_test(
                (data1, data2),
                lambda x, y: np.mean(x) - np.mean(y),
                n_resamples=1000,
                alternative='two-sided'
            ).pvalue
            permutations_data = pd.concat(
                [permutations_data, pd.DataFrame({'first compared bin': i,
                                                  'second compared bin': j,
                                                  'p-value': p_value}, index=[0])],
                ignore_index=True
            )

    adjusted_results = multipletests(permutations_data['p-value'], method='fdr_bh')
    permutations_data['adjusted p-value'] = adjusted_results[1]

    def replace_p_values(value):
        if value > 0.05:
            return 'NS'
        elif 0.05 >= value > 0.01:
            return '*'
        elif 0.01 >= value > 0.001:
            return '**'
        else:
            return '***'

    permutations_data['adjusted p-value'] = permutations_data['adjusted p-value'].apply(replace_p_values)

    for index, row in permutations_data.iterrows():
        first_bin = row['first compared bin'] + 0.5
        second_bin = row['second compared bin'] - 0.5
        p_value_label = row['adjusted p-value']
        y_position = df['Slope'].max() + 0.03 + index * 0.03
        leg_size = 0.01

        # Convert X coordinates to linear for the log scale
        x_left = 10 ** (first_bin + 0.2)
        x_right = 10 ** (second_bin + 0.8)
        x_mid = 10 ** ((first_bin + second_bin + 1) / 2)

        p.line(x=[x_left, x_right], y=[y_position, y_position],
               line_color="black", line_width=LINE_WIDTH_BRACKET)
        p.line(x=[x_left, x_left],
               y=[y_position, y_position - leg_size], line_color="black", line_width=LINE_WIDTH_BRACKET_LEG)
        p.line(x=[x_right, x_right],
               y=[y_position, y_position - leg_size], line_color="black", line_width=LINE_WIDTH_BRACKET_LEG)
        label = Label(x=x_mid,
                      y=y_position + 0.002,
                      text=p_value_label,
                      text_color="black",
                      text_align="center",
                      text_font_size="12pt")
        p.add_layout(label)

    source = ColumnDataSource(df)
    # Scatter using Linear_Volume
    scatter = p.scatter(x='Linear_Volume', y='Slope', source=source, color='color', size=SCATTER_SIZE)

    box_width = 0.6
    whisker_width = 0.25
    box_line_width = LINE_WIDTH_BOX

    for (lower_bin, upper_bin), group in grouped:
        color = color_map[(lower_bin, upper_bin)]

        # box center in log space
        center = (lower_bin + upper_bin) / 2

        # quartiles and median
        q1 = group['Slope'].quantile(0.25)
        q3 = group['Slope'].quantile(0.75)
        median = group['Slope'].median()

        # whiskers
        iqr = q3 - q1
        upper_whisker = min(group['Slope'].max(), q3 + 1.5 * iqr)
        lower_whisker = max(group['Slope'].min(), q1 - 1.5 * iqr)

        # Convert boundaries to linear for rendering on the log axis
        x_center = 10 ** center
        x_left = 10 ** (center - box_width / 2)
        x_right = 10 ** (center + box_width / 2)
        x_whisker_left = 10 ** (center - whisker_width / 2)
        x_whisker_right = 10 ** (center + whisker_width / 2)

        # --- Draw Box ---
        p.quad(
            top=[q3], bottom=[q1],
            left=[x_left],
            right=[x_right],
            fill_color=color, alpha=0.60,
            line_color="black", line_width=box_line_width
        )

        # --- Median line ---
        p.segment(
            x0=[x_left], y0=[median],
            x1=[x_right], y1=[median],
            line_color="black", line_width=box_line_width
        )

        # --- Whiskers ---
        # Vertical whisker lines
        p.segment(
            x0=x_center, y0=upper_whisker,
            x1=x_center, y1=q3,
            line_color="black", line_width=box_line_width
        )

        p.segment(
            x0=x_center, y0=lower_whisker,
            x1=x_center, y1=q1,
            line_color="black", line_width=box_line_width
        )

        # Horizontal whisker caps
        p.segment(
            x0=x_whisker_left, y0=upper_whisker,
            x1=x_whisker_right, y1=upper_whisker,
            line_color="black", line_width=box_line_width
        )

        p.segment(
            x0=x_whisker_left, y0=lower_whisker,
            x1=x_whisker_right, y1=lower_whisker,
            line_color="black", line_width=box_line_width
        )

    # Line converted to linear span 10^3 to 10^8
    p.line(x=[10 ** 3, 10 ** 8], y=[mean_death_rate, mean_death_rate],
           line_dash='dashed', line_color='black', line_width=LINE_WIDTH_EMPHASIS)

    legend = Legend(items=[LegendItem(label='Metapopulation Growth rate', renderers=[p.renderers[-1]])],
                    location='top_right')
    legend.label_text_color = "black"

    tap_tool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const url = source.data['Google Drive Link'][selected_index];
            window.open(url, "_blank");

            // Highlight the selected point and shade others
            for (let i = 0; i < source.data['alpha'].length; i++) {
                source.data['alpha'][i] = (i === selected_index) ? 1.0 : 0.1;
            }
            source.change.emit();
        }
    """))
    p.add_tools(tap_tool)

    hover = HoverTool(
        tooltips=[
            ('Log 10 Volume', '@Volume'),
            ('Slope', '@{Slope}'),
            ('Droplet', '@Droplet')
        ],
        renderers=[scatter]
    )
    p.add_tools(hover)
    p.outline_line_color = None
    p.border_fill_color = None
    p.background_fill_color = None
    return p

def distance_Vs_Volume_histogram(df):
    df = df.copy()
    distance_bins = [0, 1000, 2000, 3000,4000,5000,6000, float('inf')]
    distance_labels = ["0-1000", "1000-2000", "2000-3000", "3000-4000","4000-5000","5000-6000","6000+"]
    volume_bins = [3, 4, 5, 6, 7, 8]
    volume_labels = ["3-4", "4-5", "5-6", "6-7", "7-8"]
    df['distance_bin'] = pd.cut(df['distance_to_center'], bins=distance_bins, labels=distance_labels, right=False)
    df['volume_bin'] = pd.cut(df['log_Volume'], bins=volume_bins, labels=volume_labels, right=False)
    grouped = df.groupby(['distance_bin', 'volume_bin'], observed=True).size().unstack(fill_value=0)
    normalized_grouped = grouped.div(grouped.sum(axis=1), axis=0)
    source_data = {'distance_bin': distance_labels}
    for volume_label in volume_labels:
        source_data[volume_label] = normalized_grouped.get(volume_label, [0] * len(distance_labels))
    source = ColumnDataSource(data=source_data)
    colors = Category20[len(volume_labels)]  # Colors for the stacked bars
    p = standard_figure(x_range=distance_labels, title="Normalized Stacked Histogram: Distance vs. Log Volume",
                        toolbar_location=None, tools="")
    p.vbar_stack(volume_labels, x='distance_bin', width=0.6, color=colors, source=source,
                 legend_label=volume_labels)
    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = "Distance from Center"
    p.yaxis.axis_label = "Proportion"
    p.legend.title = "Log Volume"
    p.legend.orientation = "vertical"
    p.legend.location = "top_center"
    hover = HoverTool()
    hover.tooltips = [("Distance Bin", "@distance_bin"), ("Volume Bin", "$name"), ("Count", "@$name")]
    p.add_tools(hover)
    p.add_layout(p.legend[0], 'right')

    return p


def distance_Vs_occupide_histogram(df):
    df = df.copy()
    df = df[df['Count'] > 0]
    distance_bins = [0, 1000, 2000, 3000,4000,5000,6000, float('inf')]
    distance_labels = ["0-1000", "1000-2000", "2000-3000", "3000-4000","4000-5000","5000-6000","6000+"]
    volume_bins = [3, 4, 5, 6, 7, 8]
    volume_labels = ["3-4", "4-5", "5-6", "6-7", "7-8"]
    df['distance_bin'] = pd.cut(df['distance_to_center'], bins=distance_bins, labels=distance_labels, right=False)
    df['volume_bin'] = pd.cut(df['log_Volume'], bins=volume_bins, labels=volume_labels, right=False)
    grouped = df.groupby(['distance_bin', 'volume_bin'], observed=True).size().unstack(fill_value=0)
    normalized_grouped = grouped.div(grouped.sum(axis=1), axis=0)
    source_data = {'distance_bin': distance_labels}
    for volume_label in volume_labels:
        source_data[volume_label] = normalized_grouped.get(volume_label, [0] * len(distance_labels))
    source = ColumnDataSource(data=source_data)
    colors = Category20[len(volume_labels)]  # Colors for the stacked bars
    p = standard_figure(x_range=distance_labels, title="Normalized Stacked Histogram: Distance vs. Log Volume Occupied",
                        toolbar_location=None, tools="")
    p.vbar_stack(volume_labels, x='distance_bin', width=0.6, color=colors, source=source,
                 legend_label=volume_labels)
    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = "Distance from Center"
    p.yaxis.axis_label = "Proportion"
    p.legend.title = "Log Volume"
    p.legend.orientation = "vertical"
    p.legend.location = "top_center"
    hover = HoverTool()
    hover.tooltips = [("Distance Bin", "@distance_bin"), ("Volume Bin", "$name"), ("Count", "@$name")]
    p.add_tools(hover)
    p.add_layout(p.legend[0], 'right')

    return p


def distance_Vs_Volume_circle(df):
    df = df.copy()
    df['radius'] = (df['Area'] / math.pi) ** 0.5
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
    p = standard_figure(title='Distance to Center vs. Volume',
                        x_range=(0, 13000), y_range=(0, 13000))
    circle_center_x = (df['X'].min()+df['X'].max())/2
    circle_center_y = (df['Y'].min()+df['Y'].max())/2
    radius_values = [1000, 2000, 3000, 4000,5000,6000,6500*1.04]
    labels = ['0-1000', '1000-2000', '2000-3000', '3000-4000','4000-5000','5000-6000','6000+']
    for i, radius in enumerate(radius_values):
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius,
                 line_color="black", fill_color=None, alpha=0.5, line_width=LINE_WIDTH)
        if i < len(labels):  # Skip labeling for the largest circle
            label = Label(x=circle_center_x, y=circle_center_y + radius,
                          text=labels[i], text_align='center',
                          text_baseline='middle', text_font_style='bold', text_font_size='12pt')
            p.add_layout(label)
    legend_items = []
    colors = Category20[20]
    scatter_renderers = []
    grouped = df.groupby(['lower bin', 'upper bin'])
    for index, ((lower_bin, upper_bin), group) in enumerate(grouped):
        color = colors[index]
        source = ColumnDataSource(group)
        scatter = p.circle(x='X', y='Y', radius='radius', source=source, color=color, fill_alpha=0.5, line_width=LINE_WIDTH)
        scatter_renderers.append(scatter)
        legend_items.append(LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[scatter]))
    legend = Legend(items=legend_items, location='top_left')
    p.add_layout(legend)
    p.legend.click_policy = 'hide'
    hover = HoverTool(tooltips=[('Log Volume', '@log_Volume'),
                                 ('Droplet', '@Droplet'),
                                 ('Radius', '@radius')],
                      renderers=scatter_renderers)
    p.add_tools(hover)
    return p

def distance_Vs_occupide_circle(df):
    df = df.copy()
    df = df[df['Count'] > 0]
    df['radius'] = (df['Area'] / math.pi) ** 0.5
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
    p = standard_figure(title='Distance to Center vs. Volume Occupied',
                        x_range=(0, 13000), y_range=(0, 13000))
    circle_center_x = (df['X'].min()+df['X'].max())/2
    circle_center_y = (df['Y'].min()+df['Y'].max())/2
    radius_values = [1000, 2000, 3000, 4000,5000,6000,6500*1.04]
    labels = ['0-1000', '1000-2000', '2000-3000', '3000-4000','4000-5000','5000-6000','6000+']
    for i, radius in enumerate(radius_values):
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius,
                 line_color="black", fill_color=None, alpha=0.5, line_width=LINE_WIDTH)
        if i < len(labels):  # Skip labeling for the largest circle
            label = Label(x=circle_center_x, y=circle_center_y + radius,
                          text=labels[i], text_align='center',
                          text_baseline='middle', text_font_style='bold', text_font_size='12pt')
            p.add_layout(label)
    legend_items = []
    grouped = df.groupby(['lower bin', 'upper bin'])
    scatter_renderers = []
    colors = Category20[20]
    sources = []  # List to store all sources
    for index, ((lower_bin, upper_bin), group) in enumerate(grouped):
        color = colors[index]
        source = ColumnDataSource(group)
        sources.append(source)  # Add source to the list
        scatter = p.circle(x='X', y='Y', radius='radius', source=source, color=color, fill_alpha=0.5, line_width=LINE_WIDTH)
        legend_items.append(LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[scatter]))
        scatter_renderers.append(scatter)
    legend = Legend(items=legend_items, location='top_left')
    p.add_layout(legend)
    p.legend.click_policy = 'hide'
    hover = HoverTool(tooltips=[('Log Volume', '@log_Volume'),
                                 ('Droplet', '@Droplet'),
                                 ('Radius', '@radius')],
                      renderers=scatter_renderers)
    p.add_tools(hover)
    taptool = TapTool(callback=CustomJS(args=dict(sources=sources), code="""
        for (let source of sources) {
            const selected_index = source.selected.indices[0];
            if (selected_index != null) {
                const data = source.data;
                const url = data['Google Drive Link'][selected_index];
                window.open(url, "_blank");
                break;  // Open the first selected link and exit the loop
            }
        }
    """))
    p.add_tools(taptool)

    return p


def distance_Vs_Volume_colored_by_death_rate(df, data_dict,chip):
    df = df.copy()
    df['log_Volume'] = np.log10(df['Volume'])
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
    df['radius'] = (df['Area'] / math.pi) ** 0.5
    max_death_rate = []
    droplet_ids = []
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            window_size = 4
            mask = value['Count'] > 0
            value['log_count'] = value[mask]['Count'].apply(np.log)
            value['slope'] = value['log_count'].rolling(window_size).apply(
                lambda x: linregress(range(window_size), x)[0])
            if chip=='C1- CONTROL (no antibiotics)' or chip=='C5- CONTROL (no antibiotics)':
                max_death_rate.append(value['slope'].max())
            else:
                max_death_rate.append(value['slope'].min())
            droplet_ids.append(key)
    death_rates = pd.DataFrame({'Slope': max_death_rate, 'Droplet': droplet_ids})
    death_rates.dropna(subset=['Slope'], inplace=True)  # Drop rows with NaN values
    df = pd.merge(df, death_rates, on='Droplet')
    jet_palette = [RGB(*[int(255 * c) for c in cm.jet(i)[:3]]).to_hex() for i in range(256)]
    color_mapper = LinearColorMapper(palette=jet_palette, low=-2,
                                     high=2)
    if chip=='C1- CONTROL (no antibiotics)' or chip=='C5- CONTROL (no antibiotics)':
        title = 'Distance to Center vs. Volume Colored by Maximal Slope'
    else:
        title = 'Distance to Center vs. Volume Colored by Minimal Slope'
    p = standard_figure(title=title, match_aspect=True)
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "Y"
    circle_center_x = (df['X'].min()+df['X'].max())/2
    circle_center_y = (df['Y'].min()+df['Y'].max())/2
    for radius in [6500*1.04,6000,5000,4000, 3000, 2000, 1000]:
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius, line_color="black", fill_color=None,
                 alpha=0.5, line_width=LINE_WIDTH)
        label_text = f'{radius - 1000}-{radius}' if radius < 6500 else '6000+'
        label = Label(x=circle_center_x, y=circle_center_y + (radius - 1000) * 1.05, text=label_text, text_align='center',
                      text_baseline='middle', text_font_style='bold', text_font_size='12pt')
        p.add_layout(label)
    grouped = df.groupby(['lower bin', 'upper bin'])
    scatter_renderers = []
    sources = []
    checkbox_labels = []
    for (lower_bin, upper_bin), group in grouped:
        source = ColumnDataSource(group)
        sources.append(source)
        scatter = p.circle(x='X', y='Y', radius='radius', source=source, color={'field': 'Slope', 'transform': color_mapper}, fill_alpha=0.5, line_width=LINE_WIDTH)
        scatter_renderers.append(scatter)
        checkbox_labels.append(f'Bin {lower_bin}-{upper_bin}')
    checkbox_group = CheckboxGroup(labels=checkbox_labels, active=list(range(len(checkbox_labels))))
    checkbox_group.js_on_change('active',
                                CustomJS(args=dict(scatter_renderers=scatter_renderers, checkbox_group=checkbox_group),
                                         code="""
        for (let i = 0; i < scatter_renderers.length; i++) {
            scatter_renderers[i].visible = checkbox_group.active.includes(i);
        }
    """))
    hover = HoverTool(tooltips=[('Log Volume', '@log_Volume'), ('Slope', '@{Slope}'), ('Droplet', '@Droplet')],
                      renderers=scatter_renderers)
    p.add_tools(hover)
    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), title='Slope')
    p.renderers.append(color_bar)
    p.add_layout(color_bar, 'right')

    # Add TapTool with CustomJS callback
    taptool = TapTool(callback=CustomJS(args=dict(sources=sources), code="""
        for (let source of sources) {
            const selected_index = source.selected.indices[0];
            if (selected_index != null) {
                const data = source.data;
                const url = data['Google Drive Link'][selected_index];
                window.open(url, "_blank");
                break;  // Open the first selected link and exit the loop
            }
        }
    """))
    p.add_tools(taptool)

    layout_config = row(checkbox_group, p, sizing_mode='fixed', width=FIG_WIDTH, height=FIG_HEIGHT)
    return layout_config

def distance_Vs_Volume_colored_by_fold_change(df, data_dict):
    df = df.copy()
    df['log_Volume'] = np.log10(df['Volume'])
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
    df['radius'] = (df['Area'] / math.pi) ** 0.5
    fold_change = np.array([])
    Volume = np.array([])
    droplet_id = np.array([])
    min_fc = -10
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            if value['Count'].iloc[-4:].mean() == 0:
                fold_change = np.append(fold_change, np.nan)
                Volume = np.append(Volume, value['Volume'].iloc[0])
                droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
            else:
                fold_change = np.append(fold_change, value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])
                Volume = np.append(Volume, value['Volume'].iloc[0])
                droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
    fold_change = np.log2(fold_change)
    fold_change = np.where(np.isnan(fold_change), min_fc, fold_change)
    fold_changes = pd.DataFrame({'Volume': Volume, 'fold change': fold_change, 'Droplet': droplet_id})
    df = pd.merge(df, fold_changes, on='Droplet')
    jet_palette = [RGB(*[int(255 * c) for c in cm.jet(i)[:3]]).to_hex() for i in range(256)]
    color_mapper = LinearColorMapper(palette=jet_palette, low=-10,
                                     high=10)
    p = standard_figure(title='Distance to Center vs. Volume Colored by Fold Change',
                        match_aspect=True)
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "Y"
    circle_center_x = (df['X'].min()+df['X'].max())/2
    circle_center_y = (df['Y'].min()+df['Y'].max())/2
    for radius in [6500*1.04,6000,5000,4000, 3000, 2000, 1000]:
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius, line_color="black", fill_color=None,
                 alpha=0.5, line_width=LINE_WIDTH)
        label_text = f'{radius - 1000}-{radius}' if radius < 6500 else '6000+'
        label = Label(x=circle_center_x, y=circle_center_y + (radius - 1000) * 1.05, text=label_text, text_align='center',
                      text_baseline='middle', text_font_style='bold', text_font_size='12pt')
        p.add_layout(label)
    grouped = df.groupby(['lower bin', 'upper bin'])
    scatter_renderers = []
    checkbox_labels = []
    sources = []
    for (lower_bin, upper_bin), group in grouped:
        source = ColumnDataSource(group)
        sources.append(source)  # Add source to the list
        scatter = p.circle(x='X', y='Y', radius='radius', source=source, color={'field': 'fold change', 'transform': color_mapper}, fill_alpha=0.5, line_width=LINE_WIDTH)
        scatter_renderers.append(scatter)
        checkbox_labels.append(f'Bin {lower_bin}-{upper_bin}')
    checkbox_group = CheckboxGroup(labels=checkbox_labels, active=list(range(len(checkbox_labels))))
    checkbox_group.js_on_change('active',
                                CustomJS(args=dict(scatter_renderers=scatter_renderers, checkbox_group=checkbox_group),
                                         code="""
            for (let i = 0; i < scatter_renderers.length; i++) {
                scatter_renderers[i].visible = checkbox_group.active.includes(i);
            }
        """))
    hover = HoverTool(tooltips=[('Log Volume', '@log_Volume'), ('Fold Change', '@{fold change}'), ('Droplet', '@Droplet')],
                      renderers=scatter_renderers)
    p.add_tools(hover)
    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), title='fold change')
    p.renderers.append(color_bar)
    p.add_layout(color_bar, 'right')

    # Add TapTool with CustomJS callback
    taptool = TapTool(callback=CustomJS(args=dict(sources=sources), code="""
        for (let source of sources) {
            const selected_index = source.selected.indices[0];
            if (selected_index != null) {
                const data = source.data;
                const url = data['Google Drive Link'][selected_index];
                window.open(url, "_blank");
                break;  // Open the first selected link and exit the loop
            }
        }
    """))
    p.add_tools(taptool)
    layout_config = row(checkbox_group, p, sizing_mode='fixed', width=FIG_WIDTH, height=FIG_HEIGHT)
    return layout_config

def bins_volume_Vs_distance(data_dict,chip):
    volumes = []
    distances = []
    droplet_ids = []
    fold_changes = []
    death_rates=[]
    google_drive_urls = []
    min_fc = -10
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            volumes.append(value['log_Volume'].iloc[0])
            distances.append(value['distance_to_center'].iloc[0])
            droplet_ids.append(key)
            google_drive_urls.append(value['Google Drive Link'].iloc[0])
            if value['Count'].iloc[-4:].mean() == 0:
                fold_changes.append(np.nan)
            else:
                fold_changes.append(value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])
            window_size = 4
            mask = value['Count'] > 0
            value['log_count'] = value[mask]['Count'].apply(np.log10)
            value['slope'] = value['log_count'].rolling(window_size).apply(lambda x: linregress(range(window_size), x)[0])
            if chip=='C1- CONTROL (no antibiotics)' or chip=='C5- CONTROL (no antibiotics)':
                death_rates.append(value['slope'].max())
            else:
                death_rates.append(value['slope'].min())
    df = pd.DataFrame({'Volume': volumes, 'Distance': distances, 'Droplet': droplet_ids, 'Fold Change': fold_changes, 'Slope': death_rates, 'Google Drive Link': google_drive_urls})
    df['Fold Change'] = np.log2(df['Fold Change'])
    df['Fold Change'] = np.where(np.isnan(df['Fold Change']), min_fc, df['Fold Change'])
    df['upper_volume_bin'] = df['Volume'].apply(math.ceil)
    df['lower_volume_bin'] = df['Volume'].apply(math.floor)
    distance_bins = [0, 1000, 2000, 3000,4000,5000,6000, float('inf')]
    df['lower_distance_bin'] = pd.cut(df['Distance'], bins=distance_bins, labels=distance_bins[:-1], right=False)
    df['upper_distance_bin'] = pd.cut(df['Distance'], bins=distance_bins, labels=distance_bins[1:], right=False)
    df['mean_fold_change'] = np.nan
    df['mean_slope'] = np.nan
    grouped=df.groupby(['lower_volume_bin', 'upper_volume_bin', 'lower_distance_bin', 'upper_distance_bin'], observed=True)
    for (lower_volume_bin, upper_volume_bin, lower_distance_bin, upper_distance_bin), group in grouped:
        mean_fold_change = group['Fold Change'].mean()
        mean_slope = group['Slope'].mean()
        df['Slope'] = df['Slope'].fillna(mean_slope)
        df.loc[(df['lower_volume_bin'] == lower_volume_bin) & (df['upper_volume_bin'] == upper_volume_bin) & (df['lower_distance_bin'] == lower_distance_bin) & (df['upper_distance_bin'] == upper_distance_bin), 'mean_fold_change'] = mean_fold_change
        df.loc[(df['lower_volume_bin'] == lower_volume_bin) & (df['upper_volume_bin'] == upper_volume_bin) & (df['lower_distance_bin'] == lower_distance_bin) & (df['upper_distance_bin'] == upper_distance_bin), 'mean_slope'] = mean_slope
    def create_plot(results, y_axis_label, y_column,points_column):
        if y_axis_label == 'Mean Fold Change':
            title = 'Distance Bin vs. Mean Fold Change'
        elif y_axis_label == 'Mean Slope' and chip=='C1- CONTROL (no antibiotics)' or chip=='C5- CONTROL (no antibiotics)':
            title = 'Distance Bin vs. Maximal Slope'
        elif y_axis_label == 'Mean Slope':
            title = 'Distance Bin vs. Minimal Slope'
        p = standard_figure(title=title, x_axis_label='Distance Bin', y_axis_label=y_axis_label)
        volume_bins = results['lower_volume_bin'].sort_values().unique()
        colors = Category20[len(volume_bins)]
        legend_items = []
        sources=[]
        scatter_renderers = []
        for i, volume_bin in enumerate(volume_bins):
            volume_bin_data = results[results['lower_volume_bin'] == volume_bin]
            volume_bin_data = volume_bin_data.sort_values('lower_distance_bin')
            source = ColumnDataSource(volume_bin_data)
            sources.append(source)
            line = p.line(x='lower_distance_bin', y=y_column, source=source, line_width=LINE_WIDTH_EMPHASIS, color=colors[i])
            scatter = p.scatter(x=jitter('lower_distance_bin', width=JITTER_WIDTH, range=p.x_range), y=points_column, source=source, color=colors[i], fill_alpha=0.5, size=SCATTER_SIZE)
            scatter_renderers.append(scatter)
            violin_renderers = []
            for lower_distance_bin in volume_bin_data['lower_distance_bin'].unique():
                data = volume_bin_data[volume_bin_data['lower_distance_bin'] == lower_distance_bin]
                if len(data) < 2:
                    continue
                kde = gaussian_kde(data[points_column]+np.random.normal(0, 1e-6, len(data[points_column])))
                y = np.linspace(data[points_column].min(), data[points_column].max(), 1000)
                density = kde(y)
                density=density/density.max()*VIOLIN_MAX_WIDTH
                x_centered = lower_distance_bin
                x_combined = np.concatenate([x_centered + density, x_centered - density[::-1]])
                y_combined = np.concatenate([y, y[::-1]])
                violin=p.patch(x=x_combined, y=y_combined, fill_color=colors[i],line_color=colors[i], fill_alpha=0.3)
                violin_renderers.append(violin)
            legend_item = LegendItem(label=f'Volume Bin {volume_bin}-{volume_bin + 1}', renderers=[line,scatter]+violin_renderers)
            legend_items.append(legend_item)
        hover = HoverTool(tooltips=[
            ('Distance Bin', '@lower_distance_bin'),
            (points_column, f'@{{{points_column}}}'),
            ('Volume Bin', '@lower_volume_bin')
        ], renderers=scatter_renderers)
        p.add_tools(hover)
        legend = Legend(items=legend_items, location='top_left')
        p.add_layout(legend, 'right')
        p.legend.click_policy = 'hide'
        taptool = TapTool(callback=CustomJS(args=dict(sources=sources), code="""
            for (let source of sources) {
                const selected_index = source.selected.indices[0];
                if (selected_index != null) {
                    const data = source.data;
                    const url = data['Google Drive Link'][selected_index];
                    window.open(url, "_blank");
                    break;  // Open the first selected link and exit the loop
                }
            }
        """))
        p.add_tools(taptool)
        return p

    plot_fold_change = create_plot(df, 'Mean Fold Change', 'mean_fold_change','Fold Change')
    plot_death_rate = create_plot(df, 'Mean Slope', 'mean_slope','Slope')
    return row(plot_fold_change, plot_death_rate)

def FC_vs_density(data_dict):
    fold_change = np.array([])
    density=np.array([])
    Volume = np.array([])
    droplet_id = np.array([])
    google_drive_url = np.array([])  # Array to store Google Drive URLs
    min_fc = -10
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            if value['Count'].iloc[-4:].mean() == 0:
                fold_change = np.append(fold_change, np.nan)
                Volume = np.append(Volume, value['Volume'].iloc[0])
                density=np.append(density,value['Count'].iloc[0]/value['Volume'].iloc[0])
                droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
                google_drive_url = np.append(google_drive_url, value['Google Drive Link'].iloc[0])  # Add URL
            else:
                fold_change = np.append(fold_change, value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])
                Volume = np.append(Volume, value['Volume'].iloc[0])
                density = np.append(density, value['Count'].iloc[0] / value['Volume'].iloc[0])
                droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
                google_drive_url = np.append(google_drive_url, value['Google Drive Link'].iloc[0])  # Add URL
    fold_change = np.log2(fold_change)
    fold_change = np.where(np.isnan(fold_change), min_fc, fold_change)
    df = pd.DataFrame(
        {'Volume': np.log10(Volume), 'fold change': fold_change,'Density':np.log2(density), 'Droplet': droplet_id, 'Google Drive Link': google_drive_url})
    p = standard_figure(title='Log2 Fold Change vs. Log2 Density colored by Volume',
                        x_axis_label='Log2 Density', y_axis_label='Log2 Fold Change')
    source = ColumnDataSource(df)
    jet_palette = [RGB(*[int(255 * c) for c in cm.jet(i)[:3]]).to_hex() for i in range(256)]
    color_mapper = LinearColorMapper(palette=jet_palette, low=3, high=7.5)
    p.scatter(x='Density', y='fold change', source=source,
              color=linear_cmap('Volume', jet_palette, 3, 7.5), alpha=0.8, size=SCATTER_SIZE)
    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), title='Log10 Volume')
    p.renderers.append(color_bar)
    p.add_layout(color_bar, 'right')
    hover = HoverTool(tooltips=[('Log2 Density', '@Density'), ('Log2 Fold Change', '@{fold change}'), ('Droplet', '@Droplet'),('Log10 Volume', '@Volume')])
    p.add_tools(hover)
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const url = source.data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    p.add_tools(taptool)
    return p

def FC_vs_Volume(data_dict):
    fold_change = np.array([])
    density=np.array([])
    Volume = np.array([])
    droplet_id = np.array([])
    google_drive_url = np.array([])  # Array to store Google Drive URLs
    min_fc = -10
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue
        else:
            if value['Count'].iloc[-4:].mean() == 0:
                fold_change = np.append(fold_change, np.nan)
                Volume = np.append(Volume, value['Volume'].iloc[0])
                density=np.append(density,value['Count'].iloc[0]/value['Volume'].iloc[0])
                droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
                google_drive_url = np.append(google_drive_url, value['Google Drive Link'].iloc[0])  # Add URL
            else:
                fold_change = np.append(fold_change, value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])
                Volume = np.append(Volume, value['Volume'].iloc[0])
                density = np.append(density, value['Count'].iloc[0] / value['Volume'].iloc[0])
                droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
                google_drive_url = np.append(google_drive_url, value['Google Drive Link'].iloc[0])  # Add URL
    fold_change = np.log2(fold_change)
    fold_change = np.where(np.isnan(fold_change), min_fc, fold_change)
    df = pd.DataFrame(
        {'Volume': np.log2(Volume), 'fold change': fold_change,'Density':np.log2(density), 'Droplet': droplet_id, 'Google Drive Link': google_drive_url})
    p = standard_figure(title='Log2 Fold Change vs. Log2 Volume colored by Density',
                        x_axis_label='Log2 Volume', y_axis_label='Log2 Fold Change')
    source = ColumnDataSource(df)
    jet_palette = [RGB(*[int(255 * c) for c in cm.jet(i)[:3]]).to_hex() for i in range(256)]
    color_mapper = LinearColorMapper(palette=jet_palette, low=-12, high=-4)
    p.scatter(x='Volume', y='fold change', source=source,
              color=linear_cmap('Density', jet_palette, -12, -4), alpha=0.8, size=SCATTER_SIZE)
    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), title='Log2 Density')
    p.renderers.append(color_bar)
    p.add_layout(color_bar, 'right')
    hover = HoverTool(tooltips=[('Log2 Density', '@Density'), ('Log2 Fold Change', '@{fold change}'), ('Droplet', '@Droplet'),('Log2 Volume', '@Volume')])
    p.add_tools(hover)
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const url = source.data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    p.add_tools(taptool)
    return p
def dashborde():
    chips = split_data_to_chips()
    # keys_to_extract = ['C1 - Control', 'C5 - Control']
    # chips = {k: chips[k] for k in keys_to_extract if k in chips}
    initial_densities = initial_stats(chips)
    for key, value in initial_densities.items():
        density=value['Count'].sum()/value['Volume'].sum()
        initial_densities[key]=density
    for key, value in chips.items():
        chips[key] = find_droplet_location(value)
        chips[key] = chips[key][chips[key]['log_Volume'] >= 3]
    initial_data = initial_stats(chips)
    layouts={}
    for key, value in initial_data.items():
        chip, experiment_time, time_steps = get_slice(chips, key)
        stats_box_plot=stats_box(value, experiment_time, time_steps, key)
        droplets_histogram_plot=droplet_histogram(value)
        Initial_Density_Vs_Volume_plot,volume=Initial_Density_Vs_Volume(value, initial_densities[key])
        N0_Vs_Volume_plot=N0_Vs_Volume(value,volume)
        Fraction_in_each_bin_plot=Distribution_comparison(chip, experiment_time)
        growth_curves_plot=growth_curves(chip)
        normalize_Max_growth_curves_plot=normalize_growth_curves(chip)
        normalize_first_timepoint_growth_curves_plot=normalize_growth_curves_first_timepoint(chip)
        fold_change_plot=fold_change(chip,volume)
        last_4_hours_average_plot=last_4_hours_average(chip,volume)
        death_rate_by_droplets_plot=death_rate_by_droplets(chip,key)
        death_rate_by_bins_plot=death_rate_by_bins(chip)
        distance_Vs_Volume_histogram_plot=distance_Vs_Volume_histogram(value)
        distance_Vs_occupide_histogram_plot=distance_Vs_occupide_histogram(value)
        distance_Vs_Volume_circle_plot=distance_Vs_Volume_circle(value)
        distance_Vs_occupide_circle_plot=distance_Vs_occupide_circle(value)
        distance_Vs_Volume_colored_by_death_rate_plot=distance_Vs_Volume_colored_by_death_rate(value, chip,key)
        distance_Vs_Volume_colored_by_fold_change_plot=distance_Vs_Volume_colored_by_fold_change(value, chip)
        bins_volume_Vs_distance_plot=bins_volume_Vs_distance(chip,key)
        FC_vs_density_plot=FC_vs_density(chip)
        FC_vs_Volume_plot=FC_vs_Volume(chip)
        layout = column(
                        stats_box_plot,
                        row(droplets_histogram_plot, N0_Vs_Volume_plot),
                        row(Initial_Density_Vs_Volume_plot,Fraction_in_each_bin_plot),
                        growth_curves_plot,
                        normalize_Max_growth_curves_plot,
                        normalize_first_timepoint_growth_curves_plot,
                        #row(death_rate_by_droplets_plot),
                        row(death_rate_by_droplets_plot, death_rate_by_bins_plot),
                        row(fold_change_plot,last_4_hours_average_plot),
                        # row(fold_change_plot),
                        row(distance_Vs_Volume_histogram_plot, distance_Vs_occupide_histogram_plot),
                        row(distance_Vs_Volume_circle_plot, distance_Vs_occupide_circle_plot),
                        row(distance_Vs_Volume_colored_by_death_rate_plot, distance_Vs_Volume_colored_by_fold_change_plot,spacing=75),
                        bins_volume_Vs_distance_plot,
                        row(FC_vs_density_plot,FC_vs_Volume_plot)
                        )
        layouts[key]=layout
    return layouts


def create_dashboard():
    layouts = dashborde()
    output_file('dashboard.html')
    select = Select(title="Select Chip", options=list(layouts.keys()), value=list(layouts.keys())[0])
    all_layouts_column = column(*[layout for layout in layouts.values()], name="all_layouts")
    for layout in all_layouts_column.children:
        layout.visible = False
    layouts[select.value].visible = True
    select.js_on_change('value',
                        CustomJS(args=dict(layouts=layouts, all_layouts_column=all_layouts_column, select=select), code="""
        // Hide all layouts
        for (const layout of all_layouts_column.children) {
            layout.visible = false;
        }
        // Show the selected layout
        layouts[select.value].visible = true;
    """))

    show(column(select, all_layouts_column))

if __name__ == '__main__':
    create_dashboard()

