import math
import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# --- Bokeh Imports for Visualization ---
from bokeh.io import output_file
from bokeh.models import Div, HoverTool, TapTool, ColorBar, CheckboxGroup, LogTicker, LinearColorMapper, \
    BasicTicker, Legend, LegendItem, BoxAnnotation, Span, Whisker, Label, Spacer, \
    ColumnDataSource, CDSView, Select, CustomJS, GlyphRenderer, FixedTicker
from bokeh.models.formatters import CustomJSTickFormatter
from bokeh.layouts import column, row
from bokeh.palettes import Category20, RGB, RdBu, bokeh, Viridis256, linear_palette
from bokeh.plotting import figure, show, markers
from bokeh.transform import linear_cmap, jitter, dodge
import colorcet as cc  # External library for perceptually uniform colormaps

# --- Statistical Libraries ---
from scipy.stats import linregress, permutation_test, gaussian_kde
from matplotlib import cm  # Used for colormap conversions
from statsmodels.stats.multitest import multipletests


def read_data():
    """
    Opens a system file dialog to select a CSV file.

    Returns:
        pd.DataFrame: The loaded data if a file is selected.
        None: If no file is selected.
    """
    # Hide the main tkinter window
    Tk().withdraw()
    file_path = askopenfilename()

    if file_path:
        # encoding='ISO-8859-1' is used to handle potential special characters in the CSV
        return pd.read_csv(file_path, encoding='ISO-8859-1')
    else:
        return None


def split_data_to_chips():
    """
    Loads data and splits it into a dictionary based on the 'Slice' column.
    Each 'Slice' typically represents a specific Chip or Experiment run.

    Returns:
        dict: Keys are Slice IDs, Values are DataFrames containing data for that slice.
    """
    data = read_data()
    # Create a dict comprehension to split the dataframe by 'Slice'
    # reset_index(drop=True) ensures a clean index for each sub-dataframe
    chips = {slice_id: df.reset_index(drop=True) for slice_id, df in data.groupby('Slice')}
    return chips


def initial_stats(data):
    """
    Filters the dataset to extract only the initial timepoint (Time = 0).

    Args:
        data (dict): Dictionary of DataFrames (output of split_data_to_chips).

    Returns:
        dict: Dictionary containing only rows where 'time' == 0 for each chip.
    """
    filtered_chips = {slice_id: df[df['time'] == 0].reset_index(drop=True) for slice_id, df in data.items()}
    return filtered_chips


def get_slice(data, slice_id):
    """
    Extracts specific metadata and structures the data for a single chip.

    Args:
        data (dict): The dictionary of chip dataframes.
        slice_id (str): The key for the specific chip to analyze.

    Returns:
        tuple:
            - chip (dict): Dictionary where Keys are 'Droplet' IDs and Values are DataFrames for that droplet.
            - experiment_time (float): The maximum time recorded in this slice.
            - time_steps (float): The interval between timepoints.
    """
    slice_data = data[slice_id]

    # Identify the maximum time point in the experiment
    experiment_time = slice_data['time'].max()

    # Calculate the time step interval (assumes uniform time steps)
    # Sorts unique time values and finds the difference between the first two
    time_steps = np.diff(sorted(slice_data['time'].unique()))[0]

    # Group the slice data by 'Droplet' to facilitate per-droplet analysis (growth curves, etc.)
    chip = {droplet_id: df.reset_index(drop=True) for droplet_id, df in slice_data.groupby('Droplet')}

    return chip, experiment_time, time_steps


def stats_box(df, time, max_step, chip_name):
    """
    Creates a styled text box (Div) displaying summary statistics for a specific chip.

    Args:
        df (pd.DataFrame): Dataframe containing droplet data for the chip.
        time (float): The total duration of the experiment.
        max_step (float): The time step interval.
        chip_name (str): The identifier of the chip.

    Returns:
        bokeh.layouts.column: A layout containing the stats Div.
    """
    # Calculate statistics using log-transformed volume for scale invariance
    volume = np.log10(df['Volume'].sum())
    mean = np.log10(df['Volume'].mean())
    std = np.log10(df['Volume'].std())
    bacteria_pool = df['Count'].sum()

    # Calculate chip density: Total Bacteria / Total Volume
    chip_density = np.log10(bacteria_pool / 10 ** volume)

    # HTML formatting for the text display
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
    # Inline CSS styling for the statistics box (borders, shadows, fonts)
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
    """
    Generates a histogram comparing Total Droplets vs. Occupied Droplets (Count > 0)
    across logarithmic volume bins.
    """
    # Create logarithmic bins from 10^3 to 10^8
    bins = np.logspace(3, 8, num=16)

    hist = figure(x_axis_type='log',
                  x_axis_label='Volume', y_axis_label='Frequency', output_backend="webgl")

    # Font styling
    hist.xaxis.axis_label_text_font_size = "16pt"
    hist.yaxis.axis_label_text_font_size = "16pt"
    hist.xaxis.major_label_text_font_size = "14pt"
    hist.yaxis.major_label_text_font_size = "14pt"

    # Calculate histogram data
    hist_data = np.histogram(df['Volume'], bins=bins)
    hist_data_occupied = np.histogram(df[df['Count'] > 0]['Volume'], bins=bins)

    # Create ColumnDataSource for plotting quads
    source = ColumnDataSource(data=dict(
        top=hist_data[0],
        bottom=np.zeros_like(hist_data[0]),
        left=bins[:-1],
        right=bins[1:],
        top_occupied=hist_data_occupied[0]
    ))
    view = CDSView()

    # Plot Total Droplets (Gray)
    hist.quad(top='top', bottom='bottom', left='left', right='right',
              color='gray', alpha=0.5, legend_label='Total Droplets', source=source, view=view)

    # Plot Occupied Droplets (Blue, overlay)
    hist.quad(top='top_occupied', bottom='bottom', left='left', right='right',
              color='blue', alpha=0.1, legend_label='Occupied Droplets', source=source, view=view)

    # Legend styling
    hist.legend.label_text_font_size = "14pt"
    hist.legend.glyph_width = 40
    hist.legend.glyph_height = 20
    hist.legend.spacing = 10
    hist.legend.padding = 10

    # Calculate Occupancy Statistics
    droplet_num = int(df['Volume'].count())
    occupied_droplets = int(df[df['Count'] > 0]['Volume'].count())
    occupancy_rate = occupied_droplets / droplet_num

    stats_text = f"Droplets count: {droplet_num}<br>Occupied Droplets: {occupied_droplets}<br>Occupancy Rate: {occupancy_rate:.2%}"

    stats_div = Div(text=stats_text, width=400, height=100)
    # Apply standard CSS style to stats box
    stats_div.styles = {'text-align': 'center', 'margin': '10px auto', 'font-size': '12pt',
                        'font-family': 'Arial, sans-serif', 'color': 'black', 'background-color': 'lightgray',
                        'border': '1px solid black', 'padding': '20px', 'box-shadow': '5px 5px 5px 0px lightgray',
                        'border-radius': '10px', 'line-height': '1.5em', 'font-weight': 'bold',
                        'white-space': 'pre-wrap', 'word-wrap': 'break-word', 'overflow-wrap': 'break-word',
                        'text-overflow': 'ellipsis', 'hyphens': 'auto'}

    combined_plot = column(hist, stats_div)
    return combined_plot


def N0_Vs_Volume(df, Vc):
    """
    Plots Initial Bacterial Count (N0) against Droplet Volume on a log-log scale.
    Calculates and displays a linear regression line.

    Args:
        df: Dataframe containing 'Volume' and 'Count'.
        Vc: Critical Volume (calculated elsewhere, used for filtering regression).
    """
    source = ColumnDataSource(df)
    view = CDSView()

    scatter = figure(x_axis_type='log', y_axis_type='log',
                     x_axis_label='Volume', y_axis_label='N0', output_backend="webgl")

    # Axis styling
    scatter.xaxis.axis_label_text_font_size = "16pt"
    scatter.yaxis.axis_label_text_font_size = "16pt"
    scatter.xaxis.major_label_text_font_size = "14pt"
    scatter.yaxis.major_label_text_font_size = "14pt"

    scatter_renderer = scatter.scatter('Volume', 'Count', source=source, view=view, color='gray', alpha=1,
                                       legend_label='N0 vs. Volume')

    # Add HoverTool to show details
    hover = HoverTool(tooltips=[('Volume', '@Volume'), ('Count', '@Count'), ('Droplet ID', '@Droplet')],
                      renderers=[scatter_renderer])
    scatter.add_tools(hover)

    # Add TapTool: Opens the Google Drive link for the specific droplet image when clicked
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)

    # Legend styling
    scatter.legend.label_text_font_size = "14pt"
    scatter.legend.glyph_width = 40
    scatter.legend.glyph_height = 20
    scatter.legend.spacing = 10
    scatter.legend.padding = 10

    # Linear Regression Calculation
    # Filter for non-zero counts and volumes above the critical threshold Vc
    filtered_df = df[df['Count'] > 0]
    filtered_df = filtered_df[filtered_df['Volume'] >= Vc]

    x = np.log10(filtered_df['Volume'])
    y = np.log10(filtered_df['Count'])
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Generate regression line points
    x_values = np.linspace(min(df['Volume']), max(df['Volume']), 100)
    y_values = 10 ** (intercept + slope * np.log10(x_values))

    stats_text = f'y = {slope:.2f}x + {intercept:.2f}<br>R² value: {r_value ** 2:.2f}'

    # Plot regression line
    scatter.line(x_values, y_values, color='red', legend_label='Linear Regression', line_width=3)

    stats_div = Div(text=stats_text, width=400, height=100)
    stats_div.styles = {'text-align': 'center', 'margin': '10px auto', 'font-size': '12pt',
                        'font-family': 'Arial, sans-serif', 'color': 'black', 'background-color': 'lightgray',
                        'border': '1px solid black', 'padding': '20px', 'box-shadow': '5px 5px 5px 0px lightgray',
                        'border-radius': '10px', 'line-height': '1.5em', 'font-weight': 'bold',
                        'white-space': 'pre-wrap', 'word-wrap': 'break-word', 'overflow-wrap': 'break-word',
                        'text-overflow': 'ellipsis', 'hyphens': 'auto'}

    combined_plot = column(scatter, stats_div)
    return combined_plot


def Initial_Density_Vs_Volume(df, initial_density):
    """
    Plots Initial Density (Count/Volume) against Volume.
    Calculates a rolling mean to determine at what volume the density converges to the global average.

    Args:
        df: Dataframe with droplet data.
        initial_density: The calculated global initial density of the chip.

    Returns:
        tuple: (Bokeh Plot, closest_volume)
        where closest_volume is the volume at which local density converges to global density.
    """
    df['initial density'] = df['Count'] / df['Volume']
    source = ColumnDataSource(df)
    view = CDSView()

    scatter = figure(x_axis_type='log', y_axis_type='log',
                     x_axis_label='Volume (μm³)', y_axis_label='Initial Density (pixels/μm³)', output_backend="webgl")

    # Axis styling
    scatter.xaxis.axis_label_text_font_size = "16pt"
    scatter.yaxis.axis_label_text_font_size = "16pt"
    scatter.xaxis.major_label_text_font_size = "14pt"
    scatter.yaxis.major_label_text_font_size = "14pt"

    scatter_renderer = scatter.scatter('Volume', 'initial density', source=source, view=view, color='gray', alpha=1)

    hover = HoverTool(
        tooltips=[('Volume', '@Volume'), ('Initial Density', '@{initial density}'), ('Droplet ID', '@Droplet')],
        renderers=[scatter_renderer])
    scatter.add_tools(hover)

    # TapTool for opening Google Drive links
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const data = source.data;
            const url = data['Google Drive Link'][selected_index];
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)

    # --- Rolling Mean Calculation ---
    filtered_sorted_df = df[df['initial density'] > 0].sort_values(by='Volume').reset_index()
    log_density = np.log10(filtered_sorted_df['initial density'])

    # Calculate rolling mean with window of 100 droplets
    rolling_mean = log_density.rolling(window=100, min_periods=1).mean()

    # Plot rolling mean (Red line)
    scatter.line(filtered_sorted_df['Volume'], 10 ** rolling_mean, color='red', line_width=3)

    # Plot global initial density (Black horizontal line)
    scatter.line([min(df['Volume']), max(df['Volume'])], [initial_density, initial_density], color='black',
                 line_width=3)

    # --- Convergence Detection ---
    # Find the volume where the rolling mean gets close to the global initial density
    convergence_window = 2
    tolerance = 0.05
    differences = np.abs(1 - (10 ** rolling_mean / initial_density))

    closest_index = differences.idxmin()  # Default fallback
    for i in range(len(differences) - convergence_window):
        window_mean_diff = differences.iloc[i:i + convergence_window].mean()
        if window_mean_diff <= tolerance:
            closest_index = i + convergence_window // 2
            break

    closest_point = filtered_sorted_df.loc[closest_index]
    closest_volume = closest_point['Volume']

    # Mark the convergence volume (Vertical dashed line)
    vline = Span(location=closest_volume, dimension='height', line_color='black', line_dash='dashed', line_width=2)
    scatter.add_layout(vline)
    scatter.renderers.append(vline)

    # Invisible line for legend entry of Vc
    invisible_line = scatter.line([0], [0], color='black', line_dash='dashed', line_width=2)

    legend = Legend(items=[
        LegendItem(label='Initial density vs. Volume', renderers=[scatter.renderers[0]]),
        LegendItem(label='Rolling Mean', renderers=[scatter.renderers[1]]),
        LegendItem(label='Initial Density', renderers=[scatter.renderers[2]]),
        LegendItem(label='Vc', renderers=[invisible_line])
    ], location='top_right')

    legend.label_text_font_size = "14pt"
    legend.glyph_width = 40
    legend.glyph_height = 20
    legend.spacing = 10
    legend.padding = 10
    scatter.add_layout(legend)

    return scatter, closest_volume


def Distribution_comparison(dic, time):
    """
    Compares the distribution of Droplet Volumes weighted by bacterial count
    at the start (T=0) vs. the end of the experiment.

    Uses Kernel Density Estimation (KDE) to visualize how the "biomass" shifts
    between droplet sizes over time.

    Args:
        dic (dict): Dictionary of droplet DataFrames.
        time (float): The final timepoint to compare against T=0.

    Returns:
        bokeh.plotting.figure: Area plot comparing the two distributions.
    """
    # Combine all droplet data into one DataFrame
    combined_df = pd.concat(dic.values(), ignore_index=True)
    combined_df = combined_df[combined_df['Count'] >= 0]

    # Separate Start and End data
    df_start = combined_df[combined_df['time'] == 0]
    df_end = combined_df[combined_df['time'] == time]

    # Extract Log10 Volumes and Weights (Bacterial Counts)
    # We weight the KDE by 'Count' to see where the bacteria *are*, not just where the droplets are.
    vals_start = np.log10(df_start['Volume'].values)
    weights_start = df_start['Count'].values

    vals_end = np.log10(df_end['Volume'].values)
    weights_end = df_end['Count'].values

    # Define grid for KDE evaluation
    min_x = min(vals_start.min(), vals_end.min())
    max_x = max(vals_start.max(), vals_end.max())
    x_grid = np.linspace(min_x, max_x, 200)

    # Calculate KDEs
    kde_start = gaussian_kde(vals_start, weights=weights_start)
    kde_end = gaussian_kde(vals_end, weights=weights_end)

    y_start = kde_start(x_grid)
    y_end = kde_end(x_grid)

    # Plotting
    p = figure(
        x_axis_label='Volume (log10 μm³)',
        y_axis_label='Relative abundance (density)',
        output_backend="webgl",
        width=800,
        height=600,
        y_range=(0, 1),
    )
    # Styling fonts
    p.xaxis.axis_label_text_font_size = "24pt"
    p.yaxis.axis_label_text_font_size = "24pt"
    p.xaxis.major_label_text_font_size = "20pt"
    p.yaxis.major_label_text_font_size = "20pt"

    # Draw Start Distribution (Grey)
    p.varea(x=x_grid, y1=0, y2=y_start, fill_color="grey", fill_alpha=0.3, legend_label="Start (T=0)")
    p.line(x=x_grid, y=y_start, line_color="black", line_width=3, line_dash="dashed", legend_label="Start (T=0)")

    # Draw End Distribution (Black)
    p.varea(x=x_grid, y1=0, y2=y_end, fill_color="black", fill_alpha=0.5, legend_label=f"End (T={time})")
    p.line(x=x_grid, y=y_end, line_color="black", line_width=3, legend_label=f"End (T={time})")

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    return p


def Fraction_in_each_bin(dic, time):
    """
    Calculates what percentage of the total population resides in each Volume Bin (10^3-10^4, 10^4-10^5, etc.).
    Compares Start vs. End fractions using a bar chart.
    """
    combined_df = pd.concat(dic.values(), ignore_index=True)
    combined_df = combined_df[(combined_df['time'] == 0) | (combined_df['time'] == time)]

    # Binning Logic: Floor and Ceil of Log10 Volume
    combined_df['bottom_bin'] = np.log10(combined_df['Volume']).apply(math.floor)
    combined_df['top_bin'] = np.log10(combined_df['Volume']).apply(math.ceil)

    # Group everything larger than 10^6 into a single "> 6" bin
    combined_df['bottom_bin'] = np.where(combined_df['bottom_bin'] >= 6, 6, combined_df['bottom_bin'])
    combined_df['top_bin'] = np.where(combined_df['top_bin'] >= 6, 6, combined_df['top_bin'])

    # Calculate Totals
    start_total = combined_df[combined_df['time'] == 0]['Count'].sum()
    end_total = combined_df[combined_df['time'] == time]['Count'].sum()
    ratio = end_total / start_total  # Total Metapopulation Growth Ratio

    # Aggregate counts per bin
    bins = combined_df.groupby(['bottom_bin', 'top_bin', 'time'])['Count'].sum().unstack().fillna(0)

    # Calculate Percentages
    bins['start fraction'] = bins[0] / start_total * 100
    bins['end fraction'] = bins[time] / end_total * 100

    # Formatting labels
    bin_labels = [
        "> 6" if int(start) == 6 and int(end) == 6 else f"{int(start)}–{int(end)}"
        for start, end in bins.index
    ]

    # Bar Chart
    p = figure(
        title='Fraction of Population in Each Bin at Start and End of the experiment',
        x_range=bin_labels,
        y_axis_label='Fraction of Population (%)',
        x_axis_label='Volume Bins (log10 μm³)',
        output_backend="webgl",
        y_range=(0, 100)
    )

    bar_width = 0.4
    # Start Bars (Gray, shifted left)
    p.vbar(
        x=dodge('x', -bar_width / 2, range=p.x_range),
        top='start fraction', width=bar_width,
        source={'x': bin_labels, 'start fraction': bins['start fraction']},
        color='gray', legend_label='Start'
    )
    # End Bars (Black, shifted right)
    p.vbar(
        x=dodge('x', bar_width / 2, range=p.x_range),
        top='end fraction', width=bar_width,
        source={'x': bin_labels, 'end fraction': bins['end fraction']},
        color='black', legend_label='End'
    )

    # Add labels showing the percentage change per bin
    for i, label in enumerate(bin_labels):
        start_val = bins['start fraction'].iloc[i]
        end_val = bins['end fraction'].iloc[i]
        delta = end_val - start_val
        sign = "+" if delta >= 0 else "–"
        text = f"{sign}{abs(delta):.1f}%"
        y_pos = max(start_val, end_val) + 3
        p.add_layout(
            Label(x=i + 0.5, y=y_pos, text=text, text_align='center', text_font_size='10pt', text_font_style='bold'))

    # Add Metapopulation Growth Summary
    if ratio >= 1:
        ratio_text = f"Metapopulation growth: +{(ratio - 1) * 100:.1f}%"
        text_color = 'darkgreen'
    else:
        ratio_text = f"Metapopulation decline: –{(1 - ratio) * 100:.1f}%"
        text_color = 'darkred'

    p.add_layout(Label(
        x=len(bin_labels) / 2, y=95,
        text=ratio_text, text_align='center', text_font_size='12pt',
        text_font_style='bold', text_color=text_color
    ))

    p.legend.location = "top_left"
    return p


def fold_change(data_dict, Vc):
    """
    Calculates the Log2 Fold Change (Final Count / Initial Count) for each droplet
    and plots it against Volume.

    Args:
        data_dict (dict): Dictionary of droplet DataFrames.
        Vc (float): Critical volume threshold (for plotting the vertical line).

    Returns:
        bokeh.plotting.figure: Scatter plot of Fold Change vs Volume.
    """
    fold_change_arr = np.array([])
    Volume = np.array([])
    droplet_id = np.array([])
    google_drive_url = np.array([])
    min_fc = -10  # Floor value for fold change (to handle extinction events)

    # --- Data Extraction Loop ---
    for key, value in data_dict.items():
        # Skip empty droplets (Initial count 0)
        if value['Count'].iloc[0] == 0:
            continue
        else:
            # Check if final count is 0 (Extinction)
            # using mean of last 4 timepoints to smooth noise
            if value['Count'].iloc[-4:].mean() == 0:
                fold_change_arr = np.append(fold_change_arr, np.nan)  # Handle as NaN then replace
            else:
                # Calculate ratio: End / Start
                fold_change_arr = np.append(fold_change_arr, value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])

            Volume = np.append(Volume, value['Volume'].iloc[0])
            droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
            google_drive_url = np.append(google_drive_url, value['Google Drive Link'].iloc[0])

    # Log2 transformation
    fold_change_arr = np.log2(fold_change_arr)
    # Replace extinctions (NaNs from log(0)) with min_fc floor
    fold_change_arr = np.where(np.isnan(fold_change_arr), min_fc, fold_change_arr)

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Volume': Volume,
        'fold change': fold_change_arr,
        'Droplet': droplet_id,
        'Google Drive Link': google_drive_url
    })
    df = df.sort_values(by='Volume').reset_index(drop=True)

    # Calculate Rolling Average (Moving Average)
    # Filter out the artificial floor values for the average calculation
    sub_df = df[df['fold change'] > min_fc].reset_index(drop=True)
    sub_df['moving average'] = sub_df['fold change'].rolling(window=100, min_periods=1).mean()

    # --- Plotting ---
    source = ColumnDataSource(df)
    sub_source = ColumnDataSource(sub_df)  # Source for the moving average line

    scatter = figure(x_axis_type='log', y_axis_type='linear',
                     x_axis_label='Volume (μm³)',
                     output_backend="webgl",
                     width=800, height=600, y_range=(-6, 8))

    # Styling
    scatter.xaxis.axis_label_text_font_size = "24pt"
    scatter.yaxis.axis_label_text_font_size = "24pt"

    # Plot Scatter Points
    scatter.scatter('Volume', 'fold change', source=source, color='gray', alpha=1)

    # Plot Moving Average (Red Line)
    scatter.line('Volume', 'moving average', source=sub_source, color='red', line_width=3)

    # Calculate and Plot Metapopulation Fold Change (Global Average)
    total_initial = sum(v['Count'].iloc[0] for v in data_dict.values() if v['Count'].iloc[0] != 0)
    total_final = sum(v['Count'].iloc[-4:].mean() for v in data_dict.values() if v['Count'].iloc[0] != 0)
    meta_fc = np.log2(total_final / total_initial)

    scatter.line([min(df['Volume']), max(df['Volume'])], [meta_fc, meta_fc], color='black', line_width=3)

    # Reference Lines (Baseline 0 and Vc)
    baseline = scatter.line([min(df['Volume']), max(df['Volume'])], [0, 0], color='black', line_dash='dashdot',
                            line_width=3)
    vc_line = scatter.line([Vc, Vc], [-12, 12], color='black', line_dash='dashed', line_width=3)

    # Tools
    hover = HoverTool(tooltips=[('Volume', '@Volume'), ('Fold Change', '@{fold change}'), ('Droplet ID', '@Droplet')])
    scatter.add_tools(hover)

    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const url = source.data['Google Drive Link'][selected_index]['Google Drive Link']; 
            window.open(url, "_blank");
        }
    """))
    scatter.add_tools(taptool)

    return scatter


def growth_curves(dict):
    """
    Plots the raw growth curves (Count vs Time) aggregated by Volume Bins.
    Generates two plots: Linear scale and Log scale.
    Includes the global "Metapopulation" growth curve for comparison.
    """
    # Filter valid droplets
    valid_droplets = [v for v in dict.values() if v['Count'].iloc[0] != 0]
    df = pd.concat(valid_droplets, ignore_index=True)

    # Binning Logic
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)

    # Cap bins at 6 (10^6)
    df['lower bin'] = np.where(df['lower bin'] >= 6, 6, df['lower bin'])
    df['upper bin'] = np.where(df['upper bin'] >= 6, 6, df['upper bin'])

    # Group by Bins and Time to get statistics (Mean, Std)
    grouped = df.groupby(['lower bin', 'upper bin', 'time'])
    counts = grouped.size().reset_index(name='sample_count')
    means = grouped['Count'].mean().reset_index(name='mean')
    stds = grouped['Count'].std().reset_index(name='std')

    # Merge stats
    result = pd.merge(counts, means, on=['lower bin', 'upper bin', 'time'])
    result = pd.merge(result, stds, on=['lower bin', 'upper bin', 'time'])

    # Calculate Standard Error (SE) for shading
    result['SE'] = result['std'] / np.sqrt(result['sample_count'])
    result['mean + se'] = result['mean'] + result['SE']
    result['mean - se'] = result['mean'] - result['SE']

    # Color Palette generation
    unique_bins = result.groupby(['lower bin', 'upper bin']).ngroups
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230]]
    palette = linear_palette(high_contrast_color_map, unique_bins)

    # --- Plot Setup ---
    p1 = figure(title='Growth Curves', x_axis_label='Time', y_axis_label='Mean Count',
                width=800, height=600, output_backend="webgl")
    p2 = figure(x_axis_label='Time (h)', y_axis_label='population Mean',
                width=800, height=600, output_backend="webgl", y_axis_type='log')

    legend_items_1, legend_items_2 = [], []

    # Iterate through bins and plot
    for (color, ((lower_bin, upper_bin), group)) in zip(palette, result.groupby(['lower bin', 'upper bin'])):
        source = ColumnDataSource(group)

        # Plot Lines
        line_1 = p1.line('time', 'mean', source=source, color=color, line_width=2)
        line_2 = p2.line('time', 'mean', source=source, color=color, line_width=2)

        # Plot Error Areas (Shading)
        varea_1 = p1.varea(x='time', y1='mean - se', y2='mean + se', source=source, color=color, alpha=0.2)
        varea_2 = p2.varea(x='time', y1='mean - se', y2='mean + se', source=source, color=color, alpha=0.2)

        label = f'Bin > {lower_bin}' if lower_bin == 6 else f'Bin {lower_bin}-{upper_bin}'
        legend_items_1.append(LegendItem(label=label, renderers=[line_1, varea_1]))
        legend_items_2.append(LegendItem(label=label, renderers=[line_2, varea_2]))

    # --- Metapopulation Line (Global Sum) ---
    meta = df.groupby('time')['Count'].agg(['sum', 'std', 'count']).reset_index()
    meta['SE'] = meta['std'] / np.sqrt(meta['count'])
    meta['sum + SE'] = meta['sum'] + meta['SE']
    meta['sum - SE'] = meta['sum'] - meta['SE']

    meta_source = ColumnDataSource(meta)

    # Add to Linear Plot
    meta_linear_line = p1.line('time', 'sum', source=meta_source, color='black', line_width=4, line_dash='dashed')
    meta_linear_SE = p1.varea(x='time', y1='sum - SE', y2='sum + SE', source=meta_source, color='black', alpha=0.15)
    legend_items_1.append(LegendItem(label='Metapopulation', renderers=[meta_linear_line, meta_linear_SE]))

    # Add to Log Plot
    meta_log_line = p2.line('time', 'sum', source=meta_source, color='black', line_width=4, line_dash='dashed')
    meta_log_SE = p2.varea(x='time', y1='sum - SE', y2='sum + SE', source=meta_source, color='black', alpha=0.15)
    legend_items_2.append(LegendItem(label='Metapopulation', renderers=[meta_log_line, meta_log_SE]))

    # Finalize Layouts
    legend_1 = Legend(items=legend_items_1, location='top_right')
    p1.add_layout(legend_1, 'right')
    p1.legend.click_policy = 'hide'

    return row(p1, p2)


def normalize_growth_curves(data_dict):
    """
    Plots growth curves normalized to the MAXIMUM population achieved within each bin.
    Metric: N(t) / N_max_of_bin

    This helps visualize the *timing* of growth phases (lag, log, plateau) relative to
    droplet size, ignoring absolute abundance differences.
    """
    # Filter valid droplets
    valid_droplets = [v for v in data_dict.values() if v['Count'].iloc[0] != 0]
    df = pd.concat(valid_droplets, ignore_index=True)

    # Binning Logic
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)
    df['lower bin'] = np.where(df['lower bin'] >= 6, 6, df['lower bin'])
    df['upper bin'] = np.where(df['upper bin'] >= 6, 6, df['upper bin'])

    # Group and Calculate Stats
    grouped = df.groupby(['lower bin', 'upper bin', 'time'])
    counts = grouped.size().reset_index(name='sample_count')
    means = grouped['Count'].mean().reset_index(name='mean')
    stds = grouped['Count'].std().reset_index(name='std')

    result = pd.merge(counts, means, on=['lower bin', 'upper bin', 'time'])
    result = pd.merge(result, stds, on=['lower bin', 'upper bin', 'time'])
    result['SE'] = result['std'] / np.sqrt(result['sample_count'])

    # --- Normalization Step (To Max) ---
    # Find the maximum mean population count ever reached by this specific bin
    result['max_mean'] = result.groupby(['lower bin', 'upper bin'])['mean'].transform('max')

    # Normalize current mean and SE by that max value
    result['normalized_mean'] = result['mean'] / result['max_mean']
    result['normalized_SE'] = result['SE'] / result['max_mean']
    result['mean + SE'] = result['normalized_mean'] + (result['normalized_SE'])
    result['mean - SE'] = result['normalized_mean'] - (result['normalized_SE'])

    # Plotting setup
    unique_bins = result.groupby(['lower bin', 'upper bin']).ngroups
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230]]
    palette = linear_palette(high_contrast_color_map, unique_bins)

    p1 = figure(title='Normalized to Max Growth Curves', x_axis_label='Time (h)',
                y_axis_label='Normalized population (Fraction of Max)', width=800, height=600, output_backend="webgl")
    p2 = figure(x_axis_label='Time (h)', y_axis_label='Relative population size (Log)',
                width=800, height=600, output_backend="webgl", y_axis_type='log')

    legend_items_1, legend_items_2 = [], []

    for (color, ((lower_bin, upper_bin), group)) in zip(palette, result.groupby(['lower bin', 'upper bin'])):
        source = ColumnDataSource(group)

        # Plot Lines
        line_1 = p1.line('time', 'normalized_mean', source=source, color=color, line_width=2)
        line_2 = p2.line('time', 'normalized_mean', source=source, color=color, line_width=2)

        # Plot Error Areas
        varea_1 = p1.varea(x='time', y1='mean - SE', y2='mean + SE', source=source, color=color, alpha=0.2)
        varea_2 = p2.varea(x='time', y1='mean - SE', y2='mean + SE', source=source, color=color, alpha=0.2)

        label = f'Bin {lower_bin}-{upper_bin}'
        legend_items_1.append(LegendItem(label=label, renderers=[line_1, varea_1]))
        legend_items_2.append(LegendItem(label=label, renderers=[line_2, varea_2]))

    legend_1 = Legend(items=legend_items_1, location='top_right')
    legend_2 = Legend(items=legend_items_2, location='top_right')
    p1.add_layout(legend_1, 'right')
    p2.add_layout(legend_2, 'right')

    return row(p1, p2)


def normalize_growth_curves_first_timepoint(data_dict):
    """
    Plots growth curves normalized to the INITIAL population (Time=0).
    Metric: N(t) / N(0)

    This visualizes the 'Fold Change over Time' or growth rate relative to start.
    """
    # Filter valid droplets
    valid_droplets = [v for v in data_dict.values() if v['Count'].iloc[0] != 0]
    df = pd.concat(valid_droplets, ignore_index=True)

    # Binning Logic
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)
    df['lower bin'] = np.where(df['lower bin'] >= 6, 6, df['lower bin'])
    df['upper bin'] = np.where(df['upper bin'] >= 6, 6, df['upper bin'])

    # Calculate basic stats per bin/time
    grouped = df.groupby(['lower bin', 'upper bin', 'time'])
    counts = grouped.size().reset_index(name='sample_count')
    means = grouped['Count'].mean().reset_index(name='mean')
    stds = grouped['Count'].std().reset_index(name='std')

    result = pd.merge(counts, means, on=['lower bin', 'upper bin', 'time'])
    result = pd.merge(result, stds, on=['lower bin', 'upper bin', 'time'])
    result['SE'] = result['std'] / np.sqrt(result['sample_count'])

    # --- Normalization Step (To Time 0) ---
    # Extract the mean at Time=0 for each bin
    first_means = result[result['time'] == 0][['lower bin', 'upper bin', 'mean']].rename(columns={'mean': 'first_mean'})

    # Merge this "first_mean" back into the main result dataframe
    result = result.merge(first_means, on=['lower bin', 'upper bin'], how='left')

    # Normalize
    result['normalized_mean'] = result['mean'] / result['first_mean']
    result['normalized_SE'] = result['SE'] / result['first_mean']
    result['mean + SE'] = result['normalized_mean'] + (result['normalized_SE'])
    result['mean - SE'] = result['normalized_mean'] - (result['normalized_SE'])

    # Plotting
    unique_bins = result.groupby(['lower bin', 'upper bin']).ngroups
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230]]
    palette = linear_palette(high_contrast_color_map, unique_bins)

    p1 = figure(x_axis_label='Time (h)', width=800, height=600, output_backend="webgl")
    p2 = figure(x_axis_label='Time', width=800, height=600, output_backend="webgl", y_axis_type='log')

    legend_items_1, legend_items_2 = [], []

    for (color, ((lower_bin, upper_bin), group)) in zip(palette, result.groupby(['lower bin', 'upper bin'])):
        source = ColumnDataSource(group)

        line_1 = p1.line('time', 'normalized_mean', source=source, color=color, line_width=2)
        varea_1 = p1.varea(x='time', y1='mean - SE', y2='mean + SE', source=source, color=color, alpha=0.2)
        line_2 = p2.line('time', 'normalized_mean', source=source, color=color, line_width=2)
        varea_2 = p2.varea(x='time', y1='mean - SE', y2='mean + SE', source=source, color=color, alpha=0.2)

        label = f'Bin > {lower_bin}' if lower_bin == 6 else f'Bin {lower_bin}-{upper_bin}'
        legend_items_1.append(LegendItem(label=label, renderers=[line_1, varea_1]))
        legend_items_2.append(LegendItem(label=label, renderers=[line_2, varea_2]))

    # --- Metapopulation Normalization ---
    meta = df.groupby('time')['Count'].agg(['mean', 'std', 'count']).reset_index()
    meta['SE'] = meta['std'] / np.sqrt(meta['count'])
    meta_first_mean = meta.loc[meta['time'] == 0, 'mean'].values[0]

    meta['normalized_mean'] = meta['mean'] / meta_first_mean
    meta['normalized_SE'] = meta['SE'] / meta_first_mean
    meta['mean + SE'] = meta['normalized_mean'] + meta['normalized_SE']
    meta['mean - SE'] = meta['normalized_mean'] - meta['normalized_SE']

    meta_source = ColumnDataSource(meta)

    # Add Metapopulation Lines
    meta_line_1 = p1.line('time', 'normalized_mean', source=meta_source, color='black', line_width=4,
                          line_dash='dashed')
    meta_se_1 = p1.varea(x='time', y1='mean - SE', y2='mean + SE', source=meta_source, color='black', alpha=0.15)
    legend_items_1.append(LegendItem(label='Metapopulation', renderers=[meta_line_1, meta_se_1]))

    meta_line_2 = p2.line('time', 'normalized_mean', source=meta_source, color='black', line_width=4,
                          line_dash='dashed')
    meta_se_2 = p2.varea(x='time', y1='mean - SE', y2='mean + SE', source=meta_source, color='black', alpha=0.15)
    legend_items_2.append(LegendItem(label='Metapopulation', renderers=[meta_line_2, meta_se_2]))

    # Add Legends
    legend_2 = Legend(items=legend_items_2, location='top_right')
    p2.add_layout(legend_2, 'right')

    return row(p1, p2)


def last_4_hours_average(chip, volume):
    """
    Analyzes the bacterial count in the final phase of the experiment (last 4 hours).
    It splits the data into two regimes based on a critical volume (Vc) and fits
    separate regression lines for "Small Droplets" vs "Large Droplets".

    Args:
        chip: Dictionary of droplet dataframes.
        volume (float): The critical volume threshold (Vc) used to split the regression.
    """
    # Calculate mean of last 4 timepoints for every droplet
    # Note: Assumes time > 20 is the relevant "end" period
    last_4_hours = {droplet_id: df[df['time'] > 20].reset_index(drop=True) for droplet_id, df in chip.items()}
    average_counts = np.array([df['Count'].mean() for df in last_4_hours.values()])

    # Handle zero counts (log issues)
    min_average_count = 0.1
    average_counts = np.where(average_counts == 0, min_average_count, average_counts)

    # Extract metadata
    droplet_sizes = [df['Volume'].iloc[0] for df in chip.values()]
    droplet_ids = [df['Droplet'].iloc[0] for df in chip.values()]
    google_drive_urls = [df['Google Drive Link'].iloc[0] for df in chip.values()]

    data = pd.DataFrame({
        'Volume': droplet_sizes,
        'Average Count': average_counts,
        'Droplet': droplet_ids,
        'Google Drive Link': google_drive_urls
    })
    data = data.sort_values(by='Volume').reset_index(drop=True)

    # --- Split Regressions (Before and After Vc) ---
    data_before = data[data['Volume'] <= volume]
    data_after = data[data['Volume'] > volume]

    # Filter out baseline noise for regression
    data_before = data_before[data_before['Average Count'] > data_before['Average Count'].min()]
    data_after = data_after[data_after['Average Count'] > data_after['Average Count'].min()]

    # Fit Regression: Small Droplets
    if not data_before.empty:
        slope_before, intercept_before, r_value_before, _, _ = linregress(np.log10(data_before['Volume']),
                                                                          np.log10(data_before['Average Count']))
        x_values_before = np.linspace(data_before['Volume'].min(), volume, 100)
        y_values_before = 10 ** (intercept_before + slope_before * np.log10(x_values_before))
    else:
        x_values_before, y_values_before = np.array([]), np.array([])
        slope_before = None

    # Fit Regression: Large Droplets
    if not data_after.empty:
        slope_after, intercept_after, r_value_after, _, _ = linregress(np.log10(data_after['Volume']),
                                                                       np.log10(data_after['Average Count']))
        x_values_after = np.linspace(volume, data_after['Volume'].max(), 100)
        y_values_after = 10 ** (intercept_after + slope_after * np.log10(x_values_after))
    else:
        x_values_after, y_values_after = np.array([]), np.array([])
        slope_after = None

    # --- Plotting ---
    source = ColumnDataSource(data)
    view = CDSView()
    scatter = figure(title='Average Number of Bacteria in Last 4 Hours vs. Droplet Size',
                     x_axis_type='log', y_axis_type='log',
                     x_axis_label='Volume', y_axis_label='Average Count',
                     output_backend="webgl", width=900, height=600)

    scatter.scatter('Volume', 'Average Count', source=source, view=view, color='gray', alpha=1)

    legend_items = [LegendItem(label='Average Count', renderers=[scatter.renderers[0]])]

    # Draw Regression Lines and Labels
    if x_values_before.any():
        reg_before = scatter.line(x_values_before, y_values_before, color='red')
        label_x = (x_values_before[0] + x_values_before[-1]) / 2
        label_y = 10 ** (intercept_before + slope_before * np.log10(label_x))
        scatter.add_layout(Label(x=label_x, y=label_y, text=f'Slope: {slope_before:.2f}', text_color='red'))
        legend_items.append(LegendItem(label='Regression Before', renderers=[reg_before]))

    if x_values_after.any():
        reg_after = scatter.line(x_values_after, y_values_after, color='blue')
        label_x = (x_values_after[0] + x_values_after[-1]) / 2
        label_y = 10 ** (intercept_after + slope_after * np.log10(label_x))
        scatter.add_layout(Label(x=label_x, y=label_y, text=f'Slope: {slope_after:.2f}', text_color='blue'))
        legend_items.append(LegendItem(label='Regression After', renderers=[reg_after]))

    # Vertical Line for Vc
    vline = Span(location=volume, dimension='height', line_color='blue', line_dash='dashed', line_width=2)
    scatter.add_layout(vline)

    # Legend
    scatter.add_layout(Legend(items=legend_items, location='top_right'), 'right')

    # Tap Tool
    taptool = TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            window.open(source.data['Google Drive Link'][selected_index], "_blank");
        }
    """))
    scatter.add_tools(taptool)

    return scatter


def find_droplet_location(df):
    """
    Calculates the Euclidean distance of each droplet from the physical center of the chip.
    Filters out droplets that are outside a specific radius (5000 units), effectively
    cropping the analysis to the center of the chip to avoid edge effects.

    Args:
        df: Dataframe containing 'X' and 'Y' coordinates of droplets.
    """
    # Define Chip Center (Hardcoded coordinates based on image dimensions 13200x13200)
    circle_center_x = 13200 / 2
    circle_center_y = 13200 / 2

    # Calculate Distance
    df['distance_to_center'] = np.sqrt((df['X'] - circle_center_x) ** 2 + (df['Y'] - circle_center_y) ** 2)

    # Flag droplets inside the valid radius (5000 px)
    df['is_inside_circle'] = df['distance_to_center'] <= 5000

    # Return only valid droplets
    return df[df['is_inside_circle']].reset_index(drop=True)


def death_rate_by_bins(dict):
    """
    Calculates the "Death Rate" (negative growth rate) over time for different Volume Bins.
    It computes the local slope of the log-transformed count using a rolling window.

    Returns:
        bokeh.plotting.figure: Plot of Death Rate (Slope) vs Time.
    """
    # Filter and Concatenate
    valid_droplets = [v for v in dict.values() if v['Count'].iloc[0] != 0]
    df = pd.concat(valid_droplets, ignore_index=True)

    # Binning
    df.loc[:, 'Bins_vol'] = df['log_Volume'].apply(math.floor)
    df.loc[:, 'Bins_vol_txt'] = df['log_Volume'].apply(math.ceil)
    df.rename(columns={'Bins_vol': 'lower bin', 'Bins_vol_txt': 'upper bin'}, inplace=True)

    # Group by Bin and Time -> Sum counts to get "Bin Population"
    grouped = df.groupby(['lower bin', 'upper bin', 'time'])['Count'].sum().reset_index(name='Count')

    # Calculate Log Count
    mask = grouped['Count'] > 0
    grouped['log_count'] = grouped[mask]['Count'].apply(np.log)

    # --- Rolling Slope Calculation ---
    window_size = 4
    # For each bin, apply a rolling linear regression to find the slope of log(N) vs time
    grouped['slope'] = grouped.groupby(['lower bin', 'upper bin'])['log_count'].transform(
        lambda x: x.rolling(window_size).apply(lambda y: linregress(range(window_size), y)[0]))

    # Calculate Standard Error of the slope
    grouped['standard_error'] = grouped.groupby(['lower bin', 'upper bin'])['log_count'].transform(
        lambda x: x.rolling(window_size).apply(lambda y: linregress(range(window_size), y)[4]))

    # Define error bounds for shading
    grouped['slope - standard_error'] = grouped['slope'] - grouped['standard_error']
    grouped['slope + standard_error'] = grouped['slope'] + grouped['standard_error']

    # --- Metapopulation Death Rate ---
    metapopulation = df.groupby('time')['Count'].sum().reset_index(name='metapopulation')
    metapopulation['log_metapopulation'] = np.log(metapopulation['metapopulation'])
    metapopulation['slope'] = metapopulation['log_metapopulation'].rolling(window=window_size).apply(
        lambda x: linregress(range(window_size), x)[0])

    # Plotting
    p = figure(title='Death Rate by Bins', x_axis_label='Time', y_axis_label='Slope (Growth Rate)',
               width=800, height=600, output_backend="webgl")

    colors = Category20[20]
    color_index = 0
    legend_items = []

    # Plot each bin
    for (lower_bin, upper_bin), group in grouped.groupby(['lower bin', 'upper bin']):
        source = ColumnDataSource(group)
        line = p.line('time', 'slope', source=source, color=colors[color_index], line_width=2)
        varea = p.varea(x='time', y1='slope - standard_error', y2='slope + standard_error', source=source,
                        color=colors[color_index], alpha=0.2)
        legend_items.append(LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[line, varea]))
        color_index = (color_index + 1) % len(colors)

    # Plot Metapopulation (Black Line)
    meta_source = ColumnDataSource(metapopulation)
    meta_line = p.line('time', 'slope', source=meta_source, line_width=3, color='black')
    legend_items.append(LegendItem(label='Metapopulation Death Rate', renderers=[meta_line]))

    p.add_layout(Legend(items=legend_items, location='top_right'), 'right')
    return p


def death_rate_by_droplets(data_dict, chip):
    """
    Calculates the maximum (or minimum) death rate experienced by EACH droplet individually.
    Then performs a statistical comparison (Permutation Test) between different volume bins.

    Args:
        data_dict: Droplet data.
        chip: Chip name (used to toggle between finding Max or Min slope).
              Control chips usually look for Max (growth), Antibiotic chips for Min (death).
    """
    volumes = []
    max_death_rate = []
    droplet_ids = []
    google_drive_urls = []

    # --- Per-Droplet Slope Calculation ---
    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue

        window_size = 4
        mask = value['Count'] > 0
        value['log_count'] = value[mask]['Count'].apply(np.log)

        # Calculate rolling slope
        value['slope'] = value['log_count'].rolling(window_size).apply(
            lambda x: linregress(range(window_size), x)[0])

        volumes.append(value['Volume'].iloc[0])

        # Determine specific metric based on experiment type
        if 'Control' in chip:
            # For controls, we care about max growth rate
            # Filter for reasonable biological growth rates (0 < slope < 1)
            slope = value['slope'].max()
            max_death_rate.append(slope if 0 < slope < 1 else np.nan)
        else:
            # For antibiotics, we care about max death rate (minimum slope)
            max_death_rate.append(value['slope'].min())

        droplet_ids.append(key)
        google_drive_urls.append(value['Google Drive Link'].iloc[0])

    # Construct DataFrame
    df = pd.DataFrame({
        'Volume': np.log10(volumes),
        'Slope': max_death_rate,
        'Droplet': droplet_ids,
        'Google Drive Link': google_drive_urls
    })

    # Binning and Coloring
    df['upper bin'] = df['Volume'].apply(math.ceil)
    df['lower bin'] = df['Volume'].apply(math.floor)

    grouped = df.groupby(['lower bin', 'upper bin'])
    high_contrast_color_map = [cc.CET_D1[0], cc.CET_D1[80], cc.CET_D1[180], cc.CET_D1[230], cc.CET_D1[255]]

    # Map groups to colors cyclically
    color_map = {group: high_contrast_color_map[i % len(high_contrast_color_map)]
                 for i, group in enumerate(grouped.groups.keys())}
    df['color'] = df.apply(lambda row: color_map[(row['lower bin'], row['upper bin'])], axis=1)

    # Calculate Global Metapopulation Rate (for reference line)
    valid_droplets = [v for v in data_dict.values() if v['Count'].iloc[0] != 0]
    metapopulation = pd.concat(valid_droplets, ignore_index=True).groupby('time')['Count'].sum().reset_index(
        name='meta')
    metapopulation['log_meta'] = np.log(metapopulation['meta'])
    metapopulation['slope'] = metapopulation['log_meta'].rolling(window_size).apply(
        lambda x: linregress(range(window_size), x)[0])

    mean_death_rate = metapopulation['slope'].max() if 'Control' in chip else metapopulation['slope'].min()

    # --- Plot Setup ---
    p = figure(x_axis_label='Volume (µm³)', y_axis_label='Growth rate (h⁻¹)',
               output_backend="webgl", width=1200, height=600)

    # Custom X-Axis Ticks (10^x formatting)
    p.xaxis.formatter = CustomJSTickFormatter(code="""
        const superscripts = {'0': '⁰','1': '¹','2': '²','3': '³','4': '⁴','5': '⁵','6': '⁶','7': '⁷','8': '⁸','9': '⁹'};
        return "10" + superscripts[tick.toFixed(0)];
    """)
    p.xaxis.axis_label_text_font_size = "24pt"
    p.yaxis.axis_label_text_font_size = "24pt"

    # --- Statistical Testing (Permutation Test) ---
    permutations_data = pd.DataFrame(columns=['first compared bin', 'second compared bin', 'p-value'])
    lower_bins = sorted(df['lower bin'].unique())

    # Compare adjacent bins
    for k in range(len(lower_bins) - 1):
        i = lower_bins[k]
        j = lower_bins[k + 1]
        data1 = df[df['lower bin'] == i]['Slope'].dropna()
        data2 = df[df['lower bin'] == j]['Slope'].dropna()

        if len(data1) >= 2 and len(data2) >= 2:
            # Perform Permutation Test
            p_val = permutation_test((data1, data2), lambda x, y: np.mean(x) - np.mean(y),
                                     n_resamples=1000, alternative='two-sided').pvalue
            permutations_data = pd.concat([permutations_data,
                                           pd.DataFrame(
                                               {'first compared bin': i, 'second compared bin': j, 'p-value': p_val},
                                               index=[0])],
                                          ignore_index=True)

    # FDR Correction (Benjamini-Hochberg)
    if not permutations_data.empty:
        adjusted_results = multipletests(permutations_data['p-value'], method='fdr_bh')
        permutations_data['adjusted p-value'] = adjusted_results[1]

        # Draw significance bars on plot
        for index, row in permutations_data.iterrows():
            sig_symbol = 'NS' if row['adjusted p-value'] > 0.05 else '*' if row['adjusted p-value'] > 0.01 else '**' if \
            row['adjusted p-value'] > 0.001 else '***'

            # Coordinates for drawing the bracket
            x_start, x_end = row['first compared bin'] + 0.5 + 0.2, row['second compared bin'] - 0.5 + 0.8
            y_pos = df['Slope'].max() + 0.03 + index * 0.03

            # Draw Bracket and Label
            p.line([x_start, x_end], [y_pos, y_pos], color="black", line_width=1.5)
            p.line([x_start, x_start], [y_pos, y_pos - 0.01], color="black")
            p.line([x_end, x_end], [y_pos, y_pos - 0.01], color="black")
            p.add_layout(
                Label(x=(x_start + x_end) / 2, y=y_pos + 0.002, text=sig_symbol, text_color="red", text_align="center"))

    # --- Box Plot & Scatter Overlay ---
    source = ColumnDataSource(df)
    scatter = p.scatter(x='Volume', y='Slope', source=source, color='color')

    # Draw Box Plots manually
    box_width = 0.6
    for (lower_bin, upper_bin), group in grouped:
        color = color_map[(lower_bin, upper_bin)]
        center = (lower_bin + upper_bin) / 2
        q1 = group['Slope'].quantile(0.25)
        q3 = group['Slope'].quantile(0.75)
        median = group['Slope'].median()
        iqr = q3 - q1
        upper_whisker = min(group['Slope'].max(), q3 + 1.5 * iqr)
        lower_whisker = max(group['Slope'].min(), q1 - 1.5 * iqr)

        # Box
        p.quad(top=[q3], bottom=[q1], left=[center - box_width / 2], right=[center + box_width / 2],
               fill_color=color, alpha=0.6, line_color="black")
        # Median
        p.segment(x0=center - box_width / 2, y0=median, x1=center + box_width / 2, y1=median, line_color="black",
                  line_width=2)
        # Whiskers
        p.segment(x0=center, y0=upper_whisker, x1=center, y1=q3, line_color="black")
        p.segment(x0=center, y0=lower_whisker, x1=center, y1=q1, line_color="black")
        # Caps
        p.segment(x0=center - 0.125, y0=upper_whisker, x1=center + 0.125, y1=upper_whisker, line_color="black")
        p.segment(x0=center - 0.125, y0=lower_whisker, x1=center + 0.125, y1=lower_whisker, line_color="black")

    # Metapopulation Line
    p.line(x=[3, 8], y=[mean_death_rate, mean_death_rate], line_dash='dashed', line_color='black', line_width=2)

    # Legend & Tools
    p.add_layout(Legend(items=[LegendItem(label='Metapopulation Growth rate', renderers=[p.renderers[-1]])],
                        location='top_right'), 'right')

    # Tap Tool to highlight selected point and open URL
    p.add_tools(TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            window.open(source.data['Google Drive Link'][selected_index], "_blank");
            // Simple highlighting logic (set alpha of others to 0.1)
            for (let i = 0; i < source.data['alpha'].length; i++) {
                source.data['alpha'][i] = (i === selected_index) ? 1.0 : 0.1;
            }
            source.change.emit();
        }
    """)))

    p.add_tools(HoverTool(tooltips=[('Log 10 Volume', '@Volume'), ('Slope', '@{Slope}'), ('Droplet', '@Droplet')],
                          renderers=[scatter]))

    return p


def distance_Vs_Volume_histogram(df):
    """
    Generates a Normalized Stacked Histogram showing the distribution of Droplet Volumes
    at different distances from the chip center.

    X-Axis: Distance from center (Bins: 0-1000, 1000-2000, etc.)
    Y-Axis: Proportion of droplets belonging to specific Log Volume bins.

    Purpose: To detect if there is a radial bias in droplet generation size.
    """
    df = df.copy()

    # Define Distance Bins (Radial rings)
    distance_bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, float('inf')]
    distance_labels = ["0-1000", "1000-2000", "2000-3000", "3000-4000", "4000-5000", "5000-6000", "6000+"]

    # Define Volume Bins (Log10)
    volume_bins = [3, 4, 5, 6, 7, 8]
    volume_labels = ["3-4", "4-5", "5-6", "6-7", "7-8"]

    # Apply Binning
    df['distance_bin'] = pd.cut(df['distance_to_center'], bins=distance_bins, labels=distance_labels, right=False)
    df['volume_bin'] = pd.cut(df['log_Volume'], bins=volume_bins, labels=volume_labels, right=False)

    # Group and Pivot to create stacked data structure
    grouped = df.groupby(['distance_bin', 'volume_bin'], observed=True).size().unstack(fill_value=0)

    # Normalize rows to 1 (Proportion)
    normalized_grouped = grouped.div(grouped.sum(axis=1), axis=0)

    # Prepare Data for Bokeh vbar_stack
    source_data = {'distance_bin': distance_labels}
    for volume_label in volume_labels:
        # Extract column for this volume bin, ensure it aligns with distance labels
        source_data[volume_label] = normalized_grouped.get(volume_label, [0] * len(distance_labels))

    source = ColumnDataSource(data=source_data)
    colors = Category20[len(volume_labels)]  # Assign distinct colors to volume bins

    # Plotting
    p = figure(x_range=distance_labels, title="Normalized Stacked Histogram: Distance vs. Log Volume",
               toolbar_location=None, tools="", output_backend="webgl", width=800, height=600)

    p.vbar_stack(volume_labels, x='distance_bin', width=0.6, color=colors, source=source,
                 legend_label=volume_labels)

    # Styling
    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.xaxis.axis_label = "Distance from Center"
    p.yaxis.axis_label = "Proportion"

    # Legend Styling
    p.legend.title = "Log Volume"
    p.legend.label_text_font_size = "10pt"
    p.legend.orientation = "vertical"
    p.legend.location = "top_center"
    p.add_layout(p.legend[0], 'right')

    # Hover Tool
    hover = HoverTool()
    hover.tooltips = [("Distance Bin", "@distance_bin"), ("Volume Bin", "$name"), ("Proportion", "@$name")]
    p.add_tools(hover)

    return p


def distance_Vs_occupide_histogram(df):
    """
    Similar to the volume histogram, but filters ONLY for OCCUPIED droplets (Count > 0).

    Purpose: To check if bacterial occupancy is biased by radial position
    (e.g., do bacteria survive better in the center vs edges?).
    """
    df = df.copy()
    df = df[df['Count'] > 0]  # Filter for occupancy

    # Define Bins
    distance_bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, float('inf')]
    distance_labels = ["0-1000", "1000-2000", "2000-3000", "3000-4000", "4000-5000", "5000-6000", "6000+"]
    volume_bins = [3, 4, 5, 6, 7, 8]
    volume_labels = ["3-4", "4-5", "5-6", "6-7", "7-8"]

    # Apply Binning
    df['distance_bin'] = pd.cut(df['distance_to_center'], bins=distance_bins, labels=distance_labels, right=False)
    df['volume_bin'] = pd.cut(df['log_Volume'], bins=volume_bins, labels=volume_labels, right=False)

    # Group, Pivot, Normalize
    grouped = df.groupby(['distance_bin', 'volume_bin'], observed=True).size().unstack(fill_value=0)
    normalized_grouped = grouped.div(grouped.sum(axis=1), axis=0)

    # Prepare Data
    source_data = {'distance_bin': distance_labels}
    for volume_label in volume_labels:
        source_data[volume_label] = normalized_grouped.get(volume_label, [0] * len(distance_labels))

    source = ColumnDataSource(data=source_data)
    colors = Category20[len(volume_labels)]

    # Plotting
    p = figure(x_range=distance_labels, title="Normalized Stacked Histogram: Distance vs. Log Volume Occupied",
               toolbar_location=None, tools="", output_backend="webgl", width=800, height=600)

    p.vbar_stack(volume_labels, x='distance_bin', width=0.6, color=colors, source=source,
                 legend_label=volume_labels)

    # Styling & Layout
    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.xaxis.axis_label = "Distance from Center"
    p.yaxis.axis_label = "Proportion"
    p.legend.title = "Log Volume"
    p.legend.label_text_font_size = "10pt"
    p.legend.orientation = "vertical"
    p.legend.location = "top_center"
    p.add_layout(p.legend[0], 'right')

    hover = HoverTool()
    hover.tooltips = [("Distance Bin", "@distance_bin"), ("Volume Bin", "$name"), ("Proportion", "@$name")]
    p.add_tools(hover)

    return p


def distance_Vs_Volume_circle(df):
    """
    Creates a physical map of the chip (Scatter Plot of X vs Y).
    Points are colored by their Volume Bin.
    Concentric circles are drawn to visualize the distance rings.

    Args:
        df: Dataframe with 'X', 'Y', 'Area', 'log_Volume'.
    """
    df = df.copy()

    # Calculate radius for visual representation of the droplet point itself
    df['radius'] = (df['Area'] / math.pi) ** 0.5

    # Binning for Color Mapping
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)

    p = figure(title='Distance to Center vs. Volume',
               output_backend="webgl", x_range=(0, 13000), y_range=(0, 13000))

    # Determine Chip Center
    circle_center_x = (df['X'].min() + df['X'].max()) / 2
    circle_center_y = (df['Y'].min() + df['Y'].max()) / 2

    # Draw Reference Concentric Circles (The "Target" Pattern)
    radius_values = [1000, 2000, 3000, 4000, 5000, 6000, 6500 * 1.04]
    labels = ['0-1000', '1000-2000', '2000-3000', '3000-4000', '4000-5000', '5000-6000', '6000+']

    for i, radius in enumerate(radius_values):
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius,
                 line_color="black", fill_color=None, alpha=0.5)
        # Add labels for the rings
        if i < len(labels):
            label = Label(x=circle_center_x, y=circle_center_y + radius,
                          text=labels[i], text_align='center',
                          text_baseline='middle', text_font_style='bold', text_font_size='12pt')
            p.add_layout(label)

    # Draw Droplets colored by Bin
    legend_items = []
    colors = Category20[20]
    scatter_renderers = []

    grouped = df.groupby(['lower bin', 'upper bin'])
    for index, ((lower_bin, upper_bin), group) in enumerate(grouped):
        color = colors[index]
        source = ColumnDataSource(group)
        scatter = p.circle(x='X', y='Y', radius='radius', source=source, color=color, fill_alpha=0.5)
        scatter_renderers.append(scatter)
        legend_items.append(LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[scatter]))

    # Legend & Tools
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
    """
    Same as the circle map above, but filters for OCCUPIED droplets only.
    Includes interaction (TapTool) to open Google Drive links.
    """
    df = df.copy()
    df = df[df['Count'] > 0]  # Filter

    df['radius'] = (df['Area'] / math.pi) ** 0.5
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)

    p = figure(title='Distance to Center vs. Volume Occupied',
               output_backend="webgl", x_range=(0, 13000), y_range=(0, 13000))

    # Chip Center & Reference Rings
    circle_center_x = (df['X'].min() + df['X'].max()) / 2
    circle_center_y = (df['Y'].min() + df['Y'].max()) / 2
    radius_values = [1000, 2000, 3000, 4000, 5000, 6000, 6500 * 1.04]
    labels = ['0-1000', '1000-2000', '2000-3000', '3000-4000', '4000-5000', '5000-6000', '6000+']

    for i, radius in enumerate(radius_values):
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius,
                 line_color="black", fill_color=None, alpha=0.5)
        if i < len(labels):
            label = Label(x=circle_center_x, y=circle_center_y + radius,
                          text=labels[i], text_align='center',
                          text_baseline='middle', text_font_style='bold', text_font_size='12pt')
            p.add_layout(label)

    # Draw Scatter Points
    legend_items = []
    grouped = df.groupby(['lower bin', 'upper bin'])
    scatter_renderers = []
    colors = Category20[20]
    sources = []

    for index, ((lower_bin, upper_bin), group) in enumerate(grouped):
        color = colors[index]
        source = ColumnDataSource(group)
        sources.append(source)
        scatter = p.circle(x='X', y='Y', radius='radius', source=source, color=color, fill_alpha=0.5)
        legend_items.append(LegendItem(label=f'Bin {lower_bin}-{upper_bin}', renderers=[scatter]))
        scatter_renderers.append(scatter)

    legend = Legend(items=legend_items, location='top_left')
    p.add_layout(legend)
    p.legend.click_policy = 'hide'

    # Interaction Tools
    hover = HoverTool(tooltips=[('Log Volume', '@log_Volume'),
                                ('Droplet', '@Droplet'),
                                ('Radius', '@radius')],
                      renderers=scatter_renderers)
    p.add_tools(hover)

    # Tap Tool to open Google Drive Link (iterates through sources to find selection)
    taptool = TapTool(callback=CustomJS(args=dict(sources=sources), code="""
        for (let source of sources) {
            const selected_index = source.selected.indices[0];
            if (selected_index != null) {
                const data = source.data;
                const url = data['Google Drive Link'][selected_index];
                window.open(url, "_blank");
                break;
            }
        }
    """))
    p.add_tools(taptool)

    return p


def distance_Vs_Volume_colored_by_death_rate(df, data_dict, chip):
    """
    Generates a physical map of the chip (X vs Y), where each droplet is colored
    according to its Death Rate (Slope).

    - Cool Colors (Blue): Slower death (or growth in controls).
    - Warm Colors (Red): Faster death.

    Includes a CheckboxGroup to toggle visibility of specific Volume Bins.
    """
    df = df.copy()

    # Pre-calculate derived metrics
    df['log_Volume'] = np.log10(df['Volume'])
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
    df['radius'] = (df['Area'] / math.pi) ** 0.5

    # --- Calculate Death Rates per Droplet ---
    max_death_rate = []
    droplet_ids = []

    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue

        window_size = 4
        mask = value['Count'] > 0
        value['log_count'] = value[mask]['Count'].apply(np.log)
        value['slope'] = value['log_count'].rolling(window_size).apply(
            lambda x: linregress(range(window_size), x)[0])

        # Select Max or Min slope based on experiment type
        if 'Control' in chip:
            max_death_rate.append(value['slope'].max())
        else:
            max_death_rate.append(value['slope'].min())

        droplet_ids.append(key)

    # Merge Death Rates into main DataFrame
    death_rates = pd.DataFrame({'Slope': max_death_rate, 'Droplet': droplet_ids})
    death_rates.dropna(subset=['Slope'], inplace=True)
    df = pd.merge(df, death_rates, on='Droplet')

    # --- Color Mapping ---
    # Use Jet colormap (Blue -> Red)
    jet_palette = [RGB(*[int(255 * c) for c in cm.jet(i)[:3]]).to_hex() for i in range(256)]
    color_mapper = LinearColorMapper(palette=jet_palette, low=-2, high=2)

    title = 'Distance to Center vs. Volume Colored by Maximal Slope' if 'Control' in chip else 'Distance to Center vs. Volume Colored by Minimal Slope'

    p = figure(title=title, match_aspect=True, output_backend="webgl", width=800, height=600)
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "Y"

    # Draw Reference Rings
    circle_center_x = (df['X'].min() + df['X'].max()) / 2
    circle_center_y = (df['Y'].min() + df['Y'].max()) / 2
    for radius in [6500 * 1.04, 6000, 5000, 4000, 3000, 2000, 1000]:
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius, line_color="black", fill_color=None,
                 alpha=0.5)
        label_text = f'{radius - 1000}-{radius}' if radius < 6500 else '6000+'
        p.add_layout(Label(x=circle_center_x, y=circle_center_y + (radius - 1000) * 1.05,
                           text=label_text, text_align='center', text_baseline='middle',
                           text_font_style='bold', text_font_size='12pt'))

    # --- Plot Droplets by Bin ---
    grouped = df.groupby(['lower bin', 'upper bin'])
    scatter_renderers = []
    sources = []
    checkbox_labels = []

    for (lower_bin, upper_bin), group in grouped:
        source = ColumnDataSource(group)
        sources.append(source)

        # Color mapping applied here
        scatter = p.circle(x='X', y='Y', radius='radius', source=source,
                           color={'field': 'Slope', 'transform': color_mapper}, fill_alpha=0.5)

        scatter_renderers.append(scatter)
        checkbox_labels.append(f'Bin {lower_bin}-{upper_bin}')

    # Add Checkbox Control (JavaScript callback to toggle visibility)
    checkbox_group = CheckboxGroup(labels=checkbox_labels, active=list(range(len(checkbox_labels))))
    checkbox_group.js_on_change('active',
                                CustomJS(args=dict(scatter_renderers=scatter_renderers, checkbox_group=checkbox_group),
                                         code="""
        for (let i = 0; i < scatter_renderers.length; i++) {
            scatter_renderers[i].visible = checkbox_group.active.includes(i);
        }
    """))

    # Tools & Layout
    p.add_tools(HoverTool(tooltips=[('Log Volume', '@log_Volume'), ('Slope', '@{Slope}'), ('Droplet', '@Droplet')],
                          renderers=scatter_renderers))

    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), title='Slope')
    p.add_layout(color_bar, 'right')

    # TapTool for Google Drive Links
    p.add_tools(TapTool(callback=CustomJS(args=dict(sources=sources), code="""
        for (let source of sources) {
            const selected_index = source.selected.indices[0];
            if (selected_index != null) {
                window.open(source.data['Google Drive Link'][selected_index], "_blank");
                break;
            }
        }
    """)))

    return row(checkbox_group, p, sizing_mode='fixed', width=800, height=600)


def distance_Vs_Volume_colored_by_fold_change(df, data_dict):
    """
    Similar to the Death Rate map, but colors droplets by Log2 Fold Change (Growth Yield).

    - Blue: Low yield (or extinction).
    - Red: High yield.
    """
    df = df.copy()
    df['log_Volume'] = np.log10(df['Volume'])
    df['lower bin'] = df['log_Volume'].apply(math.floor)
    df['upper bin'] = df['log_Volume'].apply(math.ceil)
    df['radius'] = (df['Area'] / math.pi) ** 0.5

    # --- Calculate Fold Change ---
    fold_change_arr = np.array([])
    Volume = np.array([])
    droplet_id = np.array([])
    min_fc = -10

    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue

        if value['Count'].iloc[-4:].mean() == 0:
            fold_change_arr = np.append(fold_change_arr, np.nan)
        else:
            fold_change_arr = np.append(fold_change_arr, value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])

        Volume = np.append(Volume, value['Volume'].iloc[0])
        droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])

    fold_change_arr = np.log2(fold_change_arr)
    fold_change_arr = np.where(np.isnan(fold_change_arr), min_fc, fold_change_arr)

    fold_changes = pd.DataFrame({'Volume': Volume, 'fold change': fold_change_arr, 'Droplet': droplet_id})
    df = pd.merge(df, fold_changes, on='Droplet')

    # Color Map (Range: -10 to 10 Log2 Fold Change)
    jet_palette = [RGB(*[int(255 * c) for c in cm.jet(i)[:3]]).to_hex() for i in range(256)]
    color_mapper = LinearColorMapper(palette=jet_palette, low=-10, high=10)

    p = figure(title='Distance to Center vs. Volume Colored by Fold Change', match_aspect=True, output_backend="webgl",
               width=800, height=600)
    p.xaxis.axis_label = "X"
    p.yaxis.axis_label = "Y"

    # Reference Rings
    circle_center_x = (df['X'].min() + df['X'].max()) / 2
    circle_center_y = (df['Y'].min() + df['Y'].max()) / 2
    for radius in [6500 * 1.04, 6000, 5000, 4000, 3000, 2000, 1000]:
        p.circle(x=[circle_center_x], y=[circle_center_y], radius=radius, line_color="black", fill_color=None,
                 alpha=0.5)
        label_text = f'{radius - 1000}-{radius}' if radius < 6500 else '6000+'
        p.add_layout(Label(x=circle_center_x, y=circle_center_y + (radius - 1000) * 1.05,
                           text=label_text, text_align='center', text_baseline='middle', text_font_style='bold',
                           text_font_size='12pt'))

    # Plotting & Checkboxes
    grouped = df.groupby(['lower bin', 'upper bin'])
    scatter_renderers = []
    checkbox_labels = []
    sources = []

    for (lower_bin, upper_bin), group in grouped:
        source = ColumnDataSource(group)
        sources.append(source)
        scatter = p.circle(x='X', y='Y', radius='radius', source=source,
                           color={'field': 'fold change', 'transform': color_mapper}, fill_alpha=0.5)
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

    p.add_tools(
        HoverTool(tooltips=[('Log Volume', '@log_Volume'), ('Fold Change', '@{fold change}'), ('Droplet', '@Droplet')],
                  renderers=scatter_renderers))

    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), title='fold change')
    p.add_layout(color_bar, 'right')

    # TapTool
    p.add_tools(TapTool(callback=CustomJS(args=dict(sources=sources), code="""
        for (let source of sources) {
            const selected_index = source.selected.indices[0];
            if (selected_index != null) {
                window.open(source.data['Google Drive Link'][selected_index], "_blank");
                break;
            }
        }
    """)))

    return row(checkbox_group, p, sizing_mode='fixed', width=800, height=600)


def bins_volume_Vs_distance(data_dict, chip):
    """
    Aggregates biological metrics (Fold Change and Death Rate) by BOTH Volume Bin AND Distance Bin.

    It produces two plots side-by-side:
    1. Distance vs Mean Fold Change
    2. Distance vs Mean Death Rate

    Each plot contains:
    - Line plot: Connecting the means across distances.
    - Violin/Scatter plot: Showing the underlying distribution of data points.
    """
    # Data Aggregation Lists
    volumes = []
    distances = []
    droplet_ids = []
    fold_changes = []
    death_rates = []
    google_drive_urls = []
    min_fc = -10

    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue

        volumes.append(value['log_Volume'].iloc[0])
        distances.append(value['distance_to_center'].iloc[0])
        droplet_ids.append(key)
        google_drive_urls.append(value['Google Drive Link'].iloc[0])

        # Fold Change Calc
        if value['Count'].iloc[-4:].mean() == 0:
            fold_changes.append(np.nan)
        else:
            fold_changes.append(value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])

        # Slope Calc
        window_size = 4
        mask = value['Count'] > 0
        value['log_count'] = value[mask]['Count'].apply(np.log10)
        value['slope'] = value['log_count'].rolling(window_size).apply(lambda x: linregress(range(window_size), x)[0])

        if 'Control' in chip:
            death_rates.append(value['slope'].max())
        else:
            death_rates.append(value['slope'].min())

    df = pd.DataFrame({'Volume': volumes, 'Distance': distances, 'Droplet': droplet_ids,
                       'Fold Change': fold_changes, 'Slope': death_rates, 'Google Drive Link': google_drive_urls})

    df['Fold Change'] = np.log2(df['Fold Change'])
    df['Fold Change'] = np.where(np.isnan(df['Fold Change']), min_fc, df['Fold Change'])

    # Binning
    df['upper_volume_bin'] = df['Volume'].apply(math.ceil)
    df['lower_volume_bin'] = df['Volume'].apply(math.floor)
    distance_bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, float('inf')]
    df['lower_distance_bin'] = pd.cut(df['Distance'], bins=distance_bins, labels=distance_bins[:-1], right=False)
    df['upper_distance_bin'] = pd.cut(df['Distance'], bins=distance_bins, labels=distance_bins[1:], right=False)

    # Calculate Means per Group
    df['mean_fold_change'] = np.nan
    df['mean_slope'] = np.nan
    grouped = df.groupby(['lower_volume_bin', 'upper_volume_bin', 'lower_distance_bin', 'upper_distance_bin'],
                         observed=True)

    for (l_vol, u_vol, l_dist, u_dist), group in grouped:
        mean_fc = group['Fold Change'].mean()
        mean_sl = group['Slope'].mean()

        # Fill mean values back into DF for plotting lines
        mask = (df['lower_volume_bin'] == l_vol) & (df['upper_volume_bin'] == u_vol) & \
               (df['lower_distance_bin'] == l_dist) & (df['upper_distance_bin'] == u_dist)
        df.loc[mask, 'mean_fold_change'] = mean_fc
        df.loc[mask, 'mean_slope'] = mean_sl

        # Handle NaN slopes
        df['Slope'] = df['Slope'].fillna(mean_sl)

    # --- Helper Function to Create Each Subplot ---
    def create_plot(results, y_axis_label, y_column, points_column):
        title = f'Distance Bin vs. {y_axis_label}'
        p = figure(title=title, x_axis_label='Distance Bin', y_axis_label=y_axis_label, output_backend="webgl",
                   width=800, height=600)

        volume_bins = results['lower_volume_bin'].sort_values().unique()
        colors = Category20[len(volume_bins)]
        legend_items = []
        sources = []
        scatter_renderers = []

        for i, volume_bin in enumerate(volume_bins):
            # Extract data for this volume bin across ALL distances
            volume_bin_data = results[results['lower_volume_bin'] == volume_bin].sort_values('lower_distance_bin')
            source = ColumnDataSource(volume_bin_data)
            sources.append(source)

            # Line connecting means
            line = p.line(x='lower_distance_bin', y=y_column, source=source, line_width=3, color=colors[i])

            # Scatter points (Jittered)
            scatter = p.scatter(x=jitter('lower_distance_bin', width=100, range=p.x_range), y=points_column,
                                source=source, color=colors[i], fill_alpha=0.5)
            scatter_renderers.append(scatter)

            # Violin Plots (KDE)
            violin_renderers = []
            for lower_distance_bin in volume_bin_data['lower_distance_bin'].unique():
                data = volume_bin_data[volume_bin_data['lower_distance_bin'] == lower_distance_bin]
                if len(data) < 2: continue

                # KDE calc
                kde = gaussian_kde(data[points_column] + np.random.normal(0, 1e-6, len(data[points_column])))
                y = np.linspace(data[points_column].min(), data[points_column].max(), 1000)
                density = kde(y)
                density = density / density.max() * 250  # Scale width

                x_centered = lower_distance_bin
                x_combined = np.concatenate([x_centered + density, x_centered - density[::-1]])
                y_combined = np.concatenate([y, y[::-1]])

                violin = p.patch(x=x_combined, y=y_combined, fill_color=colors[i], line_color=colors[i], fill_alpha=0.3)
                violin_renderers.append(violin)

            legend_items.append(LegendItem(label=f'Volume Bin {volume_bin}-{volume_bin + 1}',
                                           renderers=[line, scatter] + violin_renderers))

        # Tools
        p.add_tools(HoverTool(tooltips=[('Distance Bin', '@lower_distance_bin'),
                                        (points_column, f'@{{{points_column}}}'),
                                        ('Volume Bin', '@lower_volume_bin')],
                              renderers=scatter_renderers))

        p.add_layout(Legend(items=legend_items, location='top_left'), 'right')
        p.legend.click_policy = 'hide'

        p.add_tools(TapTool(callback=CustomJS(args=dict(sources=sources), code="""
            for (let source of sources) {
                const selected_index = source.selected.indices[0];
                if (selected_index != null) {
                    window.open(source.data['Google Drive Link'][selected_index], "_blank");
                    break;
                }
            }
        """)))

        return p

    plot_fold_change = create_plot(df, 'Mean Fold Change', 'mean_fold_change', 'Fold Change')
    plot_death_rate = create_plot(df, 'Mean Slope', 'mean_slope', 'Slope')

    return row(plot_fold_change, plot_death_rate)


def FC_vs_density(data_dict):
    """
    Scatter plot: Log2 Fold Change vs. Log2 Initial Density.

    Color: Points are colored by Droplet Volume.
    Purpose: To check if initial crowding (density) affects the final yield,
    independent of droplet size.
    """
    # Initialize arrays
    fold_change = np.array([])
    density = np.array([])
    Volume = np.array([])
    droplet_id = np.array([])
    google_drive_url = np.array([])
    min_fc = -10  # Floor for extinction events

    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue

        # Calculate Initial Density (N0 / V)
        current_vol = value['Volume'].iloc[0]
        current_density = value['Count'].iloc[0] / current_vol

        # Calculate Fold Change
        if value['Count'].iloc[-4:].mean() == 0:
            fold_change = np.append(fold_change, np.nan)
        else:
            fold_change = np.append(fold_change, value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])

        Volume = np.append(Volume, current_vol)
        density = np.append(density, current_density)
        droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
        google_drive_url = np.append(google_drive_url, value['Google Drive Link'].iloc[0])

    # Log transformations
    fold_change = np.log2(fold_change)
    fold_change = np.where(np.isnan(fold_change), min_fc, fold_change)

    # Create DataFrame
    df = pd.DataFrame({
        'Volume': np.log10(Volume),
        'fold change': fold_change,
        'Density': np.log2(density),
        'Droplet': droplet_id,
        'Google Drive Link': google_drive_url
    })

    # Plotting
    p = figure(title='Log2 Fold Change vs. Log2 Density colored by Volume',
               x_axis_label='Log2 Density', y_axis_label='Log2 Fold Change',
               output_backend="webgl", width=800, height=600)

    source = ColumnDataSource(df)

    # Color Mapping: Map Log10 Volume (approx 3 to 7.5) to Jet colormap
    jet_palette = [RGB(*[int(255 * c) for c in cm.jet(i)[:3]]).to_hex() for i in range(256)]
    color_mapper = LinearColorMapper(palette=jet_palette, low=3, high=7.5)

    p.scatter(x='Density', y='fold change', source=source,
              color=linear_cmap('Volume', jet_palette, 3, 7.5), alpha=0.8)

    # Add Color Bar
    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), title='Log10 Volume')
    p.renderers.append(color_bar)
    p.add_layout(color_bar, 'right')

    # Tools
    hover = HoverTool(tooltips=[('Log2 Density', '@Density'), ('Log2 Fold Change', '@{fold change}'),
                                ('Droplet', '@Droplet'), ('Log10 Volume', '@Volume')])
    p.add_tools(hover)

    p.add_tools(TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            window.open(source.data['Google Drive Link'][selected_index], "_blank");
        }
    """)))

    return p


def FC_vs_Volume(data_dict):
    """
    Scatter plot: Log2 Fold Change vs. Log2 Volume.

    Color: Points are colored by Initial Density.
    Purpose: To check if droplet size affects yield, while visualizing density effects.
    """
    # (Similar data extraction logic as FC_vs_density...)
    fold_change = np.array([])
    density = np.array([])
    Volume = np.array([])
    droplet_id = np.array([])
    google_drive_url = np.array([])
    min_fc = -10

    for key, value in data_dict.items():
        if value['Count'].iloc[0] == 0:
            continue

        if value['Count'].iloc[-4:].mean() == 0:
            fold_change = np.append(fold_change, np.nan)
        else:
            fold_change = np.append(fold_change, value['Count'].iloc[-4:].mean() / value['Count'].iloc[0])

        Volume = np.append(Volume, value['Volume'].iloc[0])
        density = np.append(density, value['Count'].iloc[0] / value['Volume'].iloc[0])
        droplet_id = np.append(droplet_id, value['Droplet'].iloc[0])
        google_drive_url = np.append(google_drive_url, value['Google Drive Link'].iloc[0])

    fold_change = np.log2(fold_change)
    fold_change = np.where(np.isnan(fold_change), min_fc, fold_change)

    df = pd.DataFrame({
        'Volume': np.log2(Volume),
        'fold change': fold_change,
        'Density': np.log2(density),
        'Droplet': droplet_id,
        'Google Drive Link': google_drive_url
    })

    p = figure(title='Log2 Fold Change vs. Log2 Volume colored by Density',
               x_axis_label='Log2 Volume', y_axis_label='Log2 Fold Change',
               output_backend="webgl", width=800, height=600)

    source = ColumnDataSource(df)

    # Color Mapping: Map Log2 Density (approx -12 to -4) to Jet colormap
    jet_palette = [RGB(*[int(255 * c) for c in cm.jet(i)[:3]]).to_hex() for i in range(256)]
    color_mapper = LinearColorMapper(palette=jet_palette, low=-12, high=-4)

    p.scatter(x='Volume', y='fold change', source=source,
              color=linear_cmap('Density', jet_palette, -12, -4), alpha=0.8)

    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), title='Log2 Density')
    p.renderers.append(color_bar)
    p.add_layout(color_bar, 'right')

    hover = HoverTool(tooltips=[('Log2 Density', '@Density'), ('Log2 Fold Change', '@{fold change}'),
                                ('Droplet', '@Droplet'), ('Log2 Volume', '@Volume')])
    p.add_tools(hover)

    p.add_tools(TapTool(callback=CustomJS(args=dict(source=source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            window.open(source.data['Google Drive Link'][selected_index], "_blank");
        }
    """)))

    return p


def dashborde():
    """
    MAIN DATA PROCESSING PIPELINE.

    1. Splits raw data into chips.
    2. Calculates stats for each chip.
    3. Generates all enabled plots for each chip.
    4. Returns a dictionary of Layouts (one per Chip).
    """
    chips = split_data_to_chips()
    initial_densities = initial_stats(chips)

    # Calculate global initial density per chip
    for key, value in initial_densities.items():
        density = value['Count'].sum() / value['Volume'].sum()
        initial_densities[key] = density

    # Spatial Filtering & Small Droplet Removal
    for key, value in chips.items():
        chips[key] = find_droplet_location(value)
        # Filter droplets smaller than 10^3 (LogVol=3)
        chips[key] = chips[key][chips[key]['log_Volume'] >= 3]

    initial_data = initial_stats(chips)
    layouts = {}

    # --- Per-Chip Analysis Loop ---
    for key, value in initial_data.items():
        # Prepare chip data structure
        chip, experiment_time, time_steps = get_slice(chips, key)

        # --- Generate Plots ---
        # stats_box_plot = stats_box(value, experiment_time, time_steps, key)
        # droplets_histogram_plot = droplet_histogram(value)
        # Calculates Vc (closest_volume) where density converges
        Initial_Density_Vs_Volume_plot, volume = Initial_Density_Vs_Volume(value, initial_densities[key])
        # N0_Vs_Volume_plot = N0_Vs_Volume(value, volume)
        # Fraction_in_each_bin_plot = Distribution_comparison(chip, experiment_time)
        # growth_curves_plot = growth_curves(chip)
        # normalize_Max_growth_curves_plot = normalize_growth_curves(chip)
        normalize_first_timepoint_growth_curves_plot = normalize_growth_curves_first_timepoint(chip)
        fold_change_plot = fold_change(chip, volume)
        last_4_hours_average_plot = last_4_hours_average(chip, volume)
        # death_rate_by_droplets_plot = death_rate_by_droplets(chip, key)
        # death_rate_by_bins_plot = death_rate_by_bins(chip)
        # distance_Vs_Volume_histogram_plot = distance_Vs_Volume_histogram(value)
        # distance_Vs_occupide_histogram_plot = distance_Vs_occupide_histogram(value)
        # distance_Vs_Volume_circle_plot = distance_Vs_Volume_circle(value)
        # distance_Vs_occupide_circle_plot = distance_Vs_occupide_circle(value)
        # distance_Vs_Volume_colored_by_death_rate_plot = distance_Vs_Volume_colored_by_death_rate(value, chip, key)
        # distance_Vs_Volume_colored_by_fold_change_plot = distance_Vs_Volume_colored_by_fold_change(value, chip)
        # bins_volume_Vs_distance_plot = bins_volume_Vs_distance(chip, key)
        # FC_vs_density_plot = FC_vs_density(chip)
        # FC_vs_Volume_plot = FC_vs_Volume(chip)

        # --- Organize Layout ---
        # Current Layout: Fold Change, Last 4 Hours, Normalized Growth Curves
        layout = column(
            row(fold_change_plot, last_4_hours_average_plot),
            normalize_first_timepoint_growth_curves_plot,
        )
        layouts[key] = layout

    return layouts


def create_dashboard():
    """
    ENTRY POINT.
    Generates the HTML file and sets up the Dropdown Menu for chip selection.
    """
    # 1. Generate all plots
    layouts = dashborde()

    # 2. Output Setting
    output_file('dashboard.html')

    # 3. Create Dropdown Select Widget
    select = Select(title="Select Chip", options=list(layouts.keys()), value=list(layouts.keys())[0])

    # 4. Stack all layouts into one column
    all_layouts_column = column(*[layout for layout in layouts.values()], name="all_layouts")

    # 5. Hide all but the first layout initially
    for layout in all_layouts_column.children:
        layout.visible = False
    layouts[select.value].visible = True

    # 6. Add JavaScript for Interactivity (Switching Chips)
    select.js_on_change('value',
                        CustomJS(args=dict(layouts=layouts, all_layouts_column=all_layouts_column, select=select), code="""
        // Hide all layouts
        for (const layout of all_layouts_column.children) {
            layout.visible = false;
        }
        // Show the selected layout
        layouts[select.value].visible = true;
    """))

    # 7. Render
    show(column(select, all_layouts_column))


if __name__ == '__main__':
    create_dashboard()

