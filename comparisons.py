from bokeh.layouts import column, row
from bokeh.models import Select, CustomJS, Spacer, Range1d
from bokeh.plotting import output_file, show

# Import the main analysis script to reuse its data processing and plotting functions
import main


def dashborde():
    """
    Pre-calculates ALL plots for ALL chips and organizes them into a dictionary.

    Structure:
    {
        'plot_type_1': [Plot_Chip_A, Plot_Chip_B, Plot_Chip_C...],
        'plot_type_2': [Plot_Chip_A, Plot_Chip_B, Plot_Chip_C...],
        ...
    }

    This pre-calculation allows for instant switching between chips in the dashboard
    without needing a Python server backend (pure HTML/JS).
    """
    # --- Step 1: Data Loading & Preprocessing (Reused from main.py) ---
    chips = main.split_data_to_chips()
    initial_densities = main.initial_stats(chips)

    # Calculate global initial density per chip
    for key, value in initial_densities.items():
        density = value['Count'].sum() / value['Volume'].sum()
        initial_densities[key] = density

    # Spatial Filtering: Remove edge droplets & small noise (< 10^3)
    for key, value in chips.items():
        chips[key] = main.find_droplet_location(value)
        chips[key] = chips[key][chips[key]['log_Volume'] >= 3]

    initial_data = main.initial_stats(chips)

    # --- Step 2: Initialize Plot Storage ---
    # Keys correspond to the plot options in the dropdown menu
    plot_types = {
        'droplet_histogram': [],
        'N0_Vs_Volume': [],
        'Initial_Density_Vs_Volume': [],
        'Fraction_in_each_bin': [],
        'growth_curves': [],
        'normalize_growth_curves': [],
        'fold_change': [],
        'last_4_hours_average': [],
        'death_rate_by_droplets': [],
        'death_rate_by_bins': [],
        'distance_Vs_Volume_histogram': [],
        'distance_Vs_occupide_histogram': [],
        'distance_Vs_Volume_circle': [],
        'distance_Vs_occupide_circle': [],
        'distance_Vs_Volume_colored_by_death_rate': [],
        'distance_Vs_Volume_colored_by_fold_change': [],
        'bins_volume_Vs_distance': [],
        'FC_vs_density': [],
        'FC_vs_Volume': []
    }

    # --- Step 3: Generation Loop ---
    # Iterate through every chip and generate every plot type for it
    for key, value in initial_data.items():
        chip, experiment_time, time_steps = main.get_slice(chips, key)

        # 1. Droplet Histogram
        droplet_histogram_column = main.droplet_histogram(value)
        droplet_histogram_column.children[0].update(title=f'Histogram of Droplet Size for {key}')
        # Standardize Y-range for valid side-by-side comparison
        droplet_histogram_column.children[0].y_range = Range1d(start=0, end=1800)
        plot_types['droplet_histogram'].append(droplet_histogram_column)

        # 2. Initial Density vs Volume (Calculation of Vc)
        Initial_Density_Vs_Volume_column, volume = main.Initial_Density_Vs_Volume(value, initial_densities[key])
        Initial_Density_Vs_Volume_column.update(title=f'Initial Density Vs Volume for {key}')
        Initial_Density_Vs_Volume_column.y_range = Range1d(start=10 ** (-6), end=10 ** (-0.5))
        plot_types['Initial_Density_Vs_Volume'].append(Initial_Density_Vs_Volume_column)

        # 3. N0 vs Volume (Poisson Check)
        N0_Vs_Volume_column = main.N0_Vs_Volume(value, volume)
        N0_Vs_Volume_column.children[0].update(title=f'N0 Vs Volume for {key}')
        N0_Vs_Volume_column.children[0].y_range = Range1d(start=1, end=10 ** 5)
        plot_types['N0_Vs_Volume'].append(N0_Vs_Volume_column)

        # 4. Fraction in Bin
        Fraction_in_each_bin_column = main.Fraction_in_each_bin(chip, experiment_time)
        Fraction_in_each_bin_column.update(title=f'Fraction of Population in Each Bin at Start for {key}')
        plot_types['Fraction_in_each_bin'].append(Fraction_in_each_bin_column)

        # 5. Growth Curves (Linear & Log)
        growth_curves_column = main.growth_curves(chip)
        # Re-stack them vertically for this dashboard view
        growth_curves_column = column(
            growth_curves_column.children[0],  # Linear scale plot
            growth_curves_column.children[1]  # Log scale plot
        )
        growth_curves_column.children[0].update(title=f'Growth Curves for {key}')
        growth_curves_column.children[1].update(title=f'Growth Curves log scale for {key}')
        # Fixed ranges for comparison
        growth_curves_column.children[0].y_range = Range1d(start=0, end=10 ** 7.5)
        growth_curves_column.children[1].y_range = Range1d(start=1, end=10 ** 7.5)
        plot_types['growth_curves'].append(growth_curves_column)

        # 6. Normalized Growth Curves (To Max)
        normalize_growth_curves_column = main.normalize_growth_curves(chip)
        normalize_growth_curves_column = column(
            normalize_growth_curves_column.children[0],  # Linear scale plot
            normalize_growth_curves_column.children[1]  # Log scale plot
        )
        normalize_growth_curves_column.children[0].update(title=f'Normalized Growth Curves for {key}')
        normalize_growth_curves_column.children[1].update(title=f'Normalized Growth Curves log scale for {key}')
        normalize_growth_curves_column.children[0].y_range = Range1d(start=0, end=1.3)
        normalize_growth_curves_column.children[1].y_range = Range1d(start=0.0085, end=1.3)
        plot_types['normalize_growth_curves'].append(normalize_growth_curves_column)
        # 7. Fold Change vs Volume
        fold_change_column = main.fold_change(chip, volume)
        fold_change_column.update(title=f'Fold Change for {key}')
        fold_change_column.y_range = Range1d(start=-10.5, end=10)
        plot_types['fold_change'].append(fold_change_column)

        # 8. Last 4 Hours Average
        last_4_hours_average_column = main.last_4_hours_average(chip, volume)
        last_4_hours_average_column.update(title=f'Average Number of Bacteria in Last 4 Hours for {key}')
        last_4_hours_average_column.y_range = Range1d(start=0.05, end=10 ** 6)
        plot_types['last_4_hours_average'].append(last_4_hours_average_column)

        # 9. Death Rate (Droplets)
        death_rate_by_droplets_column = main.death_rate_by_droplets(chip, key)
        death_rate_by_droplets_column.update(title=f'Slope by Droplet Size for {key}')
        death_rate_by_droplets_column.y_range = Range1d(start=-1.2, end=1.8)
        plot_types['death_rate_by_droplets'].append(death_rate_by_droplets_column)

        # 10. Death Rate (Bins)
        death_rate_by_bins_column = main.death_rate_by_bins(chip)
        death_rate_by_bins_column.y_range = Range1d(start=-2.5, end=1.2)
        death_rate_by_bins_column.update(title=f'Slope by Bin for {key}')
        plot_types['death_rate_by_bins'].append(death_rate_by_bins_column)

        # 11. Spatial Histogram (Volume)
        distance_Vs_Volume_histogram_column = main.distance_Vs_Volume_histogram(value)
        distance_Vs_Volume_histogram_column.update(
            title=f'Normalized Stacked Histogram: Distance vs. Log Volume for {key}')
        plot_types['distance_Vs_Volume_histogram'].append(distance_Vs_Volume_histogram_column)

        # 12. Spatial Histogram (Occupied)
        distance_Vs_occupide_histogram_column = main.distance_Vs_occupide_histogram(value)
        distance_Vs_occupide_histogram_column.update(
            title=f'Normalized Stacked Histogram: Distance vs. Log Volume Occupied for {key}')
        plot_types['distance_Vs_occupide_histogram'].append(distance_Vs_occupide_histogram_column)

        # 13. Spatial Map (Volume)
        distance_Vs_Volume_circle_column = main.distance_Vs_Volume_circle(value)
        distance_Vs_Volume_circle_column.update(title=f'Distance to Center vs. Volume for {key}')
        plot_types['distance_Vs_Volume_circle'].append(distance_Vs_Volume_circle_column)

        # 14. Spatial Map (Occupied)
        distance_Vs_occupide_circle_column = main.distance_Vs_occupide_circle(value)
        distance_Vs_occupide_circle_column.update(title=f'Distance to Center vs. Volume Occupied for {key}')
        plot_types['distance_Vs_occupide_circle'].append(distance_Vs_occupide_circle_column)

        # 15. Spatial Heatmap (Death Rate)
        distance_Vs_Volume_colored_by_death_rate_column = main.distance_Vs_Volume_colored_by_death_rate(value, chip,
                                                                                                        key)
        # Note: Accessing children[1] because main.py returns row(checkbox, plot)
        distance_Vs_Volume_colored_by_death_rate_column.children[1].update(
            title=f'Distance to Center vs. Volume Colored by Slope for {key}')
        plot_types['distance_Vs_Volume_colored_by_death_rate'].append(distance_Vs_Volume_colored_by_death_rate_column)

        # 16. Spatial Heatmap (Fold Change)
        distance_Vs_Volume_colored_by_fold_change_column = main.distance_Vs_Volume_colored_by_fold_change(value, chip)
        distance_Vs_Volume_colored_by_fold_change_column.children[1].update(
            title=f'Distance to Center vs. Volume Colored by Fold Change for {key}')
        plot_types['distance_Vs_Volume_colored_by_fold_change'].append(distance_Vs_Volume_colored_by_fold_change_column)

        # 17. Metrics vs Distance
        bins_volume_Vs_distance_column = main.bins_volume_Vs_distance(chip, key)
        # Reformat from Row (main.py) to Column (comparisons.py) to fit side-by-side
        bins_volume_Vs_distance_column = column(
            bins_volume_Vs_distance_column.children[0],
            bins_volume_Vs_distance_column.children[1]
        )
        bins_volume_Vs_distance_column.children[0].y_range = Range1d(start=-10.5, end=10)
        bins_volume_Vs_distance_column.children[1].y_range = Range1d(start=-1.2, end=0.8)
        bins_volume_Vs_distance_column.children[0].update(
            title=f'Bin volumes vs. Distance to Center for {key} by Mean Fold Change')
        bins_volume_Vs_distance_column.children[1].update(
            title=f'Bin volumes vs. Distance to Center for {key} by Mean Death Rate')
        plot_types['bins_volume_Vs_distance'].append(bins_volume_Vs_distance_column)

        # 18. FC vs Density
        FC_vs_density_column = main.FC_vs_density(chip)
        FC_vs_density_column.update(title=f'Fold Change vs. Density for {key}')
        FC_vs_density_column.y_range = Range1d(start=-10.5, end=10)
        FC_vs_density_column.x_range = Range1d(start=-20, end=-1)
        plot_types['FC_vs_density'].append(FC_vs_density_column)

        # 19. FC vs Volume
        FC_vs_Volume_column = main.FC_vs_Volume(chip)
        FC_vs_Volume_column.update(title=f'Fold Change vs. Volume for {key}')
        FC_vs_Volume_column.y_range = Range1d(start=-10.5, end=10)
        FC_vs_Volume_column.x_range = Range1d(start=9, end=25)
        plot_types['FC_vs_Volume'].append(FC_vs_Volume_column)

    return plot_types, list(chips.keys())

def create_dashboard():
    """
    Sets up the Comparative Dashboard UI.
    Creates 3 selection widgets (Plot Type, Chip 1, Chip 2) and defines the
    JavaScript callbacks to update the view dynamically.
    """
    # Load all plots
    plot_types, chips_names = dashborde()

    output_file('comparisons.html')

    # --- UI Widgets ---
    select = Select(title="Select plot", options=list(plot_types.keys()), value=list(plot_types.keys())[0])
    first_chip_selection = Select(title="Select chip (Left)", options=chips_names, value=chips_names[0])
    second_chip_selection = Select(title="Select chip (Right)", options=chips_names, value=chips_names[1])

    # Layout Spacer (to separate left and right plots)
    spacer = Spacer(width=75)

    # Initial View: Row containing [Left Plot, Spacer, Right Plot]
    # We grab the plots corresponding to the initial indices [0] and [1]
    plot_layout = row(plot_types[select.value][0], spacer, plot_types[select.value][1])

    # --- CustomJS Callbacks ---

    # Callback 1: Main Update Logic
    # Triggered when ANY of the 3 dropdowns change.
    # It finds the index of the selected chips and retrieves the correct plot from 'plot_types'.
    callback = CustomJS(args=dict(plot_types=plot_types, chips_names=chips_names,
                                    select=select, first_chip_selection=first_chip_selection,
                                    second_chip_selection=second_chip_selection, plot_layout=plot_layout,
                                    spacer=spacer),
                        code="""
            var selected_plot = select.value;
            var first_chip = first_chip_selection.value;
            var second_chip = second_chip_selection.value;

            // Find list index of the selected chips
            var first_chip_index = chips_names.indexOf(first_chip);
            var second_chip_index = chips_names.indexOf(second_chip);

            // Update the layout children: [Plot A, Spacer, Plot B]
            plot_layout.children = [plot_types[selected_plot][first_chip_index],
                                        spacer,
                                        plot_types[selected_plot][second_chip_index]];
        """)

    # Callback 2: Prevent Duplicate Selection (Update Right Options)
    # When Left Chip changes, remove that chip from Right Chip options.
    update_second_chip_options = CustomJS(args=dict(first_chip_selection=first_chip_selection,
                                                    second_chip_selection=second_chip_selection,
                                                    chips_names=chips_names),
                                            code="""
            var first_chip = first_chip_selection.value;
            var options = chips_names.filter(chip => chip !== first_chip);
            second_chip_selection.options = options;

            // If the current selection became invalid, switch to first available option
            if (second_chip_selection.value === first_chip) {
                second_chip_selection.value = options[0];
            }
        """)

    # Callback 3: Prevent Duplicate Selection (Update Left Options)
    # When Right Chip changes, remove that chip from Left Chip options.
    update_first_chip_options = CustomJS(args=dict(first_chip_selection=first_chip_selection,
                                                    second_chip_selection=second_chip_selection,
                                                    chips_names=chips_names),
                                            code="""
            var second_chip = second_chip_selection.value;
            var options = chips_names.filter(chip => chip !== second_chip);
            first_chip_selection.options = options;

            if (first_chip_selection.value === second_chip) {
                first_chip_selection.value = options[0];
            }
        """)

    # Attach Callbacks
    select.js_on_change('value', callback)
    first_chip_selection.js_on_change('value', callback)
    second_chip_selection.js_on_change('value', callback)

    # Attach Options Filters
    first_chip_selection.js_on_change('value', update_second_chip_options)
    second_chip_selection.js_on_change('value', update_first_chip_options)

    # Final Layout
    layout = column(select, first_chip_selection, second_chip_selection, plot_layout)
    show(layout)

if __name__ == '__main__':
    create_dashboard()
