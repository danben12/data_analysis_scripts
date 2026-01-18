"""
================================================================================
SCRIPT: Data Smoothing & Gap Filling (LOWESS)
================================================================================
AUTHOR: Dan
LAST UPDATED: January 2026

DESCRIPTION:
This script performs advanced post-processing on the raw bacterial count data.
Raw time-series data often contains:
1. "Zero" values that are actually missing data (focus artifacts).
2. Noise/Jitter in the bacterial count over time.

This pipeline fixes these issues by:
1. Converting suspicious "Zero" starts to NaNs.
2. Filling gaps using a Log-Scale Moving Average (geometric mean).
3. Smoothing the final growth curves using LOWESS (Locally Weighted Scatterplot Smoothing).

The script uses parallel processing (ProcessPoolExecutor) to handle large datasets efficiently.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import concurrent.futures

# Note: 'main' is imported but not strictly used in the cleaned functions below.
# It might be used for 'find_droplet_location' if you uncomment the filtering block.
import main


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def split_data_to_chips(df):
    """
    Splits the main dataframe into a dictionary of dataframes, one per Chip (Well).
    This allows us to process each chip in parallel.

    Args:
        df (pd.DataFrame): The monolithic dataframe containing all data.

    Returns:
        dict: { 'Chip_Name': pd.DataFrame, ... }
    """
    chips = {}
    for chip in df['Well'].unique():
        chips[chip] = df[df['Well'] == chip]
    return chips


def process_chip_replace_zero_with_nan(chip_df):
    """
    Worker function: Scans one chip's data.
    If a droplet STARTS with 0 bacteria but immediately jumps to a non-zero count
    in the next few frames, that initial '0' is likely an artifact (not real extinction).

    Action: Replaces the initial '0' with NaN so it can be back-filled later.
    """
    for droplet in chip_df['Droplet'].unique():
        print(f'Processing chip {chip_df["Well"].iloc[0]}, droplet {droplet}')

        # Extract the time-series for this specific droplet
        series = chip_df[chip_df['Droplet'] == droplet]['Count']

        # Check: Is the first point 0? AND Are any of the next 5 points non-zero?
        # If yes, it's likely a focus issue at T=0, not a sterile droplet.
        if series.iloc[0] == 0 and (series.iloc[1:6] != 0).any():
            # Set the first value to NaN
            chip_df.loc[chip_df[chip_df['Droplet'] == droplet].index[0], 'Count'] = np.nan

    return chip_df


def replace_zero_with_nan(df):
    """
    Parallel wrapper for 'process_chip_replace_zero_with_nan'.
    """
    chips = split_data_to_chips(df)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_chip_replace_zero_with_nan, chips.values()))
    return pd.concat(results)


def process_chip_log_mean_fill(chip_df):
    """
    Worker function: Fills gaps (zeros/NaNs) in the time series using Log-Interpolation.

    Logic:
    - Biological growth is exponential (linear in Log scale).
    - If we have a gap [100, 0, 400], a linear average (250) is wrong.
    - A Log-average (Geometric mean) is better: 10^((log(100)+log(400))/2) = 200.
    """
    epsilon = 1e-6  # Avoid log(0) errors

    for droplet in chip_df['Droplet'].unique():
        print(f'Processing chip {chip_df["Well"].iloc[0]}, droplet {droplet}')

        series = chip_df[chip_df['Droplet'] == droplet]['Count']
        result = series.copy()

        # 1. Back-fill start: If T=0 is NaN, copy the first valid value seen later
        if pd.isna(result.iloc[0]):
            for idx in range(1, len(result)):
                if pd.notna(result.iloc[idx]):
                    result.iloc[0] = result.iloc[idx]
                    break

        # 2. Forward-fill end: If T=End is NaN, copy the last valid value
        if pd.isna(result.iloc[-1]):
            for idx in range(len(result) - 2, -1, -1):
                if pd.notna(result.iloc[idx]):
                    result.iloc[-1] = result.iloc[idx]
                    break

        # 3. Gap Filling (Middle points)
        for idx in range(1, len(series) - 1):
            if series.iloc[idx] == 0 or pd.isna(series.iloc[idx]):
                # Look at neighbors
                before = series.iloc[idx - 1] if idx - 1 >= 0 else np.nan
                after = series.iloc[idx + 1] if idx + 1 < len(series) else np.nan

                # Sanitize neighbors
                before = before if before > 0 else epsilon
                after = after if after > 0 else epsilon

                # Geometric Mean Interpolation
                if pd.notna(before) and pd.notna(after):
                    result.iloc[idx] = 10 ** ((np.log10(before) + np.log10(after)) / 2)
                else:
                    # Fallback: Just hold the previous value
                    for j in range(1, idx + 1):
                        prev_value = series.iloc[idx - j] if idx - j >= 0 else np.nan
                        if pd.notna(prev_value) and prev_value > 0:
                            result.iloc[idx] = prev_value
                            break

        # Round to integers (you can't have 0.5 bacteria)
        result = result.round().astype(int)
        chip_df.loc[chip_df['Droplet'] == droplet, 'Count'] = result.values

    return chip_df


def log_mean_fill(df):
    """
    Parallel wrapper for 'process_chip_log_mean_fill'.
    """
    chips = split_data_to_chips(df)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_chip_log_mean_fill, chips.values()))
    return pd.concat(results)


def process_chip_lowess(chip_df):
    """
    Worker function: Applies LOWESS Smoothing to the growth curves.

    Why LOWESS?
    - Biological data is noisy.
    - LOWESS (Locally Weighted Scatterplot Smoothing) fits a smooth curve 
      without assuming a specific shape (unlike linear regression).
    - We smooth on Log-Transformed data (log_counts) because bacterial growth
      noise scales with population size.
    """
    for droplet in chip_df['Droplet'].unique():
        print(f'Processing chip {chip_df["Well"].iloc[0]}, droplet {droplet}')

        series = chip_df[chip_df['Droplet'] == droplet]['Count']
        result = series.copy()
        x = np.arange(len(result))

        # Log-transform before smoothing
        log_counts = np.log10(result + 1)

        # Apply LOWESS
        # frac=0.2 means 20% of the data points are used for each local smoothing window
        lowess = sm.nonparametric.lowess(log_counts, x, frac=0.2)

        # Transform back to linear scale
        smoothed_values = np.round(10 ** lowess[:, 1] - 1).astype(int)

        chip_df.loc[chip_df['Droplet'] == droplet, 'Count'] = smoothed_values

    return chip_df


def apply_lowess(df):
    """
    Parallel wrapper for 'process_chip_lowess'.
    """
    chips = split_data_to_chips(df)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_chip_lowess, chips.values()))
    return pd.concat(results)


# ==========================================
# 2. MAIN EXECUTION
# ==========================================

if __name__ == '__main__':
    # 1. Load Raw Data
    # [!] Update path as needed
    input_csv_path = r"L:\21012025_BSF obj x10\first_24_bacteria_count.csv"
    output_csv_path = r"L:\21012025_BSF obj x10\first_24_bacteria_count_filled.csv"

    print("Loading data...")
    df = pd.read_csv(input_csv_path)

    # 2. Clean 'Fake' Zeroes at the start
    print("Cleaning start-point zeros...")
    df = replace_zero_with_nan(df)

    # 3. Fill Gaps using Log-Interpolation
    print("Filling gaps (Log-Mean)...")
    filled_df = log_mean_fill(df)

    # 4. Smooth Curves using LOWESS
    print("Smoothing curves (LOWESS)...")
    filled_df = apply_lowess(filled_df)

    # 5. Save
    print(f"Saving to {output_csv_path}...")
    filled_df.to_csv(output_csv_path, index=False)
    print("Done.")