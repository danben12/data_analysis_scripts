"""
Script Name: stitcher 2.0.py
Description: 
    Merges physical droplet data (from image analysis) with bacterial count data.
    It calculates the 3D volume of each droplet based on its 2D area and a specific 
    contact angle, then bins the data and performs spatial filtering.

    The output is a 'merged' CSV file ready for the main analysis pipeline.
"""

import pandas as pd
import numpy as np

# List of chips to process
chips = ['C1']

# Dictionary to map short codes to full names (if needed)
chips_full_names = {
    'C1': "C1"
}

# --- Main Processing Loop ---
for chip in chips:
    # 1. Load Data
    # Load physical droplet data (Area, X, Y coordinates)
    droplets_data = pd.read_csv(
        rf"L:\12032025_bocillin 80,60 and 40 uM_tris 50 mM_strain 293\C1\Alexa\best LUT\csv\{chip} final data.csv")
    # Load bacterial count data (Time series counts)
    counts_data = pd.read_csv(
        rf"L:\12032025_bocillin 80,60 and 40 uM_tris 50 mM_strain 293\C1\mCherry\counts\bacteria_counts_{chip}.csv")

    # 2. Merge Datasets
    # Link bacteria to droplets using the unique ID ('Droplet' in counts match 'Label' in droplets)
    merged = pd.merge(counts_data, droplets_data, left_on='Droplet', right_on='Label', how='left')

    # 3. Unit Conversion & Renaming
    # Convert physical Area back to pixel units (assuming pixel size 0.16 microns)
    # Note: This integer cast creates a 'pixel_Area' column for reference
    merged['pixel_Area'] = (merged['Area'] / (0.16 * 0.16)).astype(int)

    # Rename columns to standard names used in downstream scripts (main.py)
    merged = merged.rename(columns={'Chip': 'Slice', 'Bacteria_Area': 'Count', 'Hour': 'time'})

    # 4. Reorder Columns
    desired_order = ['Slice', 'Count', 'Droplet', 'pixel_Area', 'Area', 'X', 'Y', 'time']
    merged = merged[desired_order]

    # 5. Create Unique Identifiers
    # Re-index droplets to be 1-based integers
    merged['Droplet'] = pd.factorize(merged['Droplet'])[0] + 1
    merged['Well'] = chip
    merged['initialOD'] = 0.05

    # Create a unique "Droplet_Well" ID (DW)
    merged['DW'] = merged['Droplet'].astype(str) + '_' + merged['Well'].astype(str)
    merged['Slice'] = chips_full_names[chip]

    # 6. Volume Calculation (Geometric Model)
    # Convert 2D Area to 3D Volume using Sessile Drop Model (Spherical Cap)
    # Assumes a Contact Angle (Theta) of 32 degrees
    Theta = np.radians(32)

    # Calculate Diameter (D) from Area
    D = 2 * np.sqrt(merged['Area'] / np.pi)

    # Volume Formula: V = (π * D³ / 24) * f(θ)
    merged.loc[:, 'Volume'] = ((np.pi * D ** 3) / 24) * (
            (2 - 3 * np.cos(Theta) + np.cos(Theta) ** 3) / (np.sin(Theta) ** 3))

    # Log-transform volume for binning
    merged.loc[:, 'log_Volume'] = np.log10(merged['Volume'])

    # 7. Binning
    # Assign droplets to Volume Bins (0-1, 1-2, ... 7-8)
    vol_labels = ['0 - 1', '1 - 2', '2 - 3', '3 - 4', '4 - 5', '5 - 6', '6 - 7', '7 - 8']
    cut_bins_vol = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    merged.loc[:, 'Bins_vol'] = pd.cut(merged['log_Volume'], bins=cut_bins_vol)
    merged.loc[:, 'Bins_vol_txt'] = pd.cut(merged['log_Volume'], bins=cut_bins_vol, labels=vol_labels)

    # 8. Spatial Filtering
    # Calculate distance from center (Assuming chip size 13200x13200 pixels)
    circle_radius = 13200 / 2
    circle_center_x = 13200 / 2
    circle_center_y = 13200 / 2

    merged['distance_to_center'] = np.sqrt((merged['X'] - circle_center_x) ** 2 + (merged['Y'] - circle_center_y) ** 2)
    merged['is_inside_circle'] = merged['distance_to_center'] <= circle_radius

    # Debug print: Check how many droplets are inside vs outside
    true_false_counts = merged['is_inside_circle'].value_counts()

    # Add empty placeholder for Google Drive Links (to be filled manually if needed)
    merged['Google Drive Link'] = ''

    # 9. Save Result
    merged.to_csv(
        rf"L:\12032025_bocillin 80,60 and 40 uM_tris 50 mM_strain 293\C1\mCherry\counts\merged_bacteria_counts_C1.csv",
        index=False)
    print(f'chip {chip} done')

# Optional: Concatenation block (Commented out in original)
# concatenated_df= pd.DataFrame()
# for chip in chips:
#     path=rf"L:\21012025_BSF obj x10\{chip}\GFP\new counts\merged_bacteria_counts{chip}.csv"
#     df = pd.read_csv(path)
#     concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
# concatenated_df.to_csv(r'L:\21012025_BSF obj x10\merged_bacteria_count.csv', index=False)