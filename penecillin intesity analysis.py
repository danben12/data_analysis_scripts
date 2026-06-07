import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import itertools
import matplotlib.cm as cm
from scipy import stats
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr



droplets_data = pd.read_csv(r"C:\Users\danbe\Desktop\37C investigation\Bocillin FL\C1 final data.csv")
bac_data = pd.read_csv(r"C:\Users\danbe\Desktop\37C investigation\Bocillin FL\mCherry Final_Intensity_Analysis_Formatted.csv")
signal_data= pd.read_csv(r"C:\Users\danbe\Desktop\37C investigation\Bocillin FL\GFP Final_Intensity_Analysis_Formatted.csv")



signal_data=signal_data[signal_data['Mean_Top25_Intensity']>=0]
# bac_data=bac_data[bac_data['Mean_Top25_Intensity']<=6000]
# signal_data=signal_data[signal_data['Mean_Top25_Intensity']<=500]
signal_data=signal_data.rename(columns={'Mean_Top25_Intensity':'signal intesity'})
merged = pd.merge(bac_data, droplets_data, left_on='Droplet', right_on='Label', how='left')
merged = pd.merge(merged, signal_data[['Bacterium', 'signal intesity']], on='Bacterium', how='left')
# merged['Area'] = merged['Area'] * 0.33 * 0.33
merged['pixel_Area'] = merged['Area'] / (0.16 * 0.16)
Theta = np.radians(32)
D = 2 * np.sqrt(merged['Area'] / np.pi)
merged.loc[:, 'Volume'] = ((np.pi * D ** 3) / 24) * (
                (2 - 3 * np.cos(Theta) + np.cos(Theta) ** 3) / (np.sin(Theta) ** 3))
merged.loc[:, 'log_Volume'] = np.log10(merged['Volume'])
merged=merged[merged['log_Volume']>=3]
cut_bins_vol = [3, 4, 5, 6, 7, 8]
vol_labels = ['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8']
merged.loc[:, 'Bins_vol'] = pd.cut(merged['log_Volume'], bins=cut_bins_vol)
merged.loc[:, 'Bins_vol_txt'] = pd.cut(merged['log_Volume'], bins=cut_bins_vol, labels=vol_labels)
merged['Bins_vol'] = merged['Bins_vol'].astype(str)
merged['intesity ratio']= merged['Mean_Top25_Intensity']/merged['signal intesity']
# merged=pd.read_csv(r"L:\12032025_bocillin 80,60 and 40 uM_tris 50 mM_strain 293\C1\final_data.csv",encoding='ISO-8859-1')
merged=merged[merged['signal intesity']<=1000]
t5_per_droplet_DAPI = merged.groupby('Droplet').agg({
    'Mean_Top25_Intensity': 'mean',
    'signal intesity': 'mean',
    'Volume': 'first',
    'Bins_vol': 'first',
    'log_Volume': 'first'
}).reset_index()




# --------------------------------------------------------------------------------------
# figure 1 Volume Vs GFP Intensity per bacterium T=5
# --------------------------------------------------------------------------------------

rho, p_val = spearmanr(np.log10(merged['Volume']), merged['signal intesity'])
plt.figure(figsize = (10,6))
plt.scatter(merged['Volume'], merged['signal intesity'], s=5, alpha=0.5, c='gray')
window = 1000
t5_sorted = merged.sort_values('Volume').reset_index(drop=True)
t5_sorted['signal_MA'] = t5_sorted['signal intesity'].rolling(window=window, min_periods=25).mean()
plt.plot(t5_sorted['Volume'], t5_sorted['signal_MA'], color='green', linewidth=3,
         label=f'{window}-point MA (Spearman $\\rho$ = {rho:.2f})')
plt.xlabel(r'Volume (µm$^3$)', fontsize=18)
plt.xscale('log')
plt.ylabel('GFP Intensity per bacterium', fontsize=18)
plt.ylim(0,1000)
plt.legend(fontsize=12)
plt.tick_params(axis='both', labelsize=18)
plt.grid(alpha=0.3)
plt.show()



rho, p_val = spearmanr(np.log10(t5_per_droplet_DAPI['Volume']), t5_per_droplet_DAPI['signal intesity'])
plt.figure(figsize = (10,6))
plt.scatter(t5_per_droplet_DAPI['Volume'], t5_per_droplet_DAPI['signal intesity'], s=5, alpha=0.5, c='gray')
window = 100
t5_sorted = t5_per_droplet_DAPI.sort_values('Volume').reset_index(drop=True)
t5_sorted['signal_MA'] = t5_sorted['signal intesity'].rolling(window=window, min_periods=25).mean()
plt.plot(t5_sorted['Volume'], t5_sorted['signal_MA'], color='green', linewidth=3,
         label=f'{window}-point MA (Spearman $\\rho$ = {rho:.2f})')
plt.xlabel(r'Volume (µm$^3$)', fontsize=18)
plt.xscale('log')
plt.ylabel('GFP Intensity per droplet', fontsize=18)
plt.ylim(0,1000)
plt.legend(fontsize=12,loc='upper left')
plt.tick_params(axis='both', labelsize=16)
plt.grid(alpha=0.3)
plt.show()





# --------------------------------------------------------------------------------------
# figure 7 Boxplot plot DAPI intesity per Volume bin per bacterium T=5
# -------------------------------------------------------------------------------------

order = sorted(merged['Bins_vol'].unique())
pairs = list(zip(order[:-1], order[1:]))
plt.figure(figsize=(10, 6))


# --- Base boxplot ---
ax = sns.boxplot(
    data=merged,
    x='Bins_vol',
    y='signal intesity',
    order=order,
    color='lightgray',
    width=0.6,
    linewidth=1.2,
    fliersize=0,  # hide outliers
    boxprops=dict(edgecolor='black', linewidth=1.2),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
    medianprops=dict(color='red', linewidth=2),
)

labels = [r'$10^3 - 10^4$', r'$10^4 - 10^5$', r'$10^5 - 10^6$', r'$10^6 - 10^7$', r'$10^7 - 10^8$']
ax.set_xticklabels(labels)

# --- Overlay data points ---
sns.stripplot(
    data=merged,
    x='Bins_vol',
    y='signal intesity',
    order=order,
    jitter=True,
    size=1,
    color='black',
    alpha=0.3,
    ax=ax
)

# --- Statistical annotations (neighboring only) ---
annotator = Annotator(ax, pairs, data=merged, x='Bins_vol', y='signal intesity', order=order)
annotator.configure(
    test='t-test_ind',        # or 'Mann-Whitney'
    text_format='star',
    pvalue_thresholds=[
        (1, "NS"),
        (0.05, "*"),
        (0.01, "**"),
        (0.001, "***"),
        (0.0001, "****"),
    ],
    comparisons_correction='bonferroni',
    line_offset_to_group=0.05,
    line_height=0.02,
    loc='inside',
    verbose=0,
    fontsize=18,
)
annotator.apply_and_annotate()


# --- Labels and formatting ---
plt.xlabel(r'Volume (µm$^3$)', fontsize=18)
plt.ylabel('Mean antibiotic intensity per bacterium', fontsize=18)
ax.tick_params(axis='both', labelsize=16)
sns.despine()
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
#
#
# # --------------------------------------------------------------------------------------
# # figure 13 Boxplot plot DAPI intesity per Volume bin per Droplet T=5
# # -------------------------------------------------------------------------------------
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from statannotations.Annotator import Annotator

# # --- 1. Bokeh-Style Font Configuration ---
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
# plt.rcParams['axes.labelweight'] = 'normal' # Bokeh usually uses normal weight

# # --- 2. Data Setup ---
# order = sorted(t5_per_droplet_DAPI['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))

# plt.figure(figsize=(16, 12), dpi=600)

# # --- 3. Base Boxplot ---
# ax = sns.boxplot(
#     data=t5_per_droplet_DAPI,
#     x='Bins_vol',
#     y='signal intesity',
#     order=order,
#     color='lightgray',
#     width=0.6,
#     linewidth=1.2,
#     fliersize=0,
#     boxprops=dict(edgecolor='black', linewidth=1.2),
#     whiskerprops=dict(color='black', linewidth=1.2),
#     capprops=dict(color='black', linewidth=1.2),
#     medianprops=dict(color='red', linewidth=2)
# )

# # --- 4. Overlay Data Points ---
# sns.stripplot(
#     data=t5_per_droplet_DAPI,
#     x='Bins_vol',
#     y='signal intesity',
#     order=order,
#     jitter=True,
#     size=10,
#     color='black',
#     alpha=0.3,
#     ax=ax
# )

# # --- 5. Axis Formatting (Integers, No Decimals, 200 Jumps) ---
# # X-Axis
# labels = [r'$10^3 - 10^4$', r'$10^4 - 10^5$', r'$10^5 - 10^6$', r'$10^6 - 10^7$', r'$10^7 - 10^8$']
# ax.set_xticklabels(labels, fontsize=30)

# # Y-Axis Jumps and Integer Formatting
# ytick_values = np.arange(0, 1001, 200)
# ax.set_yticks(ytick_values)
# ax.set_yticklabels([int(y) for y in ytick_values], fontsize=30)
# minor_yticks = np.arange(100, 1000, 200)
# ax.set_yticks(minor_yticks, minor=True)

# ax.tick_params(axis='y', which='major', length=12, width=2)
# ax.tick_params(axis='y', which='minor', length=6, width=2)
# ax.set_ylim(0, 1000)

# # --- 6. Statistical Annotations ---
# annotator = Annotator(ax, pairs, data=t5_per_droplet_DAPI, x='Bins_vol', y='signal intesity', order=order)
# annotator.configure(
#     test='t-test_ind',
#     text_format='star',
#     pvalue_thresholds=[(1, "NS"), (0.05, "*"), (0.01, "**"), (0.001, "***"), (0.0001, "****")],
#     comparisons_correction='bonferroni',
#     line_offset_to_group=0.05,
#     line_height=0.02,
#     loc='inside',
#     verbose=0,
#     fontsize=24,
# )
# annotator.apply_and_annotate()

# # --- 7. Labels and Styling ---
# plt.xlabel(r'Volume (µm$^3$)', fontsize=36)
# plt.ylabel('Mean antibiotic intensity per droplet', fontsize=36)

# sns.despine()
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# plt.tight_layout()

# plt.show()

# # --------------------------------------------------------------------------------------
# # figure 3 GFP Intensity Vs DAPI Intensity per bacterium T=5
# # --------------------------------------------------------------------------------------
# plt.figure(figsize = (10,6))
# plt.scatter(merged['signal intesity'], merged['Mean_Top25_Intensity'],s=1, alpha=0.4, c=merged['log_Volume'].values, cmap='jet')
# plt.xlabel('GFP Intensity')
# plt.xscale('log')
# plt.ylabel('mCherry Intensity')
# plt.yscale('log')
# plt.colorbar(label='log10(Volume)')
# plt.show()


# # --------------------------------------------------------------------------------------
# # GFP Intensity Vs DAPI Intensity per Droplet T=5
# # --------------------------------------------------------------------------------------

# plt.figure(figsize = (10,6))
# plt.scatter(t5_per_droplet_DAPI['signal intesity'], t5_per_droplet_DAPI['Mean_Top25_Intensity'],s=5, alpha=0.4, c=t5_per_droplet_DAPI['log_Volume'].values, cmap='jet')
# plt.xlabel('GFP Intensity')
# plt.xscale('log')
# plt.ylabel('mCherry Intensity')
# plt.yscale('log')
# plt.colorbar(label='log10(Volume)')
# plt.show()

# --------------------------------------------------------------------------------------
# RFP/GFP ratio Vs volume per bacterium
# --------------------------------------------------------------------------------------

# plt.figure(figsize = (10,6))
# plt.scatter(merged['Volume'], merged['intesity ratio'],s=5, alpha=0.4)
# plt.xlabel('log10(Volume)')
# plt.ylabel('RFP/GFP Intensity Ratio')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# # --------------------------------------------------------------------------------------
# # RFP/GFP ratio Vs volume per Droplet
# # --------------------------------------------------------------------------------------
# plt.figure(figsize = (10,6))
# plt.scatter(t5_per_droplet_DAPI['Volume'], t5_per_droplet_DAPI['Mean_Top25_Intensity']/t5_per_droplet_DAPI['signal intesity'],s=5, alpha=0.4)
# plt.xlabel('log10(Volume)')
# plt.ylabel('RFP/GFP Intensity Ratio')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# # --------------------------------------------------------------------------------------
# relative intesity Vs Volume per bacterium
# # --------------------------------------------------------------------------------------
plt.figure(figsize = (10,6))
window = 1000
GFP_sorted = merged.sort_values('Volume').reset_index(drop=True)
GFP_sorted['signal_MA'] = GFP_sorted['signal intesity'].rolling(window=window, min_periods=25).mean()
mCherry_sorted = merged.sort_values('Volume').reset_index(drop=True)
mCherry_sorted['bac_MA'] = mCherry_sorted['Mean_Top25_Intensity'].rolling(window=window, min_periods=25).mean()
plt.plot(GFP_sorted['Volume'], GFP_sorted['signal_MA'], color='green', linewidth=3, label=f'GFP {window}-point MA')
plt.plot(mCherry_sorted['Volume'], mCherry_sorted['bac_MA'], color='red', linewidth=3, label=f'mCherry {window}-point MA')
plt.xlabel(r'Volume (µm$^3$)', fontsize=18)
plt.ylabel('Mean Intensity per bacterium',fontsize=18)
plt.xscale('log')
plt.ylim(0, 1000)
# plt.yscale('log')
plt.legend(fontsize=12)
plt.tick_params(axis='both', labelsize=16)
plt.grid(alpha=0.3)
plt.show()
# # --------------------------------------------------------------------------------------
# relative intesity Vs Volume per droplet
# # --------------------------------------------------------------------------------------
plt.figure(figsize = (10,6))
window = 100
GFP_sorted = t5_per_droplet_DAPI.sort_values('Volume').reset_index(drop=True)
GFP_sorted['signal_MA'] = GFP_sorted['signal intesity'].rolling(window=window, min_periods=25).mean()
mCherry_sorted = t5_per_droplet_DAPI.sort_values('Volume').reset_index(drop=True)
mCherry_sorted['bac_MA'] = mCherry_sorted['Mean_Top25_Intensity'].rolling(window=window, min_periods=25).mean()
plt.plot(GFP_sorted['Volume'], GFP_sorted['signal_MA'], color='green', linewidth=3, label=f'GFP {window}-point MA')
plt.plot(mCherry_sorted['Volume'], mCherry_sorted['bac_MA'], color='red', linewidth=3, label=f'mCherry {window}-point MA')
plt.xlabel(r'Volume (µm$^3$)', fontsize=18)
plt.ylabel('Mean Intensity per droplet',fontsize=18)
plt.xscale('log')
plt.ylim(0, 1000)
plt.grid(alpha=0.3)
# plt.yscale('log')
plt.legend(fontsize=12)
plt.tick_params(axis='both', labelsize=16)
plt.show()



# intesity_data=pd.read_csv(r"L:\12032025_bocillin 80,60 and 40 uM_tris 50 mM_strain 293\C1\final_data.csv",encoding='ISO-8859-1')
# intesity_data=intesity_data.rename(columns={' ':'Droplet ID'})
# per_droplet=intesity_data.groupby('Droplet').agg({
#     'Mean_Top25_Intensity': 'mean',
#     'GFP intesity': 'mean',
#     'Droplet ID': 'first',
#     'Volume': 'first',
#     'Bins_vol': 'first',
#     'log_Volume': 'first'
# }).reset_index()
# density_data=pd.read_csv(r"L:\12032025_bocillin 80,60 and 40 uM_tris 50 mM_strain 293\C1\mCherry\counts\merged_bacteria_counts_C1.csv",encoding='ISO-8859-1')
# density_data['Density']=density_data['Count']/density_data['Volume']
# density_data=density_data[['Droplet','Density']]
# merged_intensity_density=pd.merge(intesity_data,density_data,left_on='Droplet ID',right_on='Droplet',how='left')
# plt.figure(figsize = (10,6))
# plt.scatter(merged_intensity_density['GFP intesity'], merged_intensity_density['Density'],s=5, alpha=0.4,c=merged_intensity_density['log_Volume'].values, cmap='jet')
# plt.xlabel('GFP Intensity')
# plt.xscale('log')
# plt.ylabel('Bacterial Density (cells/um^3)')
# plt.yscale('log')
# plt.colorbar(label='log10(Volume)')
# plt.show()
#
#
#
# merged_intensity_density = pd.merge(per_droplet, density_data, left_on='Droplet ID', right_on='Droplet', how='left')
# valid_data = merged_intensity_density.dropna(subset=['GFP intesity', 'Density'])
# valid_data = valid_data[(valid_data['GFP intesity'] > 0) & (valid_data['Density'] > 0)]
# log_x = np.log10(valid_data['GFP intesity'])
# log_y = np.log10(valid_data['Density'])
# r_value, p_value = pearsonr(log_x, log_y)
# plt.figure(figsize=(10, 6))
# plt.scatter(merged_intensity_density['GFP intesity'], merged_intensity_density['Density'],
#             s=5, alpha=0.4, c=merged_intensity_density['log_Volume'].values, cmap='jet')
# plt.xlabel('GFP Intensity')
# plt.xscale('log')
# plt.ylabel('Bacterial Density (cells/um^3)')
# plt.yscale('log')
# plt.colorbar(label='log10(Volume)')
# stats_text = (f"Pearson r = {r_value:.2f}")
# plt.text(1.17, 1.0, stats_text, transform=plt.gca().transAxes,
#          fontsize=10, verticalalignment='top',
#          bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
# plt.show()

