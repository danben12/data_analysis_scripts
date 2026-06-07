import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import itertools
import matplotlib.cm as cm
from scipy import stats
from scipy.stats import spearmanr


droplets_data = pd.read_csv(r"C:\Users\danbe\Desktop\37C investigation\densyl Poly B\droplets data.csv")
counts_data = pd.read_csv(r"C:\Users\danbe\Desktop\37C investigation\densyl Poly B\results_bg_corrected_per_droplet.csv")
GFP_data=pd.read_csv(r"C:\Users\danbe\Desktop\37C investigation\densyl Poly B\GFP_intesity_analysis_BG_corrected_per_droplet.csv")

merged = pd.merge(counts_data, droplets_data, left_on='Droplet', right_on='Label', how='left')
merged['Area'] = merged['Area'] * 0.33 * 0.33
# merged['pixel_Area'] = merged['Area'] / (0.33 * 0.33)
Theta = np.radians(32)
D = 2 * np.sqrt(merged['Area'] / np.pi)
merged.loc[:, 'Volume'] = ((np.pi * D ** 3) / 24) * (
                (2 - 3 * np.cos(Theta) + np.cos(Theta) ** 3) / (np.sin(Theta) ** 3))
merged.loc[:, 'log_Volume'] = np.log10(merged['Volume'])
cut_bins_vol = [3, 4, 5, 6, 7, 8,9]
vol_labels = ['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8', '8 - 9']
merged.loc[:, 'Bins_vol'] = pd.cut(merged['log_Volume'], bins=cut_bins_vol)
merged.loc[:, 'Bins_vol_txt'] = pd.cut(merged['log_Volume'], bins=cut_bins_vol, labels=vol_labels)
merged['Bins_vol'] = merged['Bins_vol'].astype(str)



t22=pd.merge(merged, GFP_data[['Bacterium','GFP intesity']], on=['Bacterium'], how='left')
t5=pd.merge(merged, GFP_data[['Bacterium','GFP intesity']], on=['Bacterium'], how='left')


t5_filtered=t5[(t5['GFP intesity'] <= 6000) & (t5['GFP intesity'] >= 100) & (t5['Mean_Top25_Intensity'] <= 220)]
t5=t5[(t5['GFP intesity'] <= 6000) &  (t5['Mean_Top25_Intensity'] <= 220)]
t5_per_droplet_filtered=t5_filtered.groupby('Droplet').agg({
    'GFP intesity': 'mean',
    'Mean_Top25_Intensity': 'mean',
    'Volume': 'first',
    'Bins_vol': 'first'
})
t5_per_droplet=t5.groupby('Droplet').agg({
    'GFP intesity': 'mean',
    'Mean_Top25_Intensity': 'mean',
    'Volume': 'first',
    'Bins_vol': 'first'
})


t5_subset_GFP=t5[t5['GFP intesity']<=6000]
t5_subset_DAPI=t5[t5['Mean_Top25_Intensity']<=220]
t5_per_droplet_GFP = t5_subset_GFP.groupby('Droplet').agg({
    'GFP intesity': 'mean',
    'Volume': 'first',
    'Bins_vol': 'first',
}).reset_index()
t5_per_droplet_DAPI = t5_subset_DAPI.groupby('Droplet').agg({
    'Mean_Top25_Intensity': 'mean',
    'Volume': 'first',
    'Bins_vol': 'first',
}).reset_index()
print(t5_per_droplet_DAPI['Bins_vol'].value_counts())
t22_subset_GFP=t22[t22['GFP intesity']<=6000]
t22_subset_DAPI=t22[t22['Mean_Top25_Intensity']<=500]
t22_per_droplet_GFP = t22_subset_GFP.groupby('Droplet').agg({
    'GFP intesity': 'mean',
    'Volume': 'first',
    'Bins_vol': 'first',
}).reset_index()
t22_per_droplet_DAPI = t22_subset_DAPI.groupby('Droplet').agg({
    'Mean_Top25_Intensity': 'mean',
    'Volume': 'first',
    'Bins_vol': 'first',
}).reset_index()


fig, ax1 = plt.subplots(figsize=(10, 6))
window = 3000
GFP_sorted = t5_filtered.sort_values('Volume').reset_index(drop=True)
GFP_sorted['signal_MA'] = GFP_sorted['Mean_Top25_Intensity'].rolling(window=window, min_periods=40).mean()
mCherry_sorted = t5.sort_values('Volume').reset_index(drop=True)
mCherry_sorted['bac_MA'] = mCherry_sorted['GFP intesity'].rolling(window=window, min_periods=40).mean()
live_mCherry_sorted = t5_filtered.sort_values('Volume').reset_index(drop=True)
live_mCherry_sorted['bac_MA']= live_mCherry_sorted['GFP intesity'].rolling(window=window, min_periods=40).mean()
color1 = 'blue'
ax1.set_xlabel(r'Volume (µm$^3$)', fontsize=18)
ax1.set_ylabel('DAPI intensity per bacterium', fontsize=18, color=color1)
line1 = ax1.plot(GFP_sorted['Volume'], GFP_sorted['signal_MA'], color=color1, linewidth=3, label=f'DAPI ({window}-pt MA)')
ax1.tick_params(axis='both', labelsize=16)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log')
ax1.set_ylim(0,200)
ax2 = ax1.twinx()
color2 = 'green'
ax2.set_ylabel('GFP intensity per bacterium', fontsize=18, color=color2)
line2 = ax2.plot(mCherry_sorted['Volume'], mCherry_sorted['bac_MA'], color=color2, linewidth=3,
                label=f'GFP all cells ({window}-pt MA)', linestyle='dashed')
line3 = ax2.plot(live_mCherry_sorted['Volume'], live_mCherry_sorted['bac_MA'], color=color2, linewidth=3,
                label=f'GFP live cells ({window}-pt MA)')
ax2.tick_params(axis='y', labelcolor=color2, labelsize=16)
ax2.set_ylim(1, 6000)
# ax2.set_yscale('log')
lines = line1 + line2 + line3
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, ncol=3, fontsize=11, frameon=True, columnspacing=1.2, handlelength=2.5)
ax1.grid(alpha=0.3)
plt.show()

# # --------------------------------------------------------------------------------------
# # relative intesity Vs Volume per droplet
# # --------------------------------------------------------------------------------------
#
fig, ax1 = plt.subplots(figsize=(10, 6))
GFP_sorted = t5_per_droplet_filtered.sort_values('Volume').reset_index(drop=True)
GFP_sorted['signal_MA'] = GFP_sorted['Mean_Top25_Intensity'].rolling(window=20, min_periods=20).mean()
mCherry_sorted = t5_per_droplet.sort_values('Volume').reset_index(drop=True)
mCherry_sorted['bac_MA'] = mCherry_sorted['GFP intesity'].rolling(window=100, min_periods=40).mean()
live_mCherry_sorted=t5_per_droplet_filtered.sort_values('Volume').reset_index(drop=True)
live_mCherry_sorted['bac_MA']= live_mCherry_sorted['GFP intesity'].rolling(window=100, min_periods=40).mean()
color1 = 'blue'
ax1.set_xlabel(r'Volume (µm$^3$)', fontsize=18)
ax1.set_ylabel('DAPI intensity per droplet', fontsize=18, color=color1)
line1 = ax1.plot(GFP_sorted['Volume'], GFP_sorted['signal_MA'], color=color1, linewidth=3, label='DAPI (20-pt MA)')
ax1.tick_params(axis='both', labelsize=16)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log')
ax1.set_ylim(0,200)
ax2 = ax1.twinx()
color2 = 'green'
ax2.set_ylabel('GFP intensity per droplet', fontsize=18, color=color2)
line2 = ax2.plot(mCherry_sorted['Volume'], mCherry_sorted['bac_MA'], color=color2, linewidth=3,
                label='GFP all cells (100-pt MA)', linestyle='dashed')
line3 = ax2.plot(live_mCherry_sorted['Volume'], live_mCherry_sorted['bac_MA'], color=color2, linewidth=3,
                label='GFP live cells (100-pt MA)')
ax2.tick_params(axis='y', labelcolor=color2, labelsize=16)
ax2.set_ylim(1, 6000)
# ax2.set_yscale('log')
lines = line1 + line2 + line3
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, ncol=3, fontsize=11, frameon=True, columnspacing=1.2, handlelength=2.5)
ax1.grid(alpha=0.3)
plt.show()
#
#
#
# # # --------------------------------------------------------------------------------------
# # # figure 1 Volume Vs GFP Intensity per bacterium T=5
# # # --------------------------------------------------------------------------------------
rho, p_val = spearmanr(np.log10(t5['Volume']), t5['GFP intesity'])
plt.figure(figsize=(10, 6))
plt.scatter(t5['Volume'], t5['GFP intesity'], s=5, alpha=0.5, c='gray')
window = 3000
t5_sorted = t5.sort_values('Volume').reset_index(drop=True)
t5_sorted['GFP_MA'] = t5_sorted['GFP intesity'].rolling(window=window, min_periods=40).mean()
t5_live_sorted = t5_filtered.sort_values('Volume').reset_index(drop=True)
t5_live_sorted['GFP_MA'] = t5_live_sorted['GFP intesity'].rolling(window=window, min_periods=40).mean()
plt.plot(t5_sorted['Volume'], t5_sorted['GFP_MA'], color='green', linewidth=3,
         label=f'GFP all cells ({window}-pt MA)', linestyle='dashed')
plt.plot(t5_live_sorted['Volume'], t5_live_sorted['GFP_MA'], color='green', linewidth=3,
         label=f'GFP live cells ({window}-pt MA)')
plt.xlabel(r'Volume (µm$^3$)', fontsize=18)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('GFP intensity per bacterium', fontsize=18)
plt.legend(fontsize=12)
plt.tick_params(axis='both', labelsize=16)
plt.grid(alpha=0.3)
plt.show()
#
# # # --------------------------------------------------------------------------------------
# # # figure 2 Volume Vs DAPI Intensity per bacterium T=5
# # # --------------------------------------------------------------------------------------
# #
rho, p_val = spearmanr(np.log10(t5_filtered['Volume']), t5_filtered['Mean_Top25_Intensity'])
plt.figure(figsize = (10,6))
plt.scatter(t5_filtered['Volume'], t5_filtered['Mean_Top25_Intensity'], s=5, alpha=0.3, c='grey')
window = 3000
t5_sorted = t5_filtered.sort_values('Volume').reset_index(drop=True)
t5_sorted['DAPI_MA'] = t5_sorted['Mean_Top25_Intensity'].rolling(window=window, min_periods=10).mean()
plt.plot(t5_sorted['Volume'], t5_sorted['DAPI_MA'], color='blue', linewidth=3,
         label=f'{window}-point MA (Spearman $\\rho$ = {rho:.2f})')
plt.xlabel(r'Volume (µm$^3$)', fontsize=18)
plt.xscale('log')
plt.ylabel('DAPI Intensity per bacterium', fontsize=18)
plt.ylim(0,200)
plt.legend(fontsize=12)
plt.tick_params(axis='both', labelsize=16)
plt.grid(alpha=0.3)
plt.show()
# #
# # # --------------------------------------------------------------------------------------
# # # figure 3 GFP Intensity Vs DAPI Intensity per bacterium T=5
# # # --------------------------------------------------------------------------------------
# plt.figure(figsize = (10,6))
# plt.scatter(t5_subset_GFP['GFP intesity'], t5_subset_GFP['Mean_Top25_Intensity'],s=1, alpha=0.4, c=t5_subset_GFP['log_Volume'].values, cmap='jet')
# plt.xlabel('GFP Intensity')
# plt.xscale('log')
# plt.ylabel('DAPI Intensity')
# plt.yscale('log')
# plt.colorbar(label='log10(Volume)')
# plt.title('DAPI vs GFP Intensity per Bacterium T=5')
# plt.show()
# #
# # # --------------------------------------------------------------------------------------
# # # figure 4 Violin plot GFP intesity per Volume bin per bacterium T=5
# # # -------------------------------------------------------------------------------------
# #
# order = sorted(t5_subset_GFP['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
#
# plt.figure(figsize=(10, 6))
#
# ax = sns.violinplot(
#     data=t5_subset_GFP,
#     x='Bins_vol',
#     y='GFP intesity',
#     inner=None,
#     order=order,
#     color='lightgray'
# )
#
# labels=['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8']
# ax.set_xticklabels(labels)
#
# sns.stripplot(
#     data=t5_subset_GFP,
#     x='Bins_vol',
#     y='GFP intesity',
#     order=order,
#     jitter=True,
#     size=1,
#     color='black',
#     ax=ax,
#     alpha=0.3
#
# )
#
# annotator = Annotator(ax, pairs, data=t5_subset_GFP, x='Bins_vol', y='GFP intesity', order=order)
#
# annotator.configure(
#     test='t-test_ind',        # or 'Mann-Whitney' if non-parametric
#     text_format='star',
#     comparisons_correction='bonferroni',
#     line_offset_to_group=0.05,
#     line_height=0.02,
#     loc='inside',
#     verbose=2
# )
#
# annotator.apply_and_annotate()
#
# plt.xlabel('log(Volume) bins')
# plt.ylabel('Mean GFP Intensity per Bacterium')
# plt.title('Mean GFP Intensity per Bacterium vs. log(Volume) bin T=5')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# #
# # # --------------------------------------------------------------------------------------
# # # figure 5 Violin plot DAPI intesity per Volume bin per bacterium T=5
# # # -------------------------------------------------------------------------------------
# #
# #
# order = sorted(t5_subset_DAPI['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
#
# plt.figure(figsize=(10, 6))
#
# ax = sns.violinplot(
#     data=t5_subset_DAPI,
#     x='Bins_vol',
#     y='Mean_Top25_Intensity',
#     inner=None,
#     order=order,
#     color='lightgray'
# )
#
# labels=['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8']
# ax.set_xticklabels(labels)
#
# sns.stripplot(
#     data=t5_subset_DAPI,
#     x='Bins_vol',
#     y='Mean_Top25_Intensity',
#     order=order,
#     jitter=True,
#     size=1,
#     color='black',
#     ax=ax,
#     alpha=0.3
#
# )
#
# annotator = Annotator(ax, pairs, data=t5_subset_DAPI, x='Bins_vol', y='Mean_Top25_Intensity', order=order)
#
# annotator.configure(
#     test='t-test_ind',        # or 'Mann-Whitney' if non-parametric
#     text_format='star',
#     comparisons_correction='bonferroni',
#     line_offset_to_group=0.05,
#     line_height=0.02,
#     loc='inside',
#     verbose=2
# )
#
# annotator.apply_and_annotate()
#
# plt.xlabel('log(Volume) bins')
# plt.ylabel('Mean DAPI Intensity per Bacterium')
# plt.title('Mean DAPI Intensity per Bacterium vs. log(Volume) bin T=5')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# #
# # # --------------------------------------------------------------------------------------
# # # figure 6 Boxplot plot GFP intesity per Volume bin per bacterium T=5
# # # -------------------------------------------------------------------------------------
#
# order = sorted(t5_subset_GFP['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
# plt.figure(figsize=(10, 6))
#
# # --- Base boxplot ---
# ax = sns.boxplot(
#     data=t5_subset_GFP,
#     x='Bins_vol',
#     y='GFP intesity',
#     order=order,
#     color='lightgray',
#     width=0.6,
#     linewidth=1.2,
#     fliersize=0,  # hide outliers
#     boxprops=dict(edgecolor='black', linewidth=1.2),
#     whiskerprops=dict(color='black', linewidth=1.2),
#     capprops=dict(color='black', linewidth=1.2),
#     medianprops=dict(color='red', linewidth=2)
# )
#
# labels=['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8']
# ax.set_xticklabels(labels)
#
# # --- Overlay data points ---
# sns.stripplot(
#     data=t5_subset_GFP,
#     x='Bins_vol',
#     y='GFP intesity',
#     order=order,
#     jitter=True,
#     size=1,
#     color='black',
#     alpha=0.3,
#     ax=ax
# )
#
# # --- Statistical annotations (neighboring only) ---
# annotator = Annotator(ax, pairs, data=t5_subset_GFP, x='Bins_vol', y='GFP intesity', order=order)
# annotator.configure(
#     test='t-test_ind',        # or 'Mann-Whitney'
#     text_format='star',
#     comparisons_correction='bonferroni',
#     line_offset_to_group=0.05,
#     line_height=0.02,
#     loc='inside',
#     verbose=0
# )
# annotator.apply_and_annotate()
#
# # --- Labels and formatting ---
# plt.title('Mean GFP Intensity per Bacterium vs. log10(Volume) Bin T=5', fontsize=14, weight='bold', pad=15)
# plt.xlabel('log10(Volume) bins', fontsize=12)
# plt.ylabel('Mean GFP Intensity per Bacterium', fontsize=12)
# plt.xticks(rotation=30, fontsize=11)
# plt.yticks(fontsize=11)
# sns.despine(offset=10, trim=True)
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()
#
# # --------------------------------------------------------------------------------------
# # figure 7 Boxplot plot DAPI intesity per Volume bin per bacterium T=5
# # -------------------------------------------------------------------------------------
# #
#
#
order = sorted(t5_filtered['Bins_vol'].unique())
pairs = list(zip(order[:-1], order[1:]))
plt.figure(figsize=(10, 6))

# --- Base boxplot ---
ax = sns.boxplot(
    data=t5_filtered,
    x='Bins_vol',
    y='Mean_Top25_Intensity',
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
    data=t5_filtered,
    x='Bins_vol',
    y='Mean_Top25_Intensity',
    order=order,
    jitter=True,
    size=1,
    color='black',
    alpha=0.3,
    ax=ax
)

# --- Statistical annotations (neighboring only) ---
annotator = Annotator(ax, pairs, data=t5_filtered, x='Bins_vol', y='Mean_Top25_Intensity', order=order)
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
# # --------------------------------------------------------------------------------------
# # figure 8 Volume Vs GFP Intensity per Droplet T=5
# # --------------------------------------------------------------------------------------
# plt.figure(figsize = (10,6))
# plt.scatter(t5_per_droplet_GFP['Volume'], t5_per_droplet_GFP['GFP intesity'],s=5, alpha=0.5)
# window = 100
# t5_sorted = t5_per_droplet_GFP.sort_values('Volume').reset_index(drop=True)
# t5_sorted['GFP_MA'] = t5_sorted['GFP intesity'].rolling(window=window, min_periods=10).mean()
# plt.plot(t5_sorted['Volume'], t5_sorted['GFP_MA'], color='red', linewidth=3, label=f'{window}-point MA')
# plt.xlabel('Volume')
# plt.xscale('log')
# plt.ylabel('GFP Intensity')
# plt.yscale('log')
# plt.title('GFP Intensity vs Volume per Droplet T=5')
# plt.legend(fontsize=12)
# plt.show()
# #
# # # --------------------------------------------------------------------------------------
# # # figure 9 Volume Vs DAPI Intensity per Droplet T=5
# # # --------------------------------------------------------------------------------------
rho, p_val = spearmanr(np.log10(t5_per_droplet_filtered['Volume']), t5_per_droplet_filtered['Mean_Top25_Intensity'])
plt.figure(figsize = (10,6))
plt.scatter(t5_per_droplet_filtered['Volume'], t5_per_droplet_filtered['Mean_Top25_Intensity'], s=5, alpha=0.5, c='grey')
window = 20
t5_sorted = t5_per_droplet_filtered.sort_values('Volume').reset_index(drop=True)
t5_sorted['DAPI_MA'] = t5_sorted['Mean_Top25_Intensity'].rolling(window=window, min_periods=10).mean()
plt.plot(t5_sorted['Volume'], t5_sorted['DAPI_MA'], color='blue', linewidth=3,
         label=f'{window}-point MA (Spearman $\\rho$ = {rho:.2f})')
plt.xlabel(r'Volume (µm$^3$)', fontsize=18)
plt.xscale('log')
plt.ylabel('DAPI Intensity per droplet', fontsize=18)
plt.ylim(0,200)
plt.legend(fontsize=12)
plt.tick_params(axis='both', labelsize=16)
plt.grid(alpha=0.3)
plt.show()
#
# # # --------------------------------------------------------------------------------------
# # # figure 10 Violin plot GFP intesity per Volume bin per Droplet T=5
# # # -------------------------------------------------------------------------------------
# #
# #
# order = sorted(t5_per_droplet_GFP['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
#
# plt.figure(figsize=(10, 6))
#
# ax = sns.violinplot(
#     data=t5_subset_GFP,
#     x='Bins_vol',
#     y='GFP intesity',
#     inner=None,
#     order=order,
#     color='lightgray'
# )
#
# labels=['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8']
# ax.set_xticklabels(labels)
#
# sns.stripplot(
#     data=t5_per_droplet_GFP,
#     x='Bins_vol',
#     y='GFP intesity',
#     order=order,
#     jitter=True,
#     size=5,
#     color='black',
#     ax=ax,
#     alpha=0.3
#
# )
#
# annotator = Annotator(ax, pairs, data=t5_per_droplet_GFP, x='Bins_vol', y='GFP intesity', order=order)
#
# annotator.configure(
#     test='t-test_ind',        # or 'Mann-Whitney' if non-parametric
#     text_format='star',
#     comparisons_correction='bonferroni',
#     line_offset_to_group=0.05,
#     line_height=0.02,
#     loc='inside',
#     verbose=2
# )
#
# annotator.apply_and_annotate()
#
# plt.xlabel('log(Volume) bins')
# plt.ylabel('Mean GFP Intensity per Droplet')
# plt.title('Mean GFP Intensity per Droplet vs. log(Volume) bin T=5')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# #
# # # --------------------------------------------------------------------------------------
# # # figure 11 Violin plot DAPI intesity per Volume bin per Droplet T=5
# # # -------------------------------------------------------------------------------------
# #
# order = sorted(t5_per_droplet_DAPI['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
#
# plt.figure(figsize=(10, 6))
#
# ax = sns.violinplot(
#     data=t5_per_droplet_DAPI,
#     x='Bins_vol',
#     y='Mean_Top25_Intensity',
#     inner=None,
#     order=order,
#     color='lightgray'
# )
#
# labels=['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8']
# ax.set_xticklabels(labels)
#
# sns.stripplot(
#     data=t5_per_droplet_DAPI,
#     x='Bins_vol',
#     y='Mean_Top25_Intensity',
#     order=order,
#     jitter=True,
#     size=5,
#     color='black',
#     ax=ax,
#     alpha=0.3
#
# )
#
# annotator = Annotator(ax, pairs, data=t5_per_droplet_DAPI, x='Bins_vol', y='Mean_Top25_Intensity', order=order)
#
# annotator.configure(
#     test='t-test_ind',        # or 'Mann-Whitney' if non-parametric
#     text_format='star',
#     comparisons_correction='bonferroni',
#     line_offset_to_group=0.05,
#     line_height=0.02,
#     loc='inside',
#     verbose=2
# )
#
# annotator.apply_and_annotate()
#
# plt.xlabel('log(Volume) bins')
# plt.ylabel('Mean DAPI Intensity per Droplet')
# plt.title('Mean DAPI Intensity per Droplet vs. log(Volume) bin T=5')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# #
# # # --------------------------------------------------------------------------------------
# # # figure 12 Boxplot plot GFP intesity per Volume bin per Droplet T=5
# # # -------------------------------------------------------------------------------------
# #
# order = sorted(t5_per_droplet_GFP['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
# plt.figure(figsize=(10, 6))
#
# # --- Base boxplot ---
# ax = sns.boxplot(
#     data=t5_per_droplet_GFP,
#     x='Bins_vol',
#     y='GFP intesity',
#     order=order,
#     color='lightgray',
#     width=0.6,
#     linewidth=1.2,
#     fliersize=0,  # hide outliers
#     boxprops=dict(edgecolor='black', linewidth=1.2),
#     whiskerprops=dict(color='black', linewidth=1.2),
#     capprops=dict(color='black', linewidth=1.2),
#     medianprops=dict(color='red', linewidth=2)
# )
#
# labels=['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8']
# ax.set_xticklabels(labels)
#
# # --- Overlay data points ---
# sns.stripplot(
#     data=t5_per_droplet_GFP,
#     x='Bins_vol',
#     y='GFP intesity',
#     order=order,
#     jitter=True,
#     size=5,
#     color='black',
#     alpha=0.3,
#     ax=ax
# )
#
# # --- Statistical annotations (neighboring only) ---
# annotator = Annotator(ax, pairs, data=t5_per_droplet_GFP, x='Bins_vol', y='GFP intesity', order=order)
# annotator.configure(
#     test='t-test_ind',        # or 'Mann-Whitney'
#     text_format='star',
#     comparisons_correction='bonferroni',
#     line_offset_to_group=0.05,
#     line_height=0.02,
#     loc='inside',
#     verbose=0
# )
# annotator.apply_and_annotate()
#
# # --- Labels and formatting ---
# plt.title('Mean GFP Intensity per Droplet vs. log10(Volume) Bin T=5', fontsize=14, weight='bold', pad=15)
# plt.xlabel('log10(Volume) bins', fontsize=12)
# plt.ylabel('Mean GFP Intensity per Droplet', fontsize=12)
# plt.xticks(rotation=30, fontsize=11)
# plt.yticks(fontsize=11)
# sns.despine(offset=10, trim=True)
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()
#
# #
# # --------------------------------------------------------------------------------------
# # figure 13 Boxplot plot DAPI intesity per Volume bin per Droplet T=5
# # -------------------------------------------------------------------------------------
#
#
order = sorted(t5_per_droplet_filtered['Bins_vol'].unique())
pairs = list(zip(order[:-1], order[1:]))
plt.figure(figsize=(10, 6))

# --- Base boxplot ---
ax = sns.boxplot(
    data=t5_per_droplet_filtered,
    x='Bins_vol',
    y='Mean_Top25_Intensity',
    order=order,
    color='lightgray',
    width=0.6,
    linewidth=1.2,
    fliersize=0,  # hide outliers
    boxprops=dict(edgecolor='black', linewidth=1.2),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
    medianprops=dict(color='red', linewidth=2)
)

labels = [r'$10^3 - 10^4$', r'$10^4 - 10^5$', r'$10^5 - 10^6$', r'$10^6 - 10^7$', r'$10^7 - 10^8$']
ax.set_xticklabels(labels)

# --- Overlay data points ---
sns.stripplot(
    data=t5_per_droplet_filtered,
    x='Bins_vol',
    y='Mean_Top25_Intensity',
    order=order,
    jitter=True,
    size=5,
    color='black',
    alpha=0.3,
    ax=ax
)

# --- Statistical annotations (neighboring only) ---
annotator = Annotator(ax, pairs, data=t5_per_droplet_filtered, x='Bins_vol', y='Mean_Top25_Intensity', order=order)
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
# plt.title('Mean Fluorescence Intensity of Dansyl-Polymyxin B vs. Droplet Volume (T=5)', fontsize=14, weight='bold', pad=15)
plt.xlabel(r'Volume (µm$^3$)', fontsize=18)
plt.ylabel('Mean antibiotic intensity per droplet', fontsize=18)
ax.tick_params(axis='both', labelsize=16)
sns.despine()
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
#
# # --------------------------------------------------------------------------------------
# # figure 14 Volume Vs GFP Intensity per bacterium T=22
# # --------------------------------------------------------------------------------------
# plt.figure(figsize = (10,6))
# plt.scatter(t22_subset_GFP['Volume'], t22_subset_GFP['GFP intesity'],s=5, alpha=0.5)
# window = 1000
# t22_sorted = t22_subset_GFP.sort_values('Volume').reset_index(drop=True)
# t22_sorted['GFP_MA'] = t22_sorted['GFP intesity'].rolling(window=window, min_periods=1).mean()
# plt.plot(t22_sorted['Volume'], t22_sorted['GFP_MA'], color='red', linewidth=3, label=f'{window}-point MA')
# plt.xlabel('Volume')
# plt.xscale('log')
# plt.ylabel('GFP Intensity')
# plt.yscale('log')
# plt.title('GFP Intensity vs Volume per Bacterium T=22')
# plt.legend(fontsize=12)
# plt.show()
# # --------------------------------------------------------------------------------------
# # figure 15 Volume Vs DAPI Intensity per bacterium T=22
# # --------------------------------------------------------------------------------------
#
# plt.figure(figsize = (10,6))
# plt.scatter(t22_subset_DAPI['Volume'], t22_subset_DAPI['Mean_Top25_Intensity'],s=5, alpha=0.5)
# window = 1000
# t22_sorted = t22_subset_DAPI.sort_values('Volume').reset_index(drop=True)
# t22_sorted['DAPI_MA'] = t22_sorted['Mean_Top25_Intensity'].rolling(window=window, min_periods=1).mean()
# plt.plot(t22_sorted['Volume'], t22_sorted['DAPI_MA'], color='red', linewidth=3, label=f'{window}-point MA')
# plt.xlabel('Volume')
# plt.xscale('log')
# plt.ylabel('DAPI Intensity')
# plt.title('DAPI Intensity vs Volume per Bacterium T=22')
# plt.legend(fontsize=12)
# plt.show()
#
# # --------------------------------------------------------------------------------------
# # figure 16 GFP Intensity Vs DAPI Intensity per bacterium T=22
# # --------------------------------------------------------------------------------------
# plt.figure(figsize = (10,6))
# plt.scatter(t22_subset_DAPI['GFP intesity'], t22_subset_DAPI['Mean_Top25_Intensity'],s=1, alpha=0.4, c=t22_subset_DAPI['log_Volume'].values, cmap='jet')
# plt.xlabel('GFP Intensity')
# plt.xscale('log')
# plt.ylabel('DAPI Intensity')
# # plt.yscale('log')
# plt.colorbar(label='log10(Volume)')
# plt.title('DAPI vs GFP Intensity per Bacterium T=22')
# plt.show()
#
# # --------------------------------------------------------------------------------------
# # figure 17 Violin plot GFP intensity per Volume bin per bacterium T=22
# # -------------------------------------------------------------------------------------
#
# order = sorted(t22_subset_GFP['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
#
# plt.figure(figsize=(10, 6))
# ax = sns.violinplot(data=t22_subset_GFP, x='Bins_vol', y='GFP intesity',
#                     inner=None, order=order, color='lightgray')
# ax.set_xticklabels(['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8'])
#
# sns.stripplot(data=t22_subset_GFP, x='Bins_vol', y='GFP intesity',
#               order=order, jitter=True, size=1, color='black', alpha=0.3, ax=ax)
#
# annotator = Annotator(ax, pairs, data=t22_subset_GFP, x='Bins_vol', y='GFP intesity', order=order)
# annotator.configure(test='t-test_ind', text_format='star', comparisons_correction='bonferroni',
#                     line_offset_to_group=0.05, line_height=0.02, loc='inside', verbose=0)
# annotator.apply_and_annotate()
#
# plt.xlabel('log(Volume) bins')
# plt.ylabel('Mean GFP Intensity per Bacterium')
# plt.title('Mean GFP Intensity per Bacterium vs. log(Volume) bin T=22')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # --------------------------------------------------------------------------------------
# # figure 18 Violin plot DAPI intensity per Volume bin per bacterium T=22
# # -------------------------------------------------------------------------------------
#
# order = sorted(t22_subset_DAPI['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
#
# plt.figure(figsize=(10, 6))
# ax = sns.violinplot(data=t22_subset_DAPI, x='Bins_vol', y='Mean_Top25_Intensity',
#                     inner=None, order=order, color='lightgray')
# ax.set_xticklabels(['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8']
# )
#
# sns.stripplot(data=t22_subset_DAPI, x='Bins_vol', y='Mean_Top25_Intensity',
#               order=order, jitter=True, size=1, color='black', alpha=0.3, ax=ax)
#
# annotator = Annotator(ax, pairs, data=t22_subset_DAPI, x='Bins_vol', y='Mean_Top25_Intensity', order=order)
# annotator.configure(test='t-test_ind', text_format='star', comparisons_correction='bonferroni',
#                     line_offset_to_group=0.05, line_height=0.02, loc='inside', verbose=0)
# annotator.apply_and_annotate()
#
# plt.xlabel('log(Volume) bins')
# plt.ylabel('Mean DAPI Intensity per Bacterium')
# plt.title('Mean DAPI Intensity per Bacterium vs. log(Volume) bin T=22')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # --------------------------------------------------------------------------------------
# # figure 19 Boxplot GFP intensity per Volume bin per bacterium T=22
# # -------------------------------------------------------------------------------------
#
# order = sorted(t22_subset_GFP['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
# plt.figure(figsize=(10, 6))
# ax = sns.boxplot(data=t22_subset_GFP, x='Bins_vol', y='GFP intesity', order=order,
#                  color='lightgray', width=0.6, linewidth=1.2, fliersize=0,
#                  boxprops=dict(edgecolor='black', linewidth=1.2),
#                  whiskerprops=dict(color='black', linewidth=1.2),
#                  capprops=dict(color='black', linewidth=1.2),
#                  medianprops=dict(color='red', linewidth=2))
# ax.set_xticklabels(['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8'])
# sns.stripplot(data=t22_subset_GFP, x='Bins_vol', y='GFP intesity', order=order,
#               jitter=True, size=1, color='black', alpha=0.3, ax=ax)
# annotator = Annotator(ax, pairs, data=t22_subset_GFP, x='Bins_vol', y='GFP intesity', order=order)
# annotator.configure(test='t-test_ind', text_format='star', comparisons_correction='bonferroni',
#                     line_offset_to_group=0.05, line_height=0.02, loc='inside', verbose=0)
# annotator.apply_and_annotate()
# plt.title('Mean GFP Intensity per Bacterium vs. log10(Volume) Bin T=22', fontsize=14, weight='bold', pad=15)
# plt.xlabel('log10(Volume) bins', fontsize=12)
# plt.ylabel('Mean GFP Intensity per Bacterium', fontsize=12)
# plt.xticks(rotation=30, fontsize=11)
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# sns.despine(offset=10, trim=True)
# plt.tight_layout()
# plt.show()
#
# # --------------------------------------------------------------------------------------
# # figure 20 Boxplot DAPI intensity per Volume bin per bacterium T=22
# # -------------------------------------------------------------------------------------
#
# order = sorted(t22_subset_DAPI['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
# plt.figure(figsize=(10, 6))
# ax = sns.boxplot(data=t22_subset_DAPI, x='Bins_vol', y='Mean_Top25_Intensity', order=order,
#                  color='lightgray', width=0.6, linewidth=1.2, fliersize=0,
#                  boxprops=dict(edgecolor='black', linewidth=1.2),
#                  whiskerprops=dict(color='black', linewidth=1.2),
#                  capprops=dict(color='black', linewidth=1.2),
#                  medianprops=dict(color='red', linewidth=2))
# ax.set_xticklabels(['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8'])
# sns.stripplot(data=t22_subset_DAPI, x='Bins_vol', y='Mean_Top25_Intensity', order=order,
#               jitter=True, size=1, color='black', alpha=0.3, ax=ax)
# annotator = Annotator(ax, pairs, data=t22_subset_DAPI, x='Bins_vol', y='Mean_Top25_Intensity', order=order)
# annotator.configure(test='t-test_ind', text_format='star', comparisons_correction='bonferroni',
#                     line_offset_to_group=0.05, line_height=0.02, loc='inside', verbose=0)
# annotator.apply_and_annotate()
# plt.title('Mean DAPI Intensity per Bacterium vs. log10(Volume) Bin T=22', fontsize=14, weight='bold', pad=15)
# plt.xlabel('log10(Volume) bins', fontsize=12)
# plt.ylabel('Mean DAPI Intensity per Bacterium', fontsize=12)
# plt.xticks(rotation=30, fontsize=11)
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# sns.despine(offset=10, trim=True)
# plt.tight_layout()
# plt.show()
#
# # --------------------------------------------------------------------------------------
# # figure 21 Volume Vs GFP Intensity per Droplet T=22
# # --------------------------------------------------------------------------------------
# plt.figure(figsize = (10,6))
# plt.scatter(t22_per_droplet_GFP['Volume'], t22_per_droplet_GFP['GFP intesity'],s=5, alpha=0.5)
# window = 100
# t22_sorted = t22_per_droplet_GFP.sort_values('Volume').reset_index(drop=True)
# t22_sorted['GFP_MA'] = t22_sorted['GFP intesity'].rolling(window=window, min_periods=1).mean()
# plt.plot(t22_sorted['Volume'], t22_sorted['GFP_MA'], color='red', linewidth=3, label=f'{window}-point MA')
# plt.xlabel('Volume')
# plt.xscale('log')
# plt.ylabel('GFP Intensity')
# plt.yscale('log')
# plt.title('GFP Intensity vs Volume per Droplet T=22')
# plt.legend(fontsize=12)
# plt.show()
#
# # --------------------------------------------------------------------------------------
# # figure 22 Volume Vs DAPI Intensity per Droplet T=22
# # --------------------------------------------------------------------------------------
# plt.figure(figsize = (10,6))
# plt.scatter(t22_per_droplet_DAPI['Volume'], t22_per_droplet_DAPI['Mean_Top25_Intensity'],s=5, alpha=0.5)
# window = 100
# t22_sorted = t22_per_droplet_DAPI.sort_values('Volume').reset_index(drop=True)
# t22_sorted['DAPI_MA'] = t22_sorted['Mean_Top25_Intensity'].rolling(window=window, min_periods=1).mean()
# plt.plot(t22_sorted['Volume'], t22_sorted['DAPI_MA'], color='red', linewidth=3, label=f'{window}-point MA')
# plt.xlabel('Volume')
# plt.xscale('log')
# plt.ylabel('DAPI Intensity')
# plt.title('DAPI Intensity vs Volume per Droplet T=22')
# plt.legend(fontsize=12)
# plt.show()
#
# # --------------------------------------------------------------------------------------
# # figure 23 Violin plot GFP intesity per Volume bin per Droplet T=22
# # -------------------------------------------------------------------------------------
#
#
# order = sorted(t22_per_droplet_GFP['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
#
# plt.figure(figsize=(10, 6))
#
# ax = sns.violinplot(
#     data=t22_subset_GFP,
#     x='Bins_vol',
#     y='GFP intesity',
#     inner=None,
#     order=order,
#     color='lightgray'
# )
#
# labels=['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8']
#
# ax.set_xticklabels(labels)
#
# sns.stripplot(
#     data=t22_per_droplet_GFP,
#     x='Bins_vol',
#     y='GFP intesity',
#     order=order,
#     jitter=True,
#     size=5,
#     color='black',
#     ax=ax,
#     alpha=0.3
#
# )
#
# annotator = Annotator(ax, pairs, data=t22_per_droplet_GFP, x='Bins_vol', y='GFP intesity', order=order)
#
# annotator.configure(
#     test='t-test_ind',        # or 'Mann-Whitney' if non-parametric
#     text_format='star',
#     comparisons_correction='bonferroni',
#     line_offset_to_group=0.05,
#     line_height=0.02,
#     loc='inside',
#     verbose=2
# )
#
# annotator.apply_and_annotate()
#
# plt.xlabel('log(Volume) bins')
# plt.ylabel('Mean GFP Intensity per Droplet')
# plt.title('Mean GFP Intensity per Droplet vs. log(Volume) bin T=22')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # --------------------------------------------------------------------------------------
# # figure 24 Violin plot DAPI intesity per Volume bin per Droplet T=22
# # -------------------------------------------------------------------------------------
#
# order = sorted(t22_per_droplet_DAPI['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
#
# plt.figure(figsize=(10, 6))
#
# ax = sns.violinplot(
#     data=t22_per_droplet_DAPI,
#     x='Bins_vol',
#     y='Mean_Top25_Intensity',
#     inner=None,
#     order=order,
#     color='lightgray'
# )
#
# labels=['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8']
#
# ax.set_xticklabels(labels)
#
# sns.stripplot(
#     data=t22_per_droplet_DAPI,
#     x='Bins_vol',
#     y='Mean_Top25_Intensity',
#     order=order,
#     jitter=True,
#     size=5,
#     color='black',
#     ax=ax,
#     alpha=0.3
#
# )
#
# annotator = Annotator(ax, pairs, data=t22_per_droplet_DAPI, x='Bins_vol', y='Mean_Top25_Intensity', order=order)
#
# annotator.configure(
#     test='t-test_ind',        # or 'Mann-Whitney' if non-parametric
#     text_format='star',
#     comparisons_correction='bonferroni',
#     line_offset_to_group=0.05,
#     line_height=0.02,
#     loc='inside',
#     verbose=2
# )
#
# annotator.apply_and_annotate()
#
# plt.xlabel('log(Volume) bins')
# plt.ylabel('Mean DAPI Intensity per Droplet')
# plt.title('Mean DAPI Intensity per Droplet vs. log(Volume) bin T=22')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # --------------------------------------------------------------------------------------
# # figure 25 Boxplot plot GFP intesity per Volume bin per Droplet T=22
# # -------------------------------------------------------------------------------------
#
# order = sorted(t22_per_droplet_GFP['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
# plt.figure(figsize=(10, 6))
#
# # --- Base boxplot ---
# ax = sns.boxplot(
#     data=t22_per_droplet_GFP,
#     x='Bins_vol',
#     y='GFP intesity',
#     order=order,
#     color='lightgray',
#     width=0.6,
#     linewidth=1.2,
#     fliersize=0,  # hide outliers
#     boxprops=dict(edgecolor='black', linewidth=1.2),
#     whiskerprops=dict(color='black', linewidth=1.2),
#     capprops=dict(color='black', linewidth=1.2),
#     medianprops=dict(color='red', linewidth=2)
# )
#
# labels=['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8']
# ax.set_xticklabels(labels)
#
# # --- Overlay data points ---
# sns.stripplot(
#     data=t22_per_droplet_GFP,
#     x='Bins_vol',
#     y='GFP intesity',
#     order=order,
#     jitter=True,
#     size=5,
#     color='black',
#     alpha=0.3,
#     ax=ax
# )
#
# # --- Statistical annotations (neighboring only) ---
# annotator = Annotator(ax, pairs, data=t22_per_droplet_GFP, x='Bins_vol', y='GFP intesity', order=order)
# annotator.configure(
#     test='t-test_ind',        # or 'Mann-Whitney'
#     text_format='star',
#     comparisons_correction='bonferroni',
#     line_offset_to_group=0.05,
#     line_height=0.02,
#     loc='inside',
#     verbose=0
# )
# annotator.apply_and_annotate()
#
# # --- Labels and formatting ---
# plt.title('Mean GFP Intensity per Droplet vs. log10(Volume) Bin T=22', fontsize=14, weight='bold', pad=15)
# plt.xlabel('log10(Volume) bins', fontsize=12)
# plt.ylabel('Mean GFP Intensity per Droplet', fontsize=12)
# plt.xticks(rotation=30, fontsize=11)
# plt.yticks(fontsize=11)
# sns.despine(offset=10, trim=True)
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()
#
#
# # --------------------------------------------------------------------------------------
# # figure 26 Boxplot plot DAPI intesity per Volume bin per Droplet T=22
# # -------------------------------------------------------------------------------------
#
# order = sorted(t22_per_droplet_DAPI['Bins_vol'].unique())
# pairs = list(zip(order[:-1], order[1:]))
# plt.figure(figsize=(10, 6))
#
# # --- Base boxplot ---
# ax = sns.boxplot(
#     data=t22_per_droplet_DAPI,
#     x='Bins_vol',
#     y='Mean_Top25_Intensity',
#     order=order,
#     color='lightgray',
#     width=0.6,
#     linewidth=1.2,
#     fliersize=0,  # hide outliers
#     boxprops=dict(edgecolor='black', linewidth=1.2),
#     whiskerprops=dict(color='black', linewidth=1.2),
#     capprops=dict(color='black', linewidth=1.2),
#     medianprops=dict(color='red', linewidth=2)
# )
#
# labels=['3 - 4', '4 - 5', '5 - 6', '6 - 7','7 - 8']
# ax.set_xticklabels(labels)
#
# # --- Overlay data points ---
# sns.stripplot(
#     data=t22_per_droplet_DAPI,
#     x='Bins_vol',
#     y='Mean_Top25_Intensity',
#     order=order,
#     jitter=True,
#     size=5,
#     color='black',
#     alpha=0.3,
#     ax=ax
# )
#
# # --- Statistical annotations (neighboring only) ---
# annotator = Annotator(ax, pairs, data=t22_per_droplet_DAPI, x='Bins_vol', y='Mean_Top25_Intensity', order=order)
# annotator.configure(
#     test='t-test_ind',        # or 'Mann-Whitney'
#     text_format='star',
#     comparisons_correction='bonferroni',
#     line_offset_to_group=0.05,
#     line_height=0.02,
#     loc='inside',
#     verbose=0
# )
# annotator.apply_and_annotate()
#
# # --- Labels and formatting ---
# plt.title('Mean DAPI Intensity per Droplet vs. log10(Volume) Bin T=22', fontsize=14, weight='bold', pad=15)
# plt.xlabel('log10(Volume) bins', fontsize=12)
# plt.ylabel('Mean DAPI Intensity per Droplet', fontsize=12)
# plt.xticks(rotation=30, fontsize=11)
# plt.yticks(fontsize=11)
# sns.despine(offset=10, trim=True)
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()















