import pandas as pd
import numpy as np
from scipy.integrate import odeint
from multiprocessing import Pool,cpu_count
from functools import partial
from numba import njit
from tqdm import tqdm
import warnings
import os
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import griddata

warnings.filterwarnings("ignore")

@njit
def monod_coupled(y, t, mu_max, Ks, Y, V):
    B_live, S = y
    density = B_live / V
    mu = mu_max * (S / (Ks + S))
    dB_live_dt = mu * B_live
    dS_dt = - (1.0 / Y) * mu * density
    return np.array([dB_live_dt, dS_dt])

def solve_monod_control(t, mu_max, Ks, B0, S0, Y, V):
    y0 = [B0, S0]
    sol = odeint(monod_coupled, y0, t, args=(mu_max, Ks, Y, V))
    return sol

@njit
def log_and_RMSE(B_pred, B_data):
    den = np.maximum(B_data, 1.0) # Avoid division by zero
    RMSE = np.sqrt(np.mean(((B_pred - B_data) / den) ** 2)) #Calculate RMSE
    return RMSE

@njit
def RMSE_calc(B_pred, B_data):
    return np.sqrt(np.mean(((B_pred - B_data)) ** 2))

def fit_droplet_full(args):
    (mu_max, Ks, S0,Y, t_data, B_data, V) = args
    B0 = B_data[0]
    sol = solve_monod_control(t_data, mu_max, Ks, B0=B0, S0=S0, Y=Y, V=V)
    B_pred = sol[:, 0]
    bin_V = np.floor(np.log10(V))
    average_count = np.mean(B_data)
    # if np.any(np.isnan(B_pred)) or np.any(np.isinf(B_pred)) or np.any(B_pred < 0):
    if np.any(np.isnan(B_pred)) or np.any(np.isinf(B_pred)):
        return (float('inf'), bin_V, average_count,0)
    RMSE = RMSE_calc(B_pred, B_data)
    return (RMSE,bin_V, average_count,0)

def total_mean_grouped(results):
    results_df = pd.DataFrame(results, columns=['RMSE', 'bin_V', 'average_count', 'A_free0'])
    grouped = results_df.groupby(['bin_V','A_free0']).agg({'RMSE': 'mean', 'average_count': 'mean'}).reset_index()
    grouped['normalized RMSE'] = grouped['RMSE'] / grouped['average_count']
    print(grouped)
    mean_grouped =  grouped['normalized RMSE'].mean()
    return mean_grouped

def objective_function(x, droplets_data, pool):
    mu_max = x[0]
    Ks = x[1]
    S0 = 1
    args_list = [
        (
            mu_max, Ks, S0,
            droplet['Y param'].to_numpy()[0],
            droplet['time'].to_numpy(),
            droplet['Count'].to_numpy(),
            droplet['Volume'].to_numpy()[0],
        )
        for droplet in droplets_data
    ]
    results_iter = pool.imap_unordered(fit_droplet_full, args_list,chunksize=250)
    results = list(results_iter)
    mean_grouped = total_mean_grouped(results)
    return mean_grouped



@njit
def get_neighbor(x, step_sizes, bounds_min, bounds_max):
    x = np.asarray(x)
    step_sizes = np.asarray(step_sizes)
    bounds_min = np.asarray(bounds_min)
    bounds_max = np.asarray(bounds_max)
    neighbor = np.empty_like(x)
    for i in range(x.shape[0]):
        step = np.random.uniform(-step_sizes[i], step_sizes[i])
        neighbor[i] = x[i] + step
    if bounds_min is not None and bounds_max is not None:
        neighbor = np.minimum(bounds_max, np.maximum(bounds_min, neighbor))
    return neighbor



def simulated_annealing(objective, bounds_min,bounds_max, n_iterations, step_sizes, temp, droplets_data, pool):
    csv_file = 'objective_history.csv'
    if os.path.exists(csv_file):
        os.remove(csv_file)
    best = np.array([np.random.uniform(low, high) for low, high in zip(bounds_min, bounds_max)])
    best_eval = objective(best, droplets_data, pool)*100
    current, current_eval = best.copy(), best_eval
    scores = [best_eval]
    best_mu_max_history = [best[0]]
    best_Ks_history = [best[1]]
    initial_step_sizes = list(step_sizes)
    current_temp = temp
    pbar = tqdm(range(n_iterations), desc='Simulated Annealing Progress')
    for i in pbar:
        current_temp = current_temp * 0.99
        step_sizes = [s * (current_temp / temp) for s in initial_step_sizes]
        candidate = get_neighbor(
            np.array(current, dtype=np.float64),
            np.array(step_sizes, dtype=np.float64),
            bounds_min=np.array(bounds_min, dtype=np.float64),
            bounds_max=np.array(bounds_max, dtype=np.float64)
        )
        candidate_eval = objective(candidate, droplets_data, pool)*100
        improved = False
        delta = current_eval - candidate_eval
        accept_prob = 1 / (1 + np.exp(-delta / current_temp))
        data= {
            'Iteration': i,
            'Current Temp': current_temp,
            'Step Sizes': step_sizes,
            'Candidate mu_max': candidate[0],
            'Candidate Ks': candidate[1],
            'Candidate Eval': candidate_eval,
            'Current mu_max': current[0],
            'Current Ks': current[1],
            'Current Eval': current_eval,
            'Best': best,
            'Best Eval': best_eval
        }
        data= pd.DataFrame([data])
        data.to_csv('objective_history.csv', mode='a', header=not i, index=False)
        if candidate_eval < current_eval or np.random.rand() < accept_prob:
            current, current_eval = candidate, candidate_eval
            if candidate_eval < best_eval:
                best, best_eval = candidate, candidate_eval
                improved = True
        scores.append(best_eval)
        best_mu_max_history.append(best[0])
        best_Ks_history.append(best[1])
        # pbar.set_postfix({
        #     'Best mu max': f'{best[0]:.5f}',
        #     'Best Ks': f'{best[1]:.5f}',
        #     'Best RMSE': f'{best_eval:.5f}'
        # })
        print(f"Iteration {i}:")
        print(f"  current mu_max: {current[0]:.2f}")
        print(f"  candidate mu_max: {candidate[0]:.2f}")
        print(f"  current Ks: {current[1]:.2f}")
        print(f"  candidate Ks: {candidate[1]:.2f}")
        print(f"  current RMSE: {current_eval:.2f}")
        print(f"  candidate RMSE: {candidate_eval:.2f}")
        print(f"  acceptance prob: {accept_prob:.2f}")
    return best, best_eval, scores, best_mu_max_history, best_Ks_history


def simulate_droplet(droplet, mu_max, Ks, S0):
    t_data = droplet['time'].to_numpy()
    B_data = droplet['Count'].to_numpy()
    V = droplet['Volume'].iloc[0]
    Y = droplet['Y param'].iloc[0]
    B0 = B_data[0]
    sol = solve_monod_control(t_data, mu_max, Ks, B0=B0, S0=S0, Y=Y, V=V)
    B_pred = sol[:, 0]
    S= sol[:, 1]
    growth_rate = mu_max * S / (Ks + S)
    RMSE = log_and_RMSE(B_pred, B_data)
    return [
        {
            'DW': droplet['DW'].iloc[0],
            'time': t,
            'Volume': V,
            'B0': B0,
            'Count': B_data[i],
            'fitted': B_pred[i],
            'S': S[i],
            'growth_rate': growth_rate[i],
            'free_A': 0,
            'mu_max': mu_max,
            'Y': Y,
            'Ks': Ks,
            'S0': S0,
            'RMSE': RMSE
        }
        for i, t in enumerate(t_data)
    ]

def calculate_yield_vs_volume(df, well_names, window_size=100):
    sub_df = df[df['Well'].isin(well_names)].copy()
    T_0 = sub_df[sub_df['time'] == 0]
    T_0_filtered = T_0[T_0['Count'] != 0]
    valid_dws = T_0_filtered['DW'].unique()
    sub_df = sub_df[sub_df['DW'].isin(valid_dws)]
    droplets_data = [droplet for _, droplet in sub_df.groupby('DW') if not (droplet['Count'] == 0).any()]
    data = {'DW': [], 'Y': [], 'Volume': [],'density': []}
    for droplet in droplets_data:
        T_0_Count = droplet[droplet['time'] == 0]['Count'].to_numpy()[0]
        T_24_Count = droplet['Count'].iloc[-4:].mean()
        V = droplet['Volume'].iloc[0]
        delta = T_24_Count - T_0_Count
        Y = delta / V
        initial_density= T_0_Count / V
        data['density'].append(initial_density)
        data['DW'].append(droplet['DW'].iloc[0])
        data['Y'].append(Y)
        data['Volume'].append(V)
    result_df = pd.DataFrame(data)
    sorted_indices = np.argsort(result_df['Volume'])
    Vs_sorted = result_df['Volume'].to_numpy()[sorted_indices]
    Ys_sorted = result_df['Y'].to_numpy()[sorted_indices]
    Ys_ma = pd.Series(Ys_sorted).rolling(window=window_size, min_periods=1, center=True).mean()
    MA_data = pd.DataFrame({'Volume': Vs_sorted, 'Ys_ma': Ys_ma})
    df['Y param'] = np.interp(df['Volume'], MA_data['Volume'], MA_data['Ys_ma'])
    return df

def filter_and_preprocess_df(df, well_names=['C1', 'C5']):
    df['log_Volume'] = np.log10(df['Volume'])
    df['bin'] = df['log_Volume'].apply(lambda x: np.floor(x))
    treatment_mapping = {'C1': 0, 'C2': 30, 'C3': 10, 'C4': 3.3, 'C5': 0, 'C6': 3.3, 'C7': 30, 'C8': 10}
    df['treatment'] = df['Well'].map(treatment_mapping)
    df['Count'] = df['Count'].replace(-1, 0)
    T_0 = df[df['time'] == 0]
    T_0_filtered = T_0[T_0['Count'] != 0]
    valid_dws = T_0_filtered['DW'].unique()
    df = df[df['DW'].isin(valid_dws)]
    df = calculate_yield_vs_volume(df, well_names)
    return df

def run_single_annealing(args):
    (bounds_min, bounds_max, n_iterations, step_sizes, temp, droplets_data, pool) = args
    best, best_eval, scores, best_mu_max_history, best_Ks_history = simulated_annealing(
        objective_function,
        bounds_min,
        bounds_max,
        n_iterations,
        step_sizes,
        temp,
        droplets_data,
        pool,
    )
    return best, best_eval

if __name__== "__main__":
    df = pd.read_csv('L:/21012025_BSF obj x10/final_data.csv')
    df = filter_and_preprocess_df(df, ['C1', 'C5'])
    df= df[df['treatment'] == 0]
    # df=df[df['bin']>=7]
    # big_drops= df[(df['log_Volume'] >= 6) & (df['log_Volume'] <= 7)]
    # first_5_DWs = big_drops['DW'].unique()[:50]
    # big_drops = big_drops[big_drops['DW'].isin(first_5_DWs)]
    droplets_data = [droplet for _, droplet in df.groupby('DW') if not (droplet['Count'] == 0).any()]
    bounds_min = [0, 1]
    bounds_max = [1.5,3]
    n_iterations = 1000
    step_sizes = (0.25, 1)
    temp = 1
    with Pool(processes=cpu_count()) as pool:
        best, best_eval = run_single_annealing(
            (bounds_min, bounds_max, n_iterations, step_sizes, temp, droplets_data, pool))
    sim_results = []
    mu_max = best[0]
    Ks = best[1]
    S0 = 1
    print (f"Best mu_max: {mu_max}, Best Ks: {Ks}, Best RMSE: {best_eval}")
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(partial(simulate_droplet, mu_max=mu_max, Ks=Ks, S0=S0), droplets_data)
    sim_results = [item for sublist in results for item in sublist]
    sim_df = pd.DataFrame(sim_results)
    sim_df.to_csv('control_simulate_annealing_optimization.csv', index=False)

    # for DW in sim_df['DW'].unique():
    #     droplet_data = sim_df[sim_df['DW'] == DW]
    #     plt.plot(droplet_data['time'], droplet_data['Count'])
    #     plt.plot(droplet_data['time'], droplet_data['fitted'])
    #     plt.xlabel('Time')
    #     plt.ylabel('Count')
    #     plt.title(f'Droplet {DW} Count over Time')
    #     plt.legend(['Observed', 'Fitted'])
    #     plt.show()

    RMSE_over_time=pd.read_csv('objective_history.csv')
    plt.figure()
    plt.plot(RMSE_over_time['Iteration'], RMSE_over_time['Current Eval'], label='Current RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('Current Eval')
    plt.title('Best RMSE over Iterations')
    plt.legend()
    plt.show()

    # fig, ax = plt.subplots(figsize=(15, 15))
    # volumes = sim_df.groupby('DW')['Volume'].first()
    # log_volumes = np.log10(volumes)
    # norm = mcolors.Normalize(vmin=log_volumes.min(), vmax=log_volumes.max())
    # cmap = plt.get_cmap('jet')
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # for DW in sim_df['DW'].unique():
    #     droplet_data = sim_df[sim_df['DW'] == DW]
    #     volume = droplet_data['Volume'].iloc[0]
    #     log_volume = np.log10(volume)
    #     color = cmap(norm(log_volume))
    #     ax.plot(droplet_data['time'], droplet_data['S'], color=color)
    # ax.set_xlabel('Time')
    # ax.set_ylabel('S')
    # ax.set_title('S over time (colored by log10 droplet volume)')
    # cbar = fig.colorbar(sm, ax=ax, label='log10(Droplet Volume)')
    # plt.show()
    # fig2, ax2 = plt.subplots(figsize=(15, 15))
    # for DW in sim_df['DW'].unique():
    #     droplet_data = sim_df[sim_df['DW'] == DW]
    #     volume = droplet_data['Volume'].iloc[0]
    #     log_volume = np.log10(volume)
    #     color = cmap(norm(log_volume))
    #     ax2.plot(droplet_data['time'], droplet_data['growth_rate'], color=color)
    # ax2.set_xlabel('Time')
    # ax2.set_ylabel('Growth Rate')
    # ax2.set_title('Growth Rate over time (colored by log10 droplet volume)')
    # cbar2 = fig2.colorbar(sm, ax=ax2, label='log10(Droplet Volume)')
    # plt.show()
