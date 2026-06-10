# preprocess.py
import pickle
import numpy as np
import os
import torch
from tqdm import tqdm
from typing import Tuple

DATASET_CONFIGS = {
    'Earthquake': {
        'TEMP_WINDOWS': [64, 32, 16, 8],
        'SPAT_WINDOWS': [64, 32, 16, 8],
        'GRID_SIZES': [0.5, 0.2, 0.1, 0.05],
        'TIME_SCALES_GAT': (0, 0, 0),
        'K_NEIGHBORS_GAT': (0, 0, 0),
    },
    'COVID19': {
        'TEMP_WINDOWS': [14, 7],
        'SPAT_WINDOWS': [14, 7],
        'GRID_SIZES':  [0.1, 0.05],
        'TIME_SCALES_GAT': (0, 0, 0),
        'K_NEIGHBORS_GAT': (0, 0, 0),
    },
    'Citibike': {
        'TEMP_WINDOWS': [32, 16, 8, 4], 
        'SPAT_WINDOWS': [8, 6, 4, 2],
        'GRID_SIZES': [0.02, 0.01, 0.005, 0.002],
        'TIME_SCALES_GAT': (0, 0, 0),
        'K_NEIGHBORS_GAT': (0, 0, 0),
        
    },
    'Crime': {
        'TEMP_WINDOWS': [32, 16, 8, 4],
        'SPAT_WINDOWS': [32, 16, 8, 4],
        'GRID_SIZES': [0.1, 0.05, 0.01, 0.005], 
        'TIME_SCALES_GAT': (15, 7, 3),
        'K_NEIGHBORS_GAT': (0, 0, 0),
    }
}

DEFAULT_CONFIG = {
    'TEMP_WINDOWS': [64, 32, 8, 2],
    'SPAT_WINDOWS': [64, 32, 8, 2],
    'GRID_SIZES': [0.2, 0.1, 0.02, 0.005],
    'TIME_SCALES_GAT': (50, 20, 5),
    'K_NEIGHBORS_GAT': (20, 10, 5),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_dataset_config(dataset_name):
    """Return dataset-specific preprocessing settings."""
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        print(f"Using config for {dataset_name}")
    else:
        config = DEFAULT_CONFIG
        print(f"Dataset '{dataset_name}' not found, using DEFAULT config")
    return config

def compute_normalization_stats(train_data, loc_dim=2):
    """
    Compute normalization statistics from the training split only.
    """
    print("  Computing normalization statistics from training data only...")
    all_abs_time = []
    all_time_diff = []
    all_loc = []
    all_loc_diff = []
    
    for seq in train_data:
        if len(seq) < 2:
            continue
        seq_arr = np.array(seq)
        all_abs_time.append(seq_arr[: , 0])
        all_time_diff.append(seq_arr[:, 1])
        all_loc.append(seq_arr[:, 2:2+loc_dim])
        loc_diff = seq_arr[1:, 2:2+loc_dim] - seq_arr[:-1, 2:2+loc_dim]
        all_loc_diff.append(loc_diff)
    

    all_abs_time = np.concatenate(all_abs_time)
    all_time_diff = np.concatenate(all_time_diff)
    all_loc = np.concatenate(all_loc)
    all_loc_diff = np.concatenate(all_loc_diff)
    
    stats = {

        'abs_time_min':  float(all_abs_time.min()),
        'abs_time_max': float(all_abs_time.max()),
        

        'time_diff_mean': float(all_time_diff.mean()),
        'time_diff_std':  float(all_time_diff.std()),

        'time_diff_min':  float(all_time_diff.min()),
        'time_diff_max': float(all_time_diff.max()),
        

        'loc_min': all_loc.min(axis=0).tolist(),
        'loc_max': all_loc.max(axis=0).tolist(),
        

        'loc_diff_mean':  all_loc_diff.mean(axis=0).tolist(),
        'loc_diff_std': all_loc_diff.std(axis=0).tolist(),
    }
    
    print(f"    Abs time range: [{stats['abs_time_min']:.2f}, {stats['abs_time_max']:.2f}]")
    print(f"    Time diff:  mean={stats['time_diff_mean']:.4f}, std={stats['time_diff_std']:.4f}")
    print(f"    Loc range: {stats['loc_min']} to {stats['loc_max']}")
    print(f"    Loc diff: mean={stats['loc_diff_mean']}, std={stats['loc_diff_std']}")
    
    return stats

def _build_temporal_multiscale(seq: torch.Tensor, windows):
    """Build compact temporal multi-scale features."""
    seq = seq.squeeze(-1).to(device)
    L = seq.shape[0]
    all_window_feats = []
    

    windows_to_use = windows

    for w in windows_to_use:
        padded_seq = torch.nn.functional.pad(seq, (w, 0), value=0.0)
        windows_view = padded_seq.unfold(dimension=0, size=w, step=1)[:L]
        
        valid_counts = torch.arange(1, L + 1, device=device).float().clamp(max=w)
        
        positions = torch.arange(w, device=device).unsqueeze(0)
        start_positions = (w - valid_counts).unsqueeze(-1)
        mask = positions >= start_positions
        
        masked_view = torch.where(mask, windows_view, torch.tensor(0.0, device=device))
        


        mean_val = masked_view.sum(dim=-1) / valid_counts
        

        mean_expanded = mean_val.unsqueeze(-1)
        sq_diff = torch.where(mask, (windows_view - mean_expanded) ** 2, torch.tensor(0.0, device=device))
        variance = sq_diff.sum(dim=-1) / valid_counts.clamp(min=1)
        std_val = torch.sqrt(variance)
        std_val = torch.where(valid_counts > 1, std_val, torch.zeros_like(std_val))
        

        first_valid_idx = (w - valid_counts).long().clamp(min=0)
        first_val = windows_view.gather(1, first_valid_idx.unsqueeze(-1)).squeeze(-1)
        last_val = windows_view[:, -1]
        trend = (last_val - first_val) / valid_counts.clamp(min=1)
        trend = torch.where(valid_counts > 1, trend, torch.zeros_like(trend))
        

        feat = torch.stack([mean_val, std_val, trend], dim=-1)
        all_window_feats.append(feat)
    
    if not all_window_feats:
        return torch.zeros((0, len(windows_to_use) * 3), device=device)
    
    return torch.cat(all_window_feats, dim=-1)[1:]



def _build_spatial_multiscale(seq: torch.Tensor, windows, grid_sizes):
    """Build compact spatial multi-scale features."""
    seq = seq.to(device)
    L, D = seq.shape
    all_feats = []
    

    windows_to_use = windows
    grids_to_use = grid_sizes

    for w, g in zip(windows_to_use, grids_to_use):
        padded_seq = torch.nn.functional.pad(seq, (0, 0, w, 0), value=0.0)
        windows_view = padded_seq.unfold(dimension=0, size=w, step=1)
        windows_view = windows_view.permute(0, 2, 1)[:L]
        
        valid_counts = torch.arange(1, L + 1, device=device).float().clamp(max=w)
        
        positions = torch.arange(w, device=device).unsqueeze(0)
        start_positions = (w - valid_counts).unsqueeze(-1)
        mask = positions >= start_positions
        mask_3d = mask.unsqueeze(-1).expand(-1, -1, D)
        
        if g > 0:
            snapped = torch.round(windows_view / g) * g
        else:
            snapped = windows_view
        
        masked_snapped = torch.where(mask_3d, snapped, torch.tensor(0.0, device=device))
        


        mean = masked_snapped.sum(dim=1) / valid_counts.unsqueeze(-1)
        

        mean_expanded = mean.unsqueeze(1)
        sq_diff = torch.where(mask_3d, (snapped - mean_expanded) ** 2, torch.tensor(0.0, device=device))
        variance = sq_diff.sum(dim=1) / valid_counts.unsqueeze(-1).clamp(min=1)
        std = torch.sqrt(variance)
        std = torch.where(valid_counts.unsqueeze(-1) > 1, std, torch.zeros_like(std))
        

        first_valid_idx = (w - valid_counts).long().clamp(min=0)
        first_val = windows_view.gather(1, first_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, D)).squeeze(1)
        last_val = windows_view[:, -1, :]
        displacement = (last_val - first_val) / valid_counts.unsqueeze(-1).clamp(min=1)
        displacement = torch.where(valid_counts.unsqueeze(-1) > 1, displacement, torch.zeros_like(displacement))
        

        feat = torch.cat([mean, std, displacement], dim=-1)
        all_feats.append(feat)
    
    if not all_feats:
        return torch.zeros((0, len(windows_to_use) * D * 3), device=device)
    
    return torch.cat(all_feats, dim=-1)[1:]


def _build_multiscale_edges(loc:  torch.Tensor, times: torch.Tensor,
                            time_scales: Tuple, k_neighbors:  Tuple):
    """Build graph edges with vectorized index-scale neighborhoods."""
    num_nodes = loc.size(0)
    if num_nodes <= 1:
        return None, None
    
    loc = loc.to(device)
    times = times.to(device)
    
    space_dist_mat = torch.cdist(loc, loc, p=2.0)
    time_span = torch.clamp(times.max() - times.min(), min=1e-6)
    space_span = torch.clamp(space_dist_mat.max(), min=1e-6)
    
    edge_src_list, edge_dst_list, edge_attr_list = [], [], []
    

    base_src = torch.arange(0, num_nodes - 1, device=device, dtype=torch.long)
    base_dst = base_src + 1
    
    d_t = torch.abs(times[base_dst] - times[base_src]) / time_span
    d_s = space_dist_mat[base_dst, base_src] / space_span
    time_decay = torch.sigmoid(-d_t * 5.0)
    spatial_decay = torch.exp(-space_dist_mat[base_dst, base_src]) * 1.2
    
    edge_src_list.append(base_src)
    edge_dst_list.append(base_dst)
    edge_attr_list.append(torch.stack([
        d_t * time_decay,
        d_s * spatial_decay,
        torch.zeros_like(d_t)
    ], dim=-1))
    

    row_idx = torch.arange(num_nodes, device=device).unsqueeze(1)
    col_idx = torch.arange(num_nodes, device=device).unsqueeze(0)
    index_diff = row_idx - col_idx
    
    for scale_idx, (scale, k_base) in enumerate(zip(time_scales, k_neighbors)):
        valid_mask = (index_diff > 0) & (index_diff <= scale)
        
        masked_dist = space_dist_mat.clone()
        masked_dist[~valid_mask] = float('inf')
        
        k = min(k_base, num_nodes - 1)
        if k <= 0:
            continue
        
        dists, indices = torch.topk(masked_dist, k=k, dim=1, largest=False)
        valid_neighbor_mask = dists != float('inf')
        
        if not valid_neighbor_mask.any():
            continue
        
        src = indices[valid_neighbor_mask]
        row_indices = torch.arange(num_nodes, device=device).unsqueeze(1).expand(-1, k)
        dst = row_indices[valid_neighbor_mask]
        
        d_t = torch.abs(times[dst] - times[src]) / time_span
        d_s = dists[valid_neighbor_mask] / space_span
        time_decay = torch.sigmoid(-d_t * 5.0)
        spatial_decay = torch.exp(-dists[valid_neighbor_mask]) * 1.2
        
        scale_weight = torch.full_like(d_t, float((scale_idx + 1) / len(time_scales)))
        
        edge_src_list.append(src)
        edge_dst_list.append(dst)
        edge_attr_list.append(torch.stack([
            d_t * time_decay,
            d_s * spatial_decay,
            scale_weight
        ], dim=-1))
    
    if not edge_src_list:
        return None, None
    
    edge_index = torch.stack([torch.cat(edge_src_list), torch.cat(edge_dst_list)], dim=0)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    
    return edge_index, edge_attr

def process_and_save(data, data_type, stats, config, loc_dim=2):
    """
    Process one split using training-set normalization statistics.
    """
    TEMP_WINDOWS = config['TEMP_WINDOWS']
    SPAT_WINDOWS = config['SPAT_WINDOWS']
    GRID_SIZES = config['GRID_SIZES']
    TIME_SCALES_GAT = config['TIME_SCALES_GAT']
    K_NEIGHBORS_GAT = config['K_NEIGHBORS_GAT']
    

    abs_time_min = stats['abs_time_min']
    abs_time_max = stats['abs_time_max']
    time_diff_mean = stats['time_diff_mean']
    time_diff_std = stats['time_diff_std']
    time_diff_min = stats['time_diff_min']
    time_diff_max = stats['time_diff_max']
    loc_min = np.array(stats['loc_min'])
    loc_max = np.array(stats['loc_max'])
    loc_diff_mean = np.array(stats['loc_diff_mean'])
    loc_diff_std = np.array(stats['loc_diff_std'])
    
    output_data = []
    skipped = 0
    
    for seq in tqdm(data, desc=f"Processing {data_type}"):
        if len(seq) < 2:
            skipped += 1
            continue

        seq_tensor = torch.tensor(seq, dtype=torch.float32, device=device)
        seq_len = len(seq)
        time_abs = seq_tensor[: , 0]
        time_abs_norm = (time_abs - abs_time_min) / (abs_time_max - abs_time_min + 1e-8)
        

        time_diff = seq_tensor[:, 1]
        time_diff_zscore = (time_diff - time_diff_mean) / (time_diff_std + 1e-8)
        time_diff_zscore = time_diff_zscore.clamp(-5, 5) 

        time_diff_minmax = (time_diff - time_diff_min) / (time_diff_max - time_diff_min + 1e-8)
        

        loc_raw = seq_tensor[: , 2:2+loc_dim]
        loc_min_t = torch.tensor(loc_min, dtype=torch.float32, device=device)
        loc_max_t = torch.tensor(loc_max, dtype=torch.float32, device=device)
        loc_norm = (loc_raw - loc_min_t) / (loc_max_t - loc_min_t + 1e-8)
        

        loc_diff_raw = torch.zeros_like(loc_raw)
        loc_diff_raw[1:] = loc_raw[1:] - loc_raw[:-1]
        
        loc_diff_mean_t = torch.tensor(loc_diff_mean, dtype=torch.float32, device=device)
        loc_diff_std_t = torch.tensor(loc_diff_std, dtype=torch.float32, device=device)
        loc_diff_zscore = (loc_diff_raw - loc_diff_mean_t) / (loc_diff_std_t + 1e-8)
        loc_diff_zscore[0] = 0
        loc_diff_zscore = loc_diff_zscore.clamp(-5, 5) 

        

        temporal_feats = _build_temporal_multiscale(time_diff_zscore.unsqueeze(-1), TEMP_WINDOWS)
        

        spatial_feats = _build_spatial_multiscale(loc_norm, SPAT_WINDOWS, GRID_SIZES)
       

        history_loc_norm = loc_norm[:-1]
        history_time_abs_norm = time_abs_norm[:-1]
        
        edge_index, edge_attr = _build_multiscale_edges(
            history_loc_norm, history_time_abs_norm,
            TIME_SCALES_GAT, K_NEIGHBORS_GAT
        )

        if edge_index is None:
            skipped += 1
            continue
        
        if temporal_feats.shape[0] != seq_len - 1:
            skipped += 1
            continue
        

        t_base = time_abs_norm[: seq_len - 1].unsqueeze(-1)
        l_base = loc_norm[:seq_len - 1]
        ld_base = loc_diff_zscore[:seq_len - 1]
        td_base = time_diff_zscore[:seq_len - 1].unsqueeze(-1)
        x_combined = torch.cat([
            t_base,           # [L-1, 1]
            td_base, 
            l_base,           # [L-1, dim]
            ld_base,          # [L-1, dim]
            temporal_feats,   # [L-1, num_temp_feats]
            spatial_feats     # [L-1, num_spat_feats]
        ], dim=-1)
        

        data_entry = {
            'x': x_combined.cpu(),
            'target_time':  time_diff_minmax[1:].cpu(),
            'target_loc': loc_norm[1:].cpu(),
            'edge_index': edge_index.cpu(),
            'edge_attr': edge_attr.cpu(),
            'length': seq_len - 1
        }
        output_data.append(data_entry)
        

        del seq_tensor, time_abs_norm, time_diff_zscore, loc_norm, loc_diff_zscore
        del temporal_feats, spatial_feats, edge_index, edge_attr, x_combined
    print(f"  {data_type}:  {len(output_data)} valid, {skipped} skipped")
    
    return output_data


def add_time_interval(data):
    """Insert inter-event time intervals into each event sequence."""
    processed = []
    for seq in tqdm(data, desc="Adding intervals"):
        seq = [list(item) for item in seq]
        new_seq = []
        for i, event in enumerate(seq):
            if i > 0:
                raw_diff = event[0] - seq[i-1][0]
            else:
                raw_diff = 0.0
            time_diff = max(raw_diff, 1e-5)
            new_seq.append([event[0], time_diff] + event[1:])
        processed.append(new_seq)
    return processed


def process_dataset(dataset_name, dim):
    """Preprocess one dataset."""
    print(f"\n{'='*60}")
    print(f"Starting preprocessing for dataset: {dataset_name}")
    print(f"{'='*60}")
    
    config = get_dataset_config(dataset_name)
    base_path = f'dataset/{dataset_name}/'
    
    if not os.path.exists(base_path):
        print(f"Error: Path '{base_path}' does not exist!")
        return
    

    print("\n1/5:  Loading raw data...")
    with open(os.path.join(base_path, 'data_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(base_path, 'data_val.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(base_path, 'data_test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"  Train:  {len(train_data)} sequences")
    print(f"  Val: {len(val_data)} sequences")
    print(f"  Test: {len(test_data)} sequences")


    print("\n2/5:  Calculating time intervals...")
    train_data = add_time_interval(train_data)
    val_data = add_time_interval(val_data)
    test_data = add_time_interval(test_data)


    print("\n3/5: Computing normalization stats (TRAIN ONLY)...")
    stats = compute_normalization_stats(train_data, loc_dim=dim)
    stats['config'] = config
    

    print("\n4/5: Processing all datasets...")
    train_output = process_and_save(train_data, 'train', stats, config, loc_dim=dim)
    val_output = process_and_save(val_data, 'val', stats, config, loc_dim=dim)
    test_output = process_and_save(test_data, 'test', stats, config, loc_dim=dim)
    

    print("\n5/5: Saving processed data...")
    
    with open(os.path.join(base_path, 'stats1.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    for data, name in [(train_output, 'train'), (val_output, 'val'), (test_output, 'test')]:
        with open(os.path.join(base_path, f'data_{name}_processed1.pkl'), 'wb') as f:
            pickle.dump(data, f)
    
    print(f"\n--- Preprocessing for {dataset_name} complete!  ---")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess dataset for STPP')
    parser.add_argument('--dataset', type=str, default='Earthquake',
                        choices=['Earthquake', 'COVID19', 'Citibike', 'Crime'],
                        help='Dataset to preprocess')
    parser.add_argument('--dim', type=int, default=2, help='Spatial dimension')
    parser.add_argument('--all', action='store_true', help='Process all datasets')
    args = parser.parse_args()
    
    if args.all:
        for dataset in ['Earthquake', 'COVID19', 'Citibike', 'Crime']:
            try:
                process_dataset(dataset, args.dim)
            except Exception as e: 
                print(f"Error processing {dataset}: {e}")
    else:
        process_dataset(args.dataset, args.dim)
