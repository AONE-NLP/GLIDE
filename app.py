import torch
import torch.nn as nn
import numpy as np
from DSTPP import GaussianDiffusion_ST, Transformer_ST, Model_all, ST_Diffusion
from torch.optim import AdamW
import argparse
from torch_geometric.utils import to_dense_batch
from torch_ema import ExponentialMovingAverage
from DSTPP.Dataset import get_dataloader_preprocessed
import time
import setproctitle
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_WINDOW_CONFIGS = {
    'Earthquake': {'TEMP_WINDOWS':   [64, 32, 16, 8], 'SPAT_WINDOWS': [64, 32, 16, 8]},
    'COVID19': {'TEMP_WINDOWS': [14, 7], 'SPAT_WINDOWS': [14, 7]},
    'Citibike': {'TEMP_WINDOWS': [32, 16, 8, 4], 'SPAT_WINDOWS':  [8, 6, 4, 2]},
    'Crime': {'TEMP_WINDOWS': [32, 16, 8, 4], 'SPAT_WINDOWS': [32, 16, 8, 4]},
}

def plot_error_scatter(real_locs, errors, epoch, save_path):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(real_locs[:, 0], real_locs[:, 1],
                          c=errors, cmap='coolwarm', s=10, alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Prediction Error Scatter (Epoch: {epoch})')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Prediction Error (distance)')
    plt.grid(True, linestyle='--', alpha=0.4)
    filepath = os.path.join(save_path, f'error_scatter_epoch_{epoch}.png')
    plt.savefig(filepath)
    print(f"Error scatter saved to {filepath}")
    plt.close()

class AutomaticWeightedLoss(nn.Module):
    """
    Learnable uncertainty weighting for multi-task losses.
    Each parameter corresponds to log(sigma^2).
    """
    def __init__(self, num_losses=4):
        super(AutomaticWeightedLoss, self).__init__()
        self.params = nn.Parameter(torch.zeros(num_losses))

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):

            precision = 0.5 * torch.exp(-self.params[i])
            loss_sum += precision * loss + 0.5 * self.params[i]
        return loss_sum



def save_checkpoint(path, epoch, step, model, optimizer,scaler=None, best_metric=None, awl=None, scheduler=None, ema=None):
    ckpt = {
        'epoch': epoch,
        'step': step,
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'awl': awl.state_dict() if awl is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'ema': ema.state_dict() if ema is not None else None,
        'best_metric': best_metric,
        'rng': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate()
        }
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer,scaler, device, awl=None, scheduler=None, ema=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    if optimizer is not None and 'optim' in ckpt:
        optimizer.load_state_dict(ckpt['optim'])
    if scaler is not None and 'scaler' in ckpt and ckpt['scaler'] is not None:
        scaler.load_state_dict(ckpt['scaler'])
        print("Scaler state loaded.")
    if awl is not None and 'awl' in ckpt and ckpt['awl'] is not None:
        awl.load_state_dict(ckpt['awl'])
        print("AutomaticWeightedLoss state loaded.")
    if scheduler is not None and 'scheduler' in ckpt and ckpt['scheduler'] is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
        print("Scheduler state loaded.")
        

    if ema is not None and 'ema' in ckpt and ckpt['ema'] is not None:
        ema.load_state_dict(ckpt['ema'])
        print("EMA state loaded.")
    rng = ckpt.get('rng', None)
    if rng is not None:
        torch_state = rng.get('torch')
        if torch_state is not None:
            if not torch.is_tensor(torch_state):
                torch_state = torch.tensor(torch_state, dtype=torch.uint8)
            torch_state = torch_state.detach().cpu()
            if torch_state.dtype != torch.uint8:
                torch_state = torch_state.to(torch.uint8)
            torch_state = torch_state.type(torch.ByteTensor)
            torch.set_rng_state(torch_state)
        cuda_state = rng.get('cuda')
        if cuda_state is not None and torch.cuda.is_available():
            byte_states = []
            for cs in cuda_state:
                if not torch.is_tensor(cs):
                    cs = torch.tensor(cs, dtype=torch.uint8)
                cs = cs.detach().cpu()
                if cs.dtype != torch.uint8:
                    cs = cs.to(torch.uint8)
                byte_states.append(cs.type(torch.ByteTensor))
            torch.cuda.set_rng_state_all(byte_states)
        np.random.set_state(rng['numpy'])
        random.setstate(rng['python'])
    start_epoch = ckpt.get('epoch', -1) + 1
    step = ckpt.get('step', 0)
    best_metric = ckpt.get('best_metric', float('inf'))
    return start_epoch, step, best_metric

def setup_init(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def get_args():
    parser = argparse.ArgumentParser(description='Train or evaluate GLIDE.')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--total_epochs', type=int, default=1000, help='maximum training epochs')
    parser.add_argument('--machine', type=str, default='none', help='optional machine identifier')
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l1', 'l2', 'Euclid'], help='diffusion loss type')
    parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'], help='diffusion beta schedule')
    parser.add_argument('--dim', type=int, default=2, help='spatial dimension', choices=[1, 2, 3])
    parser.add_argument('--dataset', type=str, default='Earthquake', choices=['Citibike', 'Earthquake', 'HawkesGMM', 'Pinwheel', 'COVID19', 'Mobility', 'HawkesGMM_2d', 'Independent', 'Crime', 'Earthquake-o'], help='dataset name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--timesteps', type=int, default=100, help='diffusion training timesteps')
    parser.add_argument('--samplingsteps', type=int, default=100, help='diffusion sampling steps')
    parser.add_argument('--objective', type=str, default='pred_noise', help='diffusion objective')
    parser.add_argument('--cuda_id', type=str, default='0', help='CUDA device id')
    parser.add_argument('--resume', action='store_true', help='resume from a checkpoint')
    parser.add_argument('--ckpt', type=str, default='', help='checkpoint path, e.g. ./ModelSave/.../last.ckpt')
    parser.add_argument('--plot_heatmap', action='store_true', help='save heatmaps and trajectory plots during evaluation')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args

opt = get_args()
device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")
def safe_item(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return x

if opt.dataset == 'HawkesGMM':
    opt.dim=1

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)


def plot_heatmap_comparison(real_locs, pred_locs, epoch, save_path, bins=50):
    os.makedirs(save_path, exist_ok=True)
    


    x_min = min(real_locs[:, 0].min(), pred_locs[:, 0].min())
    x_max = max(real_locs[:, 0].max(), pred_locs[:, 0].max())
    y_min = min(real_locs[:, 1].min(), pred_locs[:, 1].min())
    y_max = max(real_locs[:, 1].max(), pred_locs[:, 1].max())
    
    common_bin_range = [[x_min, x_max], [y_min, y_max]]


    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    fig.suptitle(f'Spatial Distribution Heatmap (Epoch: {epoch})', fontsize=16)
    

    sns.histplot(
        x=real_locs[:, 0], y=real_locs[:, 1], 
        bins=bins, 
        binrange=common_bin_range,
        cmap='viridis', ax=axes[0], cbar=True
    )
    axes[0].set_title('Ground Truth Distribution')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    

    sns.histplot(
        x=pred_locs[:, 0], y=pred_locs[:, 1], 
        bins=bins, 
        binrange=common_bin_range,
        cmap='viridis', ax=axes[1], cbar=True
    )
    axes[1].set_title('Predicted Distribution (Single Sample)')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_path, f'heatmap_epoch_{epoch}.png')
    plt.savefig(filepath, dpi=150)
    print(f"Heatmap saved to {filepath}")
    plt.close(fig)


def plot_leapfrog_kde(real_locs, baseline_traj, leapfrog_traj, epoch, save_path, num_timesteps):
    """
    Plot KDE trajectories for standard diffusion and leap inference.
    """
    os.makedirs(save_path, exist_ok=True)
    
    all_x = np.concatenate([real_locs[:, 0]] + [arr[:, 0] for arr in baseline_traj.values()] + [arr[:, 0] for arr in leapfrog_traj.values()])
    all_y = np.concatenate([real_locs[:, 1]] + [arr[:, 1] for arr in baseline_traj.values()] + [arr[:, 1] for arr in leapfrog_traj.values()])
    
    x_margin = (all_x.max() - all_x.min()) * 0.05
    y_margin = (all_y.max() - all_y.min()) * 0.05
    x_lim = (all_x.min() - x_margin, all_x.max() + x_margin)
    y_lim = (all_y.min() - y_margin, all_y.max() + y_margin)
    

    fig, axes = plt.subplots(2, 6, figsize=(30, 9), sharex=True, sharey=True)
    fig.suptitle(f'Generative Trajectories: Standard Diffusion vs. Leapfrog Inference', fontsize=24, fontweight='bold')
    

    base_steps = sorted(list(baseline_traj.keys()), reverse=True)
    for i, step in enumerate(base_steps[:5]):
        ax = axes[0, i]
        sns.kdeplot(x=baseline_traj[step][:, 0], y=baseline_traj[step][:, 1], 
                    fill=True, cmap="Blues", ax=ax, thresh=0.05, levels=10)
        ax.set_title(f'Standard: Step {step}', fontsize=18)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        if i == 0: ax.set_ylabel('Latitude', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.5)


    ax_gt_top = axes[0, 5]
    sns.kdeplot(x=real_locs[:, 0], y=real_locs[:, 1], fill=True, cmap="Reds", ax=ax_gt_top, thresh=0.05, levels=10)
    ax_gt_top.set_title('Ground Truth (Real)', fontsize=18, color='darkred', fontweight='bold')
    ax_gt_top.set_xlim(x_lim)
    ax_gt_top.set_ylim(y_lim)
    ax_gt_top.grid(True, linestyle='--', alpha=0.5)


    leap_steps = ['Prior'] + sorted([k for k in leapfrog_traj.keys() if isinstance(k, int)], reverse=True)

    for i, step in enumerate(leap_steps[:5]): 
        ax = axes[1, i]
        sns.kdeplot(x=leapfrog_traj[step][:, 0], y=leapfrog_traj[step][:, 1], 
                    fill=True, cmap="Oranges", ax=ax, thresh=0.05, levels=10)
        title = 'Leap Inference: Prior' if step == 'Prior' else f'Leap Inference: Step {step}'
        ax.set_title(title, fontsize=18, color='darkorange')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel('Longitude', fontsize=14)
        if i == 0: ax.set_ylabel('Latitude', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.5)
            

    ax_gt = axes[1, 5]
    sns.kdeplot(x=real_locs[:, 0], y=real_locs[:, 1], fill=True, cmap="Reds", ax=ax_gt, thresh=0.05, levels=10)
    ax_gt.set_title('Ground Truth (Real)', fontsize=18, color='darkred', fontweight='bold')
    ax_gt.set_xlim(x_lim)
    ax_gt.set_ylim(y_lim)
    ax_gt.set_xlabel('Longitude', fontsize=14)
    ax_gt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filepath = os.path.join(save_path, f'leapfrog_trajectory_epoch_{epoch}.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Leapfrog KDE Trajectory saved to {filepath}")

def data_loader(writer):
    print("Loading preprocessed data...")
    base_path = f'dataset/{opt.dataset}/'
    with open(os.path.join(base_path, 'data_train_processed.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(base_path, 'data_val_processed.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(base_path, 'data_test_processed.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    with open(os.path.join(base_path, 'stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    
    trainloader = get_dataloader_preprocessed(train_data, opt.batch_size, shuffle=True)
    valloader = get_dataloader_preprocessed(val_data, opt.batch_size, shuffle=False)
    testloader = get_dataloader_preprocessed(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, valloader, stats

@torch.no_grad()
def generate_trajectories_for_plot(Model, batch, loc_rng, loc_min, opt_dim):
    """
    Collect intermediate spatial coordinates for standard diffusion and leap inference.
    """
    event_time, event_loc, cond_features, mask = Batch2toModel(batch, Model.transformer)
    bs = event_time.shape[0]
    real_seq_len = event_time.shape[1]
    device = event_time.device
    num_timesteps = Model.diffusion.num_timesteps
    
    mask_bool = (mask > 0).float()
    if mask_bool.dim() == 2: mask_bool = mask_bool.unsqueeze(-1)
    
    def get_real_coords(tensor, is_normalized=True):
        spatial = tensor[:, :, -opt_dim:]
        if is_normalized:
            spatial = (spatial + 1) * 0.5 
        valid_spatial = spatial[mask_bool.bool().squeeze(-1)]
        return (valid_spatial * loc_rng + loc_min).cpu().numpy()


    baseline_traj = {}
    shape = (bs, real_seq_len, Model.diffusion.seq_length)
    img_base = torch.randn(shape, device=device)
    if mask is not None: img_base = img_base * mask_bool + (-1.0) * (1.0 - mask_bool)
    
    steps_to_save_base = [num_timesteps, int(num_timesteps*0.75), int(num_timesteps*0.5), int(num_timesteps*0.25), 0]
    baseline_traj[num_timesteps] = get_real_coords(img_base)

    x_start_base = None
    for t in reversed(range(0, num_timesteps)):
        self_cond = x_start_base if Model.diffusion.self_condition else None
        img_base, x_start_base, _ = Model.diffusion.p_sample(img_base, t, self_cond, cond=cond_features, mask=mask_bool)
        if mask is not None: img_base = img_base * mask_bool + (-1.0) * (1.0 - mask_bool)
        
        if t in steps_to_save_base:
            baseline_traj[t] = get_real_coords(img_base)


    leapfrog_traj = {}
    

    mean_pred_t, mean_pred_s = Model.diffusion.model.joint_mean_predictor(cond_features, mask=mask_bool)
    mean_pred = torch.cat([mean_pred_t, mean_pred_s], dim=-1).clamp(0, 1)
    x_prior = mean_pred * 2 - 1 
    leapfrog_traj['Prior'] = get_real_coords(x_prior)
    
    start_ratio = 0.5  
    start_t = int(num_timesteps * start_ratio)  
    start_t = max(start_t, 10) 
    
    noise = torch.randn_like(x_prior)
    if mask is not None: noise = noise * mask_bool
    t_tensor = torch.full((bs,), start_t, device=device, dtype=torch.long)
    

    img_leap = Model.diffusion.q_sample(x_start=x_prior, t=t_tensor, noise=noise, mask=mask_bool)
    leapfrog_traj[start_t] = get_real_coords(img_leap)
    


    steps_to_save_leap = [start_t, int(start_t * 0.5), int(start_t * 0.25), 0]
    
    for t in reversed(range(0, start_t)):
        self_cond = x_start_leap if Model.diffusion.self_condition else None
        

        img_leap, x_start_leap, _ = Model.diffusion.p_sample(img_leap, t, self_cond, cond=cond_features, mask=mask_bool)
        
 
            
        if mask is not None: img_leap = img_leap * mask_bool + (-1.0) * (1.0 - mask_bool)
        
        if t in steps_to_save_leap:
            leapfrog_traj[t] = get_real_coords(img_leap)

    real_locs = get_real_coords(event_loc, is_normalized=False) 
    
    return real_locs, baseline_traj, leapfrog_traj
def Batch2toModel(batch, transformer):
    batch = batch.to(device, non_blocking=True)
    


    # non_pad_mask: [Batch, SeqLen, 1]
    enc_out, non_pad_mask = transformer(batch) 
    
    cond_features = enc_out 
    

    target_time, _ = to_dense_batch(batch.target_time, batch.batch, fill_value=0.0)
    target_loc, _ = to_dense_batch(batch.target_loc, batch.batch, fill_value=0.0)
    
    mask = non_pad_mask.float() # [B, L, 1]
    


    cond_features = cond_features * mask
    
    # time: [B, L] -> [B, L, 1] * [B, L, 1]
    event_time = target_time.unsqueeze(-1) * mask
    
    # loc: [B, L, Dim] * [B, L, 1]
    event_loc = target_loc * mask
    
    return event_time, event_loc, cond_features, mask
 

def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current+1) / epoch_num


if __name__ == "__main__":
    setup_init(opt)
    setproctitle.setproctitle("Model-Training")
    print('dataset:{}'.format(opt.dataset))
    
    torch.set_float32_matmul_precision('medium')
    setup_init(opt)

    logdir = "./logs/{}_timesteps_{}_gai".format( opt.dataset,  opt.timesteps)
    model_path = './ModelSave/dataset_{}_timesteps_{}_gai'.format(opt.dataset, opt.timesteps) 

    if not os.path.exists('./ModelSave'):
        os.mkdir('./ModelSave')
    if 'train' in opt.mode and not os.path.exists(model_path):
        os.mkdir(model_path)
    window_config = DATASET_WINDOW_CONFIGS.get(opt.dataset, {'TEMP_WINDOWS': [64, 32, 8, 2], 'SPAT_WINDOWS': [64, 32, 8, 2]})
    TEMP_WINDOWS = window_config['TEMP_WINDOWS']
    SPAT_WINDOWS = window_config['SPAT_WINDOWS']

    base_dim = 1 +1+ opt.dim * 2 
    temp_feat_dim = len(TEMP_WINDOWS) * 3
    spat_feat_dim = len(SPAT_WINDOWS) * 3 * opt.dim

    total_input_dim = base_dim + temp_feat_dim + spat_feat_dim
    print(f"Calculated Dynamic Input Dimension: {total_input_dim}")

    writer = SummaryWriter(log_dir = logdir,flush_secs=5)


    if opt.dataset == 'Earthquake':
        config = {
           'd_model': 64,          
            'n_layers': 4,
            'n_head': 4,
            'dropout': 0.4,         
            'spatial_dropout': 0.4, 
            'base_lr': 3e-4,        
            'num_units': 96,
            'interaction_start': 4,#4
            'start_ratio': 0.4,
            'guidance_scale': 0.1,
        }
    elif opt.dataset == 'COVID19': 
        config = {

        'd_model': 32,          
        'num_units': 32,        
        'n_layers': 4,
        'n_head': 4,
        'interaction_start': 2, #2
        'dropout': 0.5,
        'spatial_dropout': 0.5,
        'base_lr': 5e-4,
        'start_ratio': 0.4,
        'guidance_scale': 0.1,
        }
    
    elif opt.dataset == 'Citibike':
        config = {
            'd_model': 64,         
            'num_units': 64,
            'n_layers': 5,
            'n_head': 4,            # 64/4=16
            'interaction_start': 3,
            'dropout': 0.5,          
            'spatial_dropout': 0.5,
            'base_lr': 3e-4,
            'start_ratio': 0.4,
            'guidance_scale': 0.1,
        }
    elif opt.dataset == 'Crime':
        config = {

            'd_model': 32,          
            'num_units': 32,

            'n_layers': 2,          
            'n_head': 4,            
            'interaction_start': 0, 

            'dropout': 0.5,         
            'spatial_dropout': 0.5, 

            'base_lr': 3e-4,        
            'start_ratio': 0.4,
            'guidance_scale': 0.5,  
        }
    else:

        config = {
            'd_model': 96,
            'n_layers': 6,
            'n_head': 6,
            'dropout': 0.3,
            'spatial_dropout': 0.3,
            'base_lr': 5e-4,
            
            'num_units': 128,
            'interaction_start': 4,
            'start_ratio': 0.6,
            'guidance_scale': 0.25,
        }

    print(f"Using config for {opt.dataset}:  {config}")

    

    model = ST_Diffusion(
        n_steps=opt.timesteps,
        dim=1 + opt.dim,
        num_units=config['num_units'],
        condition=True,
        cond_dim=config['d_model'],
        interaction_start=config['interaction_start'],
        num_heads=config['n_head']
    ).to(device)

    diffusion = GaussianDiffusion_ST(
        model,
        loss_type=opt.loss_type,
        seq_length=1 + opt.dim,
        timesteps=opt.timesteps,
        sampling_timesteps=opt.samplingsteps,
        objective=opt.objective,
        beta_schedule=opt.beta_schedule,
        leapfrog_start_ratio=config['start_ratio'],
        leapfrog_guidance_scale=config['guidance_scale']
    ).to(device)

    transformer = Transformer_ST(
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_head=config['n_head'],
        dropout=config['dropout'],
        spatial_dropout=config['spatial_dropout'],
        device=device,
        loc_dim=opt.dim,
        input_dim=total_input_dim
    ).to(device)

    

    Model = Model_all(transformer, diffusion).to(device)
    awl = AutomaticWeightedLoss(num_losses=4)
    torch.nn.init.constant_(awl.params, 0.0)
    awl = awl.to(device)
    trainloader, testloader, valloader, stats = data_loader(writer)
    loc_min = torch.tensor(stats['loc_min'], dtype=torch.float32).to(device)
    loc_max = torch.tensor(stats['loc_max'], dtype=torch.float32).to(device)
    loc_rng = loc_max - loc_min
    time_diff_min = torch.tensor(stats['time_diff_min'], dtype=torch.float32).to(device)
    time_diff_max = torch.tensor(stats['time_diff_max'], dtype=torch.float32).to(device)
    time_diff_rng = time_diff_max - time_diff_min
    spatial_params = []
    temporal_params = []
    other_params = []
   
    for name, param in Model.named_parameters():
        if not param.requires_grad:
            continue
        
        name_lower = name.lower()
        
        if any(k in name_lower for k in ['spatial', 'loc', 'gat', 'fourier']):
            spatial_params.append(param)
        elif any(k in name_lower for k in ['temporal', 'time', 'rope']):
            temporal_params.append(param)
        else:
            other_params.append(param)

    print(f"Parameter groups: spatial={len(spatial_params)}, temporal={len(temporal_params)}, other={len(other_params)}")


    optimizer = AdamW([
        {'params':  spatial_params, 'lr': config['base_lr'] },
        {'params':  temporal_params, 'lr': config['base_lr'] },
        {'params':  other_params, 'lr': config['base_lr']},
        {'params': awl.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=opt.total_epochs,
    eta_min=1e-6
)

    # EMA
    ema = ExponentialMovingAverage(Model.parameters(), decay=0.999)


    scaler = torch.amp.GradScaler(device='cuda', enabled=opt.cuda)


    start_epoch, step = 0, 0
    best_val_loss = float('inf')
    best_val_nll = float('inf')
    best_test_nll_temporal = float('inf')
    best_test_nll_spatial = float('inf')
    best_test_nll_mean = float('inf')
    if opt.resume and opt.ckpt: 
        ckpt_path = os.path.abspath(os.path.expanduser(opt.ckpt))
        assert os.path.isfile(ckpt_path), f'checkpoint does not exist: {ckpt_path}'
        print(f'Loading checkpoint:  {ckpt_path}')
        
        start_epoch, step, best_metrics = load_checkpoint(
            ckpt_path, Model, optimizer, scaler, device, 
            awl=awl, scheduler=scheduler, ema=ema
        )

        if isinstance(best_metrics, dict):
            best_val_loss = best_metrics.get('loss', float('inf'))
            best_val_nll = best_metrics.get('nll', float('inf'))
        print(f'Resumed from epoch {start_epoch}, global step {step}')

    
    early_stop = 0

    for itr in range(start_epoch, opt.total_epochs):
        print('epoch:{}'.format(itr))

        if itr % 10 == 0 and itr >0 or opt.mode == 'test':
            print('Evaluate!')

            with ema.average_parameters():
                with torch.no_grad():
                    Model.eval()
                    n_samples = 20
                    
                    # ==========================================

                    # ==========================================
                    loss_test_all = 0.0
                    

                    nll_sum_all = 0.0       
                    nll_sum_temporal = 0.0
                    nll_sum_spatial = 0.0
                    

                    total_valid_events = 0.0
                    total_sequences = 0.0
                    

                    mae_temporal, rmse_temporal, mae_spatial = 0.0, 0.0, 0.0
                    total_events_mae = 0.0


                    
                    for batch in tqdm(valloader, desc=f'Epoch {itr} Val', leave=False):

                        event_time, event_loc, cond_features, mask = Batch2toModel(batch, Model.transformer)
                        
                        bs = event_time.shape[0]
                        current_valid_num = mask.sum().item()
                        
                        total_valid_events += current_valid_num
                        total_sequences += bs



                        vb_sum, vb_t_sum, vb_s_sum = Model.diffusion.NLL_cal(
                            torch.cat((event_time, event_loc), dim=-1), 
                            cond_features, 
                            mask=mask
                        )
                        nll_sum_all += safe_item(vb_sum)
                        nll_sum_temporal += safe_item(vb_t_sum)
                        nll_sum_spatial += safe_item(vb_s_sum)


                        loss_diff_t, loss_diff_s = Model.diffusion(
                            torch.cat((event_time, event_loc), dim=-1), 
                            cond_features, 
                            mask=mask
                        )
                       
                        current_loss = loss_diff_t + loss_diff_s 
                        loss_test_all += safe_item(current_loss) * bs


                        Model.diffusion.model._last_spatial_aux = None
                        Model.diffusion.model._last_temporal_aux = None


                        batch_samples_t = []
                        batch_samples_s = []
                        for _ in range(n_samples):
                            sampled_seq_k = Model.diffusion.sample(
                            batch_size=event_time.shape[0], 
                            cond=cond_features, 
                            mask=mask  ,
                            mode='refine' #
                        )
                            batch_samples_t.append(sampled_seq_k[: , : , :1]. cpu()) 
                            batch_samples_s.append(sampled_seq_k[:, :, -opt. dim: ].cpu())

                        mean_t_norm = torch.stack(batch_samples_t).mean(dim=0).to(device)
                        mean_x_norm = torch.stack(batch_samples_s).mean(dim=0).to(device)
                        median_t_norm = torch.stack(batch_samples_t).median(dim=0).values.to(device)
                        median_x_norm = torch.stack(batch_samples_s).median(dim=0).values.to(device)

                        valid_mask_flat = mask.bool().squeeze(-1)
                        if valid_mask_flat.any():

                            real_t = event_time.squeeze(-1)[valid_mask_flat] * time_diff_rng + time_diff_min                          
                            real_x = event_loc[valid_mask_flat] * loc_rng + loc_min

                            gen_t_mean = mean_t_norm.squeeze(-1)[valid_mask_flat] * time_diff_rng + time_diff_min
                           

                            gen_x_mean = mean_x_norm[valid_mask_flat] * loc_rng + loc_min
                            #gen_x_median = median_x_norm[valid_mask_flat] * loc_rng + loc_min
                          
                            mae_temporal += torch.abs(real_t - gen_t_mean).sum().item()
                            mae_spatial += torch.norm(real_x - gen_x_mean, dim=-1).sum().item()
                            

                            rmse_temporal += ((real_t - gen_t_mean) ** 2).sum().item()
                            
                            total_events_mae += valid_mask_flat.sum().item()

                    # --- Validation Summary ---
                    val_loss_mean = loss_test_all / total_sequences if total_sequences > 0 else float('inf')
                    

                    if total_valid_events > 0:
                        val_nll_temporal = nll_sum_temporal / total_valid_events 
                        val_nll_spatial = (nll_sum_spatial / total_valid_events) / opt.dim
                        val_nll_mean = (nll_sum_all / total_valid_events) / (1 + opt.dim)
                    else:
                        val_nll_mean = float('inf')

                    print(f"Epoch {itr} Val NLL (Per Event): Total={val_nll_mean:.4f}, T={val_nll_temporal:.4f}, S={val_nll_spatial:.4f}")

                   
                    

                    if val_loss_mean < best_val_loss:
                        best_val_loss = val_loss_mean
                        save_checkpoint(os.path.join(model_path, 'best_loss.ckpt'), itr, step, Model, optimizer, scaler=scaler, best_metric={'loss': best_val_loss, 'nll': best_val_nll}, awl=awl, scheduler=scheduler, ema=ema)

                    if val_nll_mean < best_val_nll:
                        best_val_nll = val_nll_mean
                        save_checkpoint(os.path.join(model_path, 'best_nll.ckpt'), itr, step, Model, optimizer, scaler=scaler, best_metric={'loss': best_val_loss, 'nll': best_val_nll}, awl=awl, scheduler=scheduler, ema=ema)
                    # Tensorboard
                    writer.add_scalar('Evaluation/loss_val', val_loss_mean, itr)
                    writer.add_scalar('Evaluation/NLL_val_PerEvent', val_nll_mean, itr)
                    writer.add_scalar('Evaluation/NLL_temporal_val_PerEvent', val_nll_temporal, itr)
                    writer.add_scalar('Evaluation/NLL_spatial_val_PerEvent', val_nll_spatial, itr)
                    if total_events_mae > 0:
                        writer.add_scalar('Evaluation/mae_temporal_val', mae_temporal / total_events_mae, itr)
                        writer.add_scalar('Evaluation/rmse_temporal_val', np.sqrt(rmse_temporal / total_events_mae), itr)
                        writer.add_scalar('Evaluation/distance_spatial_val', mae_spatial / total_events_mae, itr)

                    # ==========================================

                    # ==========================================
                    loss_test_all = 0.0
                    nll_sum_all, nll_sum_temporal, nll_sum_spatial = 0.0, 0.0, 0.0
                    mae_temporal_test, rmse_temporal_test, mae_spatial_test = 0.0, 0.0, 0.0
                    total_valid_events_test = 0.0
                    total_sequences_test = 0.0
                    mae_spatial_test_median = 0.0
                    
                    all_real_locs, all_pred_locs, all_errors = [], [], []
                    total_inference_time = 0.0
                    total_generated_samples = 0
                    for batch in tqdm(testloader, desc=f'Epoch {itr} Test', leave=False):
                        Model.diffusion.model._last_spatial_aux = None
                        Model.diffusion.model._last_temporal_aux = None
                        
                        event_time, event_loc, cond_features, mask = Batch2toModel(batch, Model.transformer)
                        bs = event_time.shape[0]
                        current_valid_num = mask.sum().item()
                        
                        total_valid_events_test += current_valid_num
                        total_sequences_test += bs
                        

                        vb_sum, vb_t_sum, vb_s_sum = Model.diffusion.NLL_cal(
                            torch.cat((event_time, event_loc), dim=-1), cond_features, mask=mask
                        )
                        nll_sum_all += safe_item(vb_sum)
                        nll_sum_temporal += safe_item(vb_t_sum)
                        nll_sum_spatial += safe_item(vb_s_sum)


                        batch_samples_t = []
                        batch_samples_s = []
                        t_start = time.time()
                        for _ in range(n_samples):
                            sampled_seq_k = Model.diffusion.sample(
                                batch_size=bs, 
                                cond=cond_features, 
                                mask=mask,
                                mode='refine',
 
                            )

                            batch_samples_t.append(sampled_seq_k[:, :, :1].cpu()) 
                            batch_samples_s.append(sampled_seq_k[:, :, -opt.dim:].cpu())
                        t_end = time.time()
                        total_inference_time += (t_end - t_start)
                        total_generated_samples += (bs * n_samples)
                        mean_t_norm = torch.stack(batch_samples_t).mean(dim=0).to(device) 
                        mean_x_norm = torch.stack(batch_samples_s).mean(dim=0).to(device)
                        median_t_norm = torch.stack(batch_samples_t).median(dim=0).values.to(device)
                        median_x_norm = torch.stack(batch_samples_s).median(dim=0).values.to(device)
                        valid_mask_flat = mask.bool().squeeze(-1)
                  
                        if valid_mask_flat.any():
                            real_t = event_time.squeeze(-1)[valid_mask_flat] * time_diff_rng + time_diff_min
                            

                            real_x = event_loc[valid_mask_flat] * loc_rng + loc_min
                            

                            gen_t_mean = mean_t_norm.squeeze(-1)[valid_mask_flat] * time_diff_rng + time_diff_min
                           

                            gen_x_mean = mean_x_norm[valid_mask_flat] * loc_rng + loc_min
                            gen_x_median = median_x_norm[valid_mask_flat] * loc_rng + loc_min
                            sample_0 = batch_samples_s[0].to(device)
                            
                        
                           


                            mae_temporal_test += torch.abs(real_t - gen_t_mean).sum().item()
                            mae_spatial_test += torch.norm(real_x - gen_x_mean, dim=-1).sum().item()
                            mae_spatial_test_median += torch.norm(real_x - gen_x_median, dim=-1).sum().item()

                            rmse_temporal_test += ((real_t - gen_t_mean) ** 2).sum().item()


                            if opt.plot_heatmap:
                                gen_x_sample = sample_0[valid_mask_flat] * loc_rng + loc_min
                                all_real_locs.append(real_x.cpu().numpy())
                                all_pred_locs.append(gen_x_sample.cpu().numpy()) 
                                all_errors.append(torch.norm(real_x - gen_x_mean, dim=-1).cpu().numpy())


                        loss_diff_t, loss_diff_s = Model.diffusion(torch.cat((event_time, event_loc), dim=-1), cond_features, mask=mask)
                        current_loss = loss_diff_t + loss_diff_s
                        loss_test_all += safe_item(current_loss) * bs

                    # --- Test Summary ---
                    if total_valid_events_test > 0:
                        test_nll_t = nll_sum_temporal / total_valid_events_test
                        test_nll_s = (nll_sum_spatial / total_valid_events_test) / opt.dim
                        test_nll_mean = (nll_sum_all / total_valid_events_test) / (1 + opt.dim)
                        test_mae_s_median = mae_spatial_test_median / total_valid_events_test
                        test_mae_t = mae_temporal_test / total_valid_events_test
                        test_rmse_t = np.sqrt(rmse_temporal_test / total_valid_events_test)
                        test_mae_s = mae_spatial_test / total_valid_events_test
                    else:
                        test_nll_mean = float('inf')
                        test_mae_t = 0.0
                        test_nll_t = float('inf')
                        test_nll_s = float('inf')

                    if opt.plot_heatmap and len(all_pred_locs) > 0:
                        final_real = np.concatenate(all_real_locs, axis=0)
                        final_pred = np.concatenate(all_pred_locs, axis=0)
                        final_err = np.concatenate(all_errors, axis=0)
                        heatmap_save_path = os.path.join(model_path, 'heatmaps')
                        plot_heatmap_comparison(final_real, final_pred, itr, heatmap_save_path)
                        plot_error_scatter(final_real, final_err, itr, heatmap_save_path)
                        num_batches_to_plot = len(testloader) if opt.mode == 'test' else 1 
                        
                        accumulated_gt = []
                        accumulated_base = {}
                        accumulated_leap = {}
                        
                        print(f"Generating KDE trajectory plot using {num_batches_to_plot} batches...")
                        
                        for b_idx, sample_batch in enumerate(testloader):
                            if b_idx >= num_batches_to_plot:
                                break
                                
                            gt_locs, base_traj, leap_traj = generate_trajectories_for_plot(
                                Model, sample_batch, loc_rng, loc_min, opt.dim
                            )
                            
                            accumulated_gt.append(gt_locs)
                            

                            for k, v in base_traj.items():
                                if k not in accumulated_base: accumulated_base[k] = []
                                accumulated_base[k].append(v)
                                

                            for k, v in leap_traj.items():
                                if k not in accumulated_leap: accumulated_leap[k] = []
                                accumulated_leap[k].append(v)


                        final_gt_locs = np.concatenate(accumulated_gt, axis=0)
                        final_base_traj = {k: np.concatenate(v, axis=0) for k, v in accumulated_base.items()}
                        final_leap_traj = {k: np.concatenate(v, axis=0) for k, v in accumulated_leap.items()}
                        

                        plot_leapfrog_kde(final_gt_locs, final_base_traj, final_leap_traj, itr, heatmap_save_path, opt.timesteps)

                    writer.add_scalar('Evaluation/loss_test', loss_test_all/total_sequences_test, itr)
                    writer.add_scalar('Evaluation/NLL_test_PerEvent', test_nll_mean, itr)
                    writer.add_scalar('Evaluation/NLL_temporal_test_PerEvent', test_nll_t, itr)
                    writer.add_scalar('Evaluation/NLL_spatial_test_PerEvent', test_nll_s, itr)
                    writer.add_scalar('Evaluation/mae_temporal_test', test_mae_t, itr)
                    writer.add_scalar('Evaluation/rmse_temporal_test', test_rmse_t, itr)
                    writer.add_scalar('Evaluation/distance_spatial_test', test_mae_s, itr)
                    writer.add_scalar('Evaluation/distance_spatial_test_median', test_mae_s_median, itr)
                    print(f"Epoch {itr} Test NLL: {test_nll_mean:.4f} (T={test_nll_t:.4f}, S={test_nll_s:.4f})")
                    if test_nll_mean < best_test_nll_mean:
                        best_test_nll_mean = test_nll_mean
                        save_checkpoint(
                            os.path.join(model_path, 'best_test_nll.ckpt'), 
                            itr, step, Model, optimizer, 
                            scaler=scaler, 
                            best_metric={'test_nll': best_test_nll_mean}, 
                            awl=awl if 'awl' in locals() else None, 
                            scheduler=scheduler, 
                            ema=ema
                        )
                        print(f"🔥 New Best Test NLL: {best_test_nll_mean:.4f} -> Saved to best_test_nll.ckpt")
                    if total_generated_samples > 0:
                        avg_time_per_sample = total_inference_time / total_generated_samples

                        throughput = total_generated_samples / total_inference_time
                        
                        print(f"⏱️ [Inference Speed]: Total Time: {total_inference_time:.2f}s | "
                              f"Avg Time/Seq: {avg_time_per_sample:.4f}s | "
                              f"Throughput: {throughput:.2f} seq/s")
                        writer.add_scalar('Evaluation/Total Inference_Time', total_inference_time, itr)
                        writer.add_scalar('Evaluation/Time_per_sample', avg_time_per_sample, itr)


                    improved_temporal = test_nll_t < best_test_nll_temporal
                    improved_spatial = test_nll_s < best_test_nll_spatial
                    

                    if improved_temporal:
                        best_test_nll_temporal = test_nll_t
                    if improved_spatial:
                        best_test_nll_spatial = test_nll_s
                        

                    if improved_temporal or improved_spatial:
                        print(f"--> Improvement! (Time: {improved_temporal}, Space: {improved_spatial}) - Patience Reset.")
                        early_stop = 0 
                    else:
                        early_stop += 1
                        print(f"--> No improvement in BOTH metrics.Patience: {early_stop}/50")


                    if early_stop >= 10:
                        print("Early stopping triggered based on dual-metric check.")
                        break
                    if opt.mode == 'test':
                        print("Test finished. Exiting...")
                        break
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar(tag='Statistics/lr',scalar_value=current_lr,global_step=itr)

        Model.train()

        loss_all, vb_all, vb_temporal_all, vb_spatial_all, total_num = 0.0, 0.0, 0.0, 0.0, 0.0
        for batch in trainloader:
            Model.diffusion.model._last_spatial_aux = None
            Model.diffusion.model._last_temporal_aux = None
            
            event_time, event_loc, cond_features, mask = Batch2toModel(batch, Model.transformer)
       
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=opt.cuda):
                loss_diff_t, loss_diff_s = Model.diffusion(
                    torch.cat((event_time, event_loc), dim=-1), 
                    cond_features, 
                    mask=mask
                )
                mean_loss_t = torch.tensor(0.0, device=device)
                mean_loss_s = torch.tensor(0.0, device=device)
                
                if Model.diffusion.model._last_temporal_aux is not None: 
                    pred_t = Model.diffusion.model._last_temporal_aux  # [B, L, 1]

                    mean_loss_t_raw = torch.abs(pred_t - event_time)  # [B, L, 1]
                    mean_loss_t = (mean_loss_t_raw * mask).sum() / mask.sum().clamp(min=1.)
                
                if Model.diffusion.model._last_spatial_aux is not None:
                    pred_s = Model.diffusion.model._last_spatial_aux  # [B, L, dim]
                    mean_loss_s_raw = torch.abs(pred_s - event_loc)  # [B, L, dim]

                    denom_s = mask.sum() * pred_s.shape[-1]
                    mean_loss_s = (mean_loss_s_raw * mask).sum() / denom_s.clamp(min=1.)

                
                loss = (loss_diff_t + 2*loss_diff_s) + 0.1 * (mean_loss_t + 2*mean_loss_s)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(Model.parameters(), 0.5)
            if itr < 20:
                awl.params.grad = None
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            

            bs = event_time.shape[0]
            
          
           # vb_all += safe_item(vb) 
            #vb_temporal_all += safe_item(vb_temporal) 
           # vb_spatial_all += safe_item(vb_spatial) 
            
            loss_all += safe_item(loss) * bs

            step += 1
            total_num += bs
            
        scheduler.step() 
        

        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar(tag='Statistics/lr_spatial', scalar_value=current_lr, global_step=itr)

        #with torch.cuda.device("cuda:{}".format(opt.cuda_id)):
         #   torch.cuda.empty_cache()

        writer.add_scalar(tag='Training/loss_epoch',scalar_value=loss_all/total_num,global_step=itr)
       # writer.add_scalar(tag='Training/NLL_epoch',scalar_value=vb_all/total_num,global_step=itr)
       # writer.add_scalar(tag='Training/NLL_temporal_epoch',scalar_value=vb_temporal_all/total_num,global_step=itr)
       # writer.add_scalar(tag='Training/NLL_spatial_epoch',scalar_value=vb_spatial_all/total_num,global_step=itr)

        save_checkpoint(os.path.join(model_path, 'last.ckpt'), itr, step, Model, optimizer,scaler=scaler, best_metric={'loss': best_val_loss, 'nll': best_val_nll},awl=awl,scheduler=scheduler,ema=ema)
