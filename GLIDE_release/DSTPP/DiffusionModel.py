import math
from random import random
from functools import partial
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from .Models import RotaryEmbedding, apply_rotary_pos_emb
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import numpy as np
import time



ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class ImprovedJointMeanPredictor(nn.Module):

    def __init__(self, cond_dim, hidden_dim, spatial_dim, num_heads=4):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        

        self.input_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        

        self.temporal_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        

        self.spatial_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        

        self.t2s_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.s2t_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        

        self.temporal_head = nn.Linear(hidden_dim, 1)
        self.spatial_head = nn.Linear(hidden_dim, spatial_dim)
        

        self.temporal_skip = nn.Linear(cond_dim, 1)
        self.spatial_skip = nn.Linear(cond_dim, spatial_dim)
        

        nn.init.zeros_(self.temporal_head.weight)
        nn.init.zeros_(self.temporal_head.bias)
        nn.init.zeros_(self.spatial_head.weight)
        nn.init.zeros_(self.spatial_head.bias)
        
    def forward(self, cond, mask=None):
        B, L, _ = cond.shape
        

        h = self.input_proj(cond)
        
        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=0.1 if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.hidden_dim)
        attn_out = self.attn_out_proj(attn_out)
        
        h = self.attn_norm(h + attn_out)
        

        h_t = self.temporal_norm(h + self.temporal_ffn(h))
        h_s = self.spatial_norm(h + self.spatial_ffn(h))
        

        gate_t = self.s2t_gate(torch.cat([h_t, h_s], dim=-1))
        gate_s = self.t2s_gate(torch.cat([h_s, h_t], dim=-1))
        
        h_t = h_t + gate_t * h_s
        h_s = h_s + gate_s * h_t
        

        out_t = self.temporal_head(h_t) + self.temporal_skip(cond) * 0.2
        out_s = self.spatial_head(h_s) + self.spatial_skip(cond) * 0.2
        
        if mask is not None:
            out_t = out_t * mask
            out_s = out_s * mask
            
        return out_t, out_s
def approx_standard_normal_cdf(x):
    """
    Fast approximation of the standard normal CDF.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two Gaussian distributions.
    Broadcasting is supported for tensor and scalar inputs.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"


    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
def masked_mean_flat(tensor, mask):
    dims = list(range(1, tensor.ndim))
    denom = mask.sum(dim=dims).clamp(min=1.)
    return (tensor * mask).sum(dim=dims) / denom

def mean_flat(tensor):
    """
    Average over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def discretized_gaussian_log_likelihood(z, mean, log_std):
    """
    Compute the log likelihood of a discretized Gaussian.
    """
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    log_probs = -0.5 * (tmp * tmp + 2 * log_std + c)
    assert log_probs.shape == z.shape
    return log_probs

def exists(x):
    """Return whether a value is not None."""
    return x is not None

def default(val, d):
    """Return val when it exists, otherwise return the default."""
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    """Identity function."""
    return t





def normalize_to_neg_one_to_one(img):
    """Map values from [0, 1] to [-1, 1]."""
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    """Map values from [-1, 1] back to [0, 1]."""
    return (t + 1) * 0.5



class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class DiTBlock(nn.Module):
    """
    DiT-style block with AdaLN, self-attention, and an FFN.
    Reference: https://arxiv.org/abs/2212.09748
    

    """
    def __init__(self, dim, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout
        self.rope = RotaryEmbedding(self.head_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )
        

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Self-Attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # FFN
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        
    def _build_attn_mask(self, B, L, padding_mask, device, dtype):
        
        if padding_mask is None:
            return None
        

        causal_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1
        )
        

        # padding_mask: [B, L, 1] -> [B, L]
        padding_mask_squeezed = padding_mask.squeeze(-1)
        

        key_padding = (padding_mask_squeezed == 0).unsqueeze(1).unsqueeze(2)
        

        # causal: [1, 1, L, L], key_padding: [B, 1, 1, L]

        combined = causal_mask.unsqueeze(0).unsqueeze(0) | key_padding
        

        attn_mask = torch.zeros(B, 1, L, L, device=device, dtype=dtype)
        attn_mask.masked_fill_(combined, float('-inf'))
        
        return attn_mask
        
    def forward(self, x, cond, mask=None):
        """
        x: [B, L, D]
        cond: [B, L, D] or [B, 1, D] conditioning features.
        mask: [B, L, 1] padding mask, where 1 is valid and 0 is padding.
        """
        B, L, D = x.shape
        

        modulation = self.adaLN_modulation(cond)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = modulation.chunk(6, dim=-1)
        

        x_norm = self.norm1(x) * (1 + gamma1) + beta1
        

        q = self.q_proj(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        freqs = self.rope(L, x.device)
        q = apply_rotary_pos_emb(q, freqs)
        k = apply_rotary_pos_emb(k, freqs)

        attn_mask = self._build_attn_mask(B, L, mask, x.device, x.dtype)
        

        if attn_mask is not None:

            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=self.dropout_p if self.training else 0.0
            )
        else:

            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0
            )
        
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        attn_out = self.out_proj(attn_out)
        

        x = x + alpha1 * attn_out
        

        x_norm = self.norm2(x) * (1 + gamma2) + beta2
        ffn_out = self.ffn(x_norm)
        

        x = x + alpha2 * ffn_out
        

        if mask is not None:
            x = x * mask
            
        return x


    


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



def extract(a, t, x_shape):
    """Extract values at timestep t and reshape for broadcasting."""
    b, *_ = t.shape
    out = a.gather(-1, t)

    return out.reshape(b, *((1,) * (len(x_shape) - 1)))




def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)




def cosine_beta_schedule(timesteps, s = 0.015):
    """Cosine beta schedule."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion_ST(nn.Module):
    """
    Gaussian diffusion controller for spatio-temporal event prediction.
    """
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l2',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0.,
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.,
        leapfrog_start_ratio=0.6,
        leapfrog_guidance_scale=0.25 ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length
        self.objective = objective
        self.leapfrog_start_ratio = leapfrog_start_ratio
        self.leapfrog_guidance_scale = leapfrog_guidance_scale

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')



        alphas = 1.- betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type


        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
       
        self.ddim_sampling_eta = ddim_sampling_eta


        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)


        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.- alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1.- alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1./ alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1./ alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.- alphas_cumprod_prev) / (1.- alphas_cumprod)

        # above: equal to 1.

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min = posterior_variance[1])))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.- alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1.- alphas_cumprod_prev) * torch.sqrt(alphas) / (1.- alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)



    def _vb_terms_bpd(self, x_start, x_t, t, *, clip_denoised: bool, cond=None, mask=None):
        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        model_mean, _, model_log_variance, pred_xstart, _ = self.p_mean_variance(
            x=x_t, t=t, clip_denoised=clip_denoised, cond=cond, mask=mask
        )
        
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        



        
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, model_mean, 0.5 * model_log_variance)
        
        if mask is not None:
            # mask shape: [Batch, Seq_Len, 1]
            kl = kl * mask
            decoder_nll = decoder_nll * mask



        kl_all = kl.sum(dim=(1, 2)) / np.log(np.e) 
        decoder_nll_all = decoder_nll.sum(dim=(1, 2)) / np.log(np.e)



        kl_temporal = kl[:, :, :1].sum(dim=(1, 2)) / np.log(np.e)
        kl_spatial = kl[:, :, 1:].sum(dim=(1, 2)) / np.log(np.e)

        decoder_nll_temporal = decoder_nll[:, :, :1].sum(dim=(1, 2)) / np.log(np.e)
        decoder_nll_spatial = decoder_nll[:, :, 1:].sum(dim=(1, 2)) / np.log(np.e)



        output = torch.where(t == 0, decoder_nll_all, kl_all)
        output_temporal = torch.where(t == 0, decoder_nll_temporal, kl_temporal)
        output_spatial = torch.where(t == 0, decoder_nll_spatial, kl_spatial)

        return output, output_temporal, output_spatial, pred_xstart



    def predict_start_from_noise(self, x_t, t, noise):

        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.- self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, cond=None, mask=None):
        model_output = self.model(x, t, x_self_cond, cond=cond, mask=mask)
        attn_weight = self.model.get_attn(x, t, x_self_cond,cond=cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), attn_weight

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True, cond=None, Type=None, mask=None):
        preds, attn_weight = self.model_predictions(x, t, x_self_cond, cond=cond, mask=mask)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start, attn_weight 

    @torch.no_grad()
    def sample_with_guidance(self, batch_size=16, cond=None, mask=None, guidance_scale=0.2):
       
        if cond is None: 
            raise ValueError("sample_with_guidance requires cond")
        
        batch_size = cond.shape[0]
        real_seq_len = cond.shape[1]
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = (mask > 0).float()
        

        mean_pred_t = self.model.mean_pred_temporal(cond)  # [B, L, 1]
        mean_pred_s = self.model.mean_pred_spatial(cond)   # [B, L, dim-1]
        mean_pred = torch.cat([mean_pred_t, mean_pred_s], dim=-1)  # [B, L, dim]
        

        mean_pred_clamped = mean_pred.clamp(0, 1)
        mean_pred_normalized = normalize_to_neg_one_to_one(mean_pred_clamped)
        

        feature_dim = self.seq_length
        shape = (batch_size, real_seq_len, feature_dim)
        
        if self.is_ddim_sampling:
            return self.ddim_sample_with_guidance(shape, cond=cond, mask=mask, 
                                                   mean_target=mean_pred_normalized, 
                                                   guidance_scale=guidance_scale)
        else:
            return self.p_sample_loop_with_guidance(shape, cond=cond, mask=mask,
                                                     mean_target=mean_pred_normalized, 
                                                     guidance_scale=guidance_scale)
    @torch.no_grad()
    def p_sample_loop_with_guidance(self, shape, cond, mask=None, mean_target=None, guidance_scale=0.2):
        """DDPM sampling loop with mean guidance."""
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)

        if mask is not None: 
            mask = mask.to(device=img.device, dtype=img.dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            img = img * mask + (-1.0) * (1.0 - mask)

        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling with guidance', total=self.num_timesteps, leave=False):
            self_cond = x_start if self.self_condition else None
            img, x_start, _ = self.p_sample(img, t, self_cond, cond=cond, mask=mask)
            

            if mean_target is not None and t > 0:

                progress = t / self.num_timesteps
                current_guidance = guidance_scale * progress
                img = img * (1 - current_guidance) + mean_target * current_guidance

            if mask is not None: 
                img = img * mask + (-1.0) * (1.0 - mask)
                if x_start is not None:
                    x_start = x_start * mask + (-1.0) * (1.0 - mask)

        img = unnormalize_to_zero_to_one(img)
        if mask is not None: 
            img = img * mask
        return img
    
    @torch.no_grad()
    def ddim_sample_with_guidance(self, shape, clip_denoised=True, cond=None, mask=None, mean_target=None, guidance_scale=0.2):
        """DDIM sampling with mean guidance."""
        batch, device = shape[0], self.betas.device
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)

        if mask is not None: 
            mask = mask.to(device=img.device, dtype=img.dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            img = img * mask + (-1.0) * (1.0 - mask)

        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, _ = self.model_predictions(img, time_cond, x_self_cond=self_cond, cond=cond, clip_x_start=True, mask=mask)
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
            else:
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(img)
                img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise


            if mean_target is not None and time_next >= 0:
                progress = time / total_timesteps
                current_guidance = guidance_scale * progress
                img = img * (1 - current_guidance) + mean_target * current_guidance

            if mask is not None:
                img = img * mask + (-1.0) * (1.0 - mask)

        img = unnormalize_to_zero_to_one(img)
        if mask is not None: 
            img = img * mask
        return img
    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, clip_denoised=True, cond=None, mask=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start, attn_weight = self.p_mean_variance(
            x = x, t = batched_times, x_self_cond = x_self_cond,
            clip_denoised = clip_denoised, cond=cond, mask=mask
        )
        noise = torch.randn_like(x) if t > 0 else 0.# no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        
        return pred_img, x_start, attn_weight

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, mask=None):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)


        if mask is not None:
            mask = mask.to(device=img.device, dtype=img.dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            img = img * mask + (-1.0) * (1.0 - mask)

        x_start = None
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc='sampling loop time step',
            total=self.num_timesteps,
            leave=False
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start, _ = self.p_sample(img, t, self_cond, cond=cond, mask=mask)

            if mask is not None:
                img = img * mask + (-1.0) * (1.0 - mask)
                if x_start is not None:
                    x_start = x_start * mask + (-1.0) * (1.0 - mask)

        img = unnormalize_to_zero_to_one(img)
        if mask is not None:
            img = img * mask
        return img
       
    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=True, cond=None, mask=None):
        batch, device = shape[0], self.betas.device
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)

        if mask is not None:
            mask = mask.to(device=img.device, dtype=img.dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            img = img * mask + (-1.0) * (1.0 - mask)

        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, _ = self.model_predictions(img, time_cond, x_self_cond=self_cond, cond=cond, clip_x_start=True, mask=mask)
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
            else:
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(img)
                img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if mask is not None:
                img = img * mask + (-1.0) * (1.0 - mask)

        img = unnormalize_to_zero_to_one(img)
        if mask is not None:
            img = img * mask
        return img

    @torch.no_grad()
    def sample_leapfrog(self, batch_size=16, cond=None, mask=None, 
                        start_ratio=0.5, guidance_scale=0.3):
        if cond is None:  
            raise ValueError("sample_leapfrog requires cond")
        
        batch_size = cond.shape[0]
        real_seq_len = cond.shape[1]
        device = self.betas.device
        
        if mask is not None: 
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = (mask > 0).float().to(device)
        

        mean_pred_t, mean_pred_s = self.model.joint_mean_predictor(cond, mask=mask)
        mean_pred = torch.cat([mean_pred_t, mean_pred_s], dim=-1)
        
        mean_pred_clamped = mean_pred.clamp(0, 1)
        x_prior = normalize_to_neg_one_to_one(mean_pred_clamped)
        
        start_t = int(self.num_timesteps * start_ratio)
        
        noise = torch.randn_like(x_prior)
        if mask is not None:  
            noise = noise * mask
        
        t_tensor = torch.full((batch_size,), start_t, device=device, dtype=torch.long)
        
        sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod, t_tensor, x_prior.shape)
        sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_prior.shape)
        
        img = sqrt_alpha_bar * x_prior + sqrt_one_minus_alpha_bar * noise
        
        if mask is not None:  
            img = img * mask + (-1.0) * (1.0 - mask)
        
        x_start = None
        for t in tqdm(reversed(range(0, start_t)), desc=f'leapfrog sampling', 
                    total=start_t, leave=False):
            self_cond = x_start if self.self_condition else None
            img, x_start, _ = self.p_sample(img, t, self_cond, cond=cond, mask=mask)
            
            if t > 0:
                progress = t / start_t
                current_guidance = guidance_scale * progress
                img = img * (1 - current_guidance) + x_prior * current_guidance
            
            if mask is not None: 
                img = img * mask + (-1.0) * (1.0 - mask)
                if x_start is not None:
                    x_start = x_start * mask + (-1.0) * (1.0 - mask)
        
        img = unnormalize_to_zero_to_one(img)
        if mask is not None:  
            img = img * mask
        return img

    @torch.no_grad()
    def sample_leapfrog_ddim(self, batch_size=16, cond=None, mask=None,
                            start_ratio=0.5, guidance_scale=0.3):
        if cond is None: 
            raise ValueError("sample_leapfrog_ddim requires cond")
        
        batch_size = cond.shape[0]
        real_seq_len = cond.shape[1]
        device = self.betas.device
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = (mask > 0).float().to(device)
        

        mean_pred_t, mean_pred_s = self.model.joint_mean_predictor(cond, mask=mask)
        mean_pred = torch.cat([mean_pred_t, mean_pred_s], dim=-1)
        
        mean_pred_clamped = mean_pred.clamp(0, 1)
        x_prior = normalize_to_neg_one_to_one(mean_pred_clamped)
        
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta
        
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        
        start_t = int(total_timesteps * start_ratio)
        time_pairs = [(t1, t2) for t1, t2 in zip(times[:-1], times[1:]) if t1 <= start_t]
        
        if len(time_pairs) == 0:
            img = unnormalize_to_zero_to_one(x_prior)
            if mask is not None: 
                img = img * mask
            return img
        
        noise = torch.randn_like(x_prior)
        if mask is not None: 
            noise = noise * mask
        
        first_t = time_pairs[0][0]
        t_tensor = torch.full((batch_size,), first_t, device=device, dtype=torch.long)
        
        sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod, t_tensor, x_prior.shape)
        sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_prior.shape)
        
        img = sqrt_alpha_bar * x_prior + sqrt_one_minus_alpha_bar * noise
        
        if mask is not None:  
            img = img * mask + (-1.0) * (1.0 - mask)
        
        x_start = None
        
        for time, time_next in tqdm(time_pairs, desc='leapfrog ddim', leave=False):
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            
            preds, _ = self.model_predictions(img, time_cond, x_self_cond=self_cond, 
                                            cond=cond, clip_x_start=True, mask=mask)
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start
            
            if time_next < 0:
                img = x_start
            else:
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(img)
                img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            
            if time_next >= 0:
                progress = time / start_t
                current_guidance = guidance_scale * progress
                img = img * (1 - current_guidance) + x_prior * current_guidance
            
            if mask is not None:
                img = img * mask + (-1.0) * (1.0 - mask)
        
        img = unnormalize_to_zero_to_one(img)
        if mask is not None: 
            img = img * mask
        return img


    @torch.no_grad()
    def sample_progressive_refine(self, batch_size=16, cond=None, mask=None,
                                refine_strength=0.25,
                                guidance_weight=0.1):
        """
        SDEdit-style progressive refinement sampling.
        The path is selected from the global DDIM/DDPM configuration.
        """
        if cond is None:
            raise ValueError("sample_progressive_refine requires cond")
        
        batch_size = cond.shape[0]
        device = self.betas.device
        
        if mask is not None:
            if mask.dim() == 2: mask = mask.unsqueeze(-1)
            mask = (mask > 0).float().to(device)
        

        mean_pred_t, mean_pred_s = self.model.joint_mean_predictor(cond, mask=mask)
        mean_pred = torch.cat([mean_pred_t, mean_pred_s], dim=-1).clamp(0, 1)
        x_coarse = normalize_to_neg_one_to_one(mean_pred)
        


        start_t = int(self.num_timesteps * refine_strength)
        start_t = max(1, min(start_t, self.num_timesteps - 1))
        
        noise = torch.randn_like(x_coarse)
        if mask is not None: noise = noise * mask
    
        if self.is_ddim_sampling:

            total_timesteps = self.num_timesteps
            sampling_timesteps = self.sampling_timesteps
            times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
            times = list(reversed(times.int().tolist()))
            


            time_pairs = [(t1, t2) for t1, t2 in zip(times[:-1], times[1:]) if t1 <= start_t]
            
            if not time_pairs:
                real_start_t = start_t
            else:
                real_start_t = time_pairs[0][0]
        else:
            real_start_t = start_t

        t_tensor = torch.full((batch_size,), real_start_t, device=device, dtype=torch.long)
        x_noisy = self.q_sample(x_start=x_coarse, t=t_tensor, noise=noise, mask=mask)
        

        img = x_noisy
        x_start = None
        
        if self.is_ddim_sampling:

            eta = self.ddim_sampling_eta
            

            for time, time_next in tqdm(time_pairs, desc='refining ddim', leave=False):
                time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
                self_cond = x_start if self.self_condition else None
                
                preds, _ = self.model_predictions(img, time_cond, x_self_cond=self_cond, 
                                                cond=cond, clip_x_start=True, mask=mask)
                pred_noise = preds.pred_noise
                x_start = preds.pred_x_start
                
                if time_next < 0:
                    img = x_start
                else:
                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[time_next]
                    sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()
                    noise = torch.randn_like(img)
                    img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
                

                if time > start_t * 0.5 and time_next >= 0:
                    current_guide = guidance_weight * (time / start_t)
                    img = img * (1 - current_guide) + x_coarse * current_guide
                
                if mask is not None:
                    img = img * mask + (-1.0) * (1.0 - mask)
                    
        else:


            loop_range = tqdm(reversed(range(0, real_start_t)), desc='refining ddpm', 
                            total=real_start_t, leave=False)
            
            for t in loop_range:
                self_cond = x_start if self.self_condition else None
                img, x_start, _ = self.p_sample(img, t, self_cond, cond=cond, mask=mask)
                

                if t > start_t * 0.5: 
                    current_guide = guidance_weight * (t / start_t) 
                    img = img * (1 - current_guide) + x_coarse * current_guide
                
                if mask is not None:
                    img = img * mask + (-1.0) * (1.0 - mask)
                    if x_start is not None:
                        x_start = x_start * mask + (-1.0) * (1.0 - mask)
        

        img = unnormalize_to_zero_to_one(img)
        if mask is not None:
            img = img * mask
            
        return img
    @torch.no_grad()
    def sample_hybrid(self, batch_size=16, cond=None, mask=None,
                    diffusion_weight=0.7, start_ratio=0.5, guidance_scale=0.3):
        """
        Hybrid prediction strategy that combines diffusion sampling and the mean predictor.

        The mean predictor first provides an anchor prediction, then leapfrog
        diffusion sampling refines that anchor. The final output blends the
        diffusion result and the mean prediction.

        Args:
            diffusion_weight: Weight assigned to the diffusion result.
            start_ratio: Leapfrog starting ratio.
            guidance_scale: Guidance strength.
        """
        if cond is None: 
            raise ValueError("sample_hybrid requires cond")
        
        batch_size = cond.shape[0]
        real_seq_len = cond.shape[1]
        device = self.betas.device
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = (mask > 0).float().to(device)
        

        mean_pred_t, mean_pred_s = self.model.joint_mean_predictor(cond, mask=mask)
        mean_pred = torch.cat([mean_pred_t, mean_pred_s], dim=-1)
        mean_pred_clamped = mean_pred.clamp(0, 1)
        

        x_prior = normalize_to_neg_one_to_one(mean_pred_clamped)
        

        start_t = int(self.num_timesteps * start_ratio)
        

        noise = torch.randn_like(x_prior)
        if mask is not None: 
            noise = noise * mask
        
        t_tensor = torch.full((batch_size,), start_t, device=device, dtype=torch.long)
        sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod, t_tensor, x_prior.shape)
        sqrt_one_minus_alpha_bar = extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_prior.shape)
        
        img = sqrt_alpha_bar * x_prior + sqrt_one_minus_alpha_bar * noise
        
        if mask is not None: 
            img = img * mask + (-1.0) * (1.0 - mask)
        

        x_start = None
        if self.is_ddim_sampling:

            total_timesteps = self.num_timesteps
            sampling_timesteps = self.sampling_timesteps
            eta = self.ddim_sampling_eta
            
            times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = [(t1, t2) for t1, t2 in zip(times[:-1], times[1:]) if t1 <= start_t]
            
            for time, time_next in time_pairs: 
                time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
                self_cond = x_start if self.self_condition else None
                
                preds, _ = self.model_predictions(img, time_cond, x_self_cond=self_cond,
                                                cond=cond, clip_x_start=True, mask=mask)
                pred_noise = preds.pred_noise
                x_start = preds.pred_x_start
                
                if time_next < 0:
                    img = x_start
                else:
                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[time_next]
                    sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()
                    noise = torch.randn_like(img)
                    img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
                

                if time_next >= 0:
                    progress = time / start_t
                    current_guidance = guidance_scale * progress
                    img = img * (1 - current_guidance) + x_prior * current_guidance
                
                if mask is not None:
                    img = img * mask + (-1.0) * (1.0 - mask)
        else:

            for t in reversed(range(0, start_t)):
                self_cond = x_start if self.self_condition else None
                img, x_start, _ = self.p_sample(img, t, self_cond, cond=cond, mask=mask)
                
                if t > 0:
                    progress = t / start_t
                    current_guidance = guidance_scale * progress
                    img = img * (1 - current_guidance) + x_prior * current_guidance
                
                if mask is not None:
                    img = img * mask + (-1.0) * (1.0 - mask)
                    if x_start is not None:
                        x_start = x_start * mask + (-1.0) * (1.0 - mask)
        


        diffusion_result = unnormalize_to_zero_to_one(img)
        

        hybrid_result = diffusion_weight * diffusion_result + (1 - diffusion_weight) * mean_pred_clamped
        
        if mask is not None: 
            hybrid_result = hybrid_result * mask
        
        return hybrid_result


    @torch.no_grad()
    def sample_adaptive_hybrid(self, batch_size=16, cond=None, mask=None,
                            uncertainty_threshold=0.3):
        """
        Adaptive hybrid strategy that adjusts the sampling weights from mean-prediction uncertainty.

        Confident mean predictions use more of the mean predictor, while
        uncertain predictions rely more on diffusion sampling.
        """
        if cond is None:
            raise ValueError("sample_adaptive_hybrid requires cond")
        
        batch_size = cond.shape[0]
        device = self.betas.device
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = (mask > 0).float().to(device)
        

        mean_pred_t, mean_pred_s = self.model.joint_mean_predictor(cond, mask=mask)
        mean_pred = torch.cat([mean_pred_t, mean_pred_s], dim=-1)
        mean_pred_clamped = mean_pred.clamp(0, 1)
        

        # uncertainty = 1 - 2 * |pred - 0.5| ∈ [0, 1]
        uncertainty = 1 - 2 * torch.abs(mean_pred_clamped - 0.5)
        

        if mask is not None: 
            uncertainty = uncertainty * mask
            avg_uncertainty = uncertainty.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) * mean_pred.shape[-1])
        else:
            avg_uncertainty = uncertainty.mean(dim=(1, 2))
        

        # diffusion_weight ∈ [0.3, 0.9]
        diffusion_weight = 0.3 + 0.6 * avg_uncertainty.mean().item()
        diffusion_weight = min(0.9, max(0.3, diffusion_weight))
        

        start_ratio = 0.3 + 0.4 * avg_uncertainty.mean().item()
        start_ratio = min(0.7, max(0.3, start_ratio))
        

        return self.sample_hybrid(
            batch_size=batch_size,
            cond=cond,
            mask=mask,
            diffusion_weight=diffusion_weight,
            start_ratio=start_ratio,
            guidance_scale=self.leapfrog_guidance_scale
        )
    @torch.no_grad()
    def sample(self, batch_size=16, cond=None, mask=None, mode='leapfrog'):
        """
        Unified sampling entry point.

        Args:
            mode: Sampling mode.
                - 'leapfrog': Leapfrog sampling by default.
                - 'hybrid': Fixed-weight hybrid sampling.
                - 'adaptive': Adaptive hybrid sampling.
                - 'pure_diffusion': Pure diffusion sampling without mean guidance.
                - 'pure_mean': Mean predictor only.
                - 'refine': Progressive refinement sampling.
        """
        if hasattr(self.model, 'joint_mean_predictor') and self.model.joint_mean_predictor is not None: 
            if mode == 'refine':
            
                return self.sample_progressive_refine(
                    batch_size, cond, mask,
                    refine_strength=self.leapfrog_start_ratio,
                    guidance_weight=self.leapfrog_guidance_scale   
                )
            if mode == 'hybrid':
                return self.sample_hybrid(
                    batch_size, cond, mask,
                    diffusion_weight=0.7,
                    start_ratio=self.leapfrog_start_ratio,
                    guidance_scale=self.leapfrog_guidance_scale
                )
            
            elif mode == 'adaptive':
                return self.sample_adaptive_hybrid(batch_size, cond, mask)
            
            elif mode == 'pure_mean':

                if mask is not None: 
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(-1)
                    mask = (mask > 0).float().to(self.betas.device)
                
                mean_pred_t, mean_pred_s = self.model.joint_mean_predictor(cond, mask=mask)
                mean_pred = torch.cat([mean_pred_t, mean_pred_s], dim=-1)
                result = mean_pred.clamp(0, 1)
                
                if mask is not None:
                    result = result * mask
                return result
            
            elif mode == 'pure_diffusion': 

                if cond is not None:
                    batch_size = cond.shape[0]
                    real_seq_len = cond.shape[1]
                
                if mask is not None: 
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(-1)
                    mask = (mask > 0).float()
                
                shape = (batch_size, real_seq_len, self.seq_length)
                
                if self.is_ddim_sampling: 
                    return self.ddim_sample(shape, cond=cond, mask=mask)
                return self.p_sample_loop(shape, cond=cond, mask=mask)
            
            else:
                if self.is_ddim_sampling:
                    return self.sample_leapfrog_ddim(
                        batch_size, cond, mask,
                        start_ratio=self.leapfrog_start_ratio,
                        guidance_scale=self.leapfrog_guidance_scale
                    )
                else:
                    return self.sample_leapfrog(
                        batch_size, cond, mask,
                        start_ratio=self.leapfrog_start_ratio,
                        guidance_scale=self.leapfrog_guidance_scale
                    )
        

        if cond is not None:
            batch_size = cond.shape[0]
            real_seq_len = cond.shape[1]
        elif mask is not None: 
            batch_size = mask.shape[0]
            real_seq_len = mask.shape[1]
        else:
            raise ValueError("sample() requires cond or mask")
            
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = (mask > 0).float()

        shape = (batch_size, real_seq_len, self.seq_length)

        if self.is_ddim_sampling:
            return self.ddim_sample(shape, cond=cond, mask=mask)
        return self.p_sample_loop(shape, cond=cond, mask=mask)


    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample (self, x_start, t, noise=None,mask=None):
        """
        Forward noising process q_sample.

        Given clean data x_start and timestep t, generate noisy data x_t:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        if mask is not None:
            noise = noise * mask
        result = (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                  extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        if mask is not None:
            result = result * mask + (-1.) * (1 - mask)
        return result
       
    
        

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'Euclid':
            return F.pairwise_distance
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise=None, cond=None, mask=None):
        """
        Compute the training loss for one diffusion step.
        """
        b, c, n = x_start.shape
        if mask is None:
            mask = torch.ones_like(x_start[..., :1])
        else:
            mask = mask.unsqueeze(-1) if mask.dim() == 2 else mask
            mask = mask.type_as(x_start)
        noise = default(noise, lambda: torch.randn_like(x_start))
        noise = noise * mask

        x_t = self.q_sample(x_start = x_start, t = t, noise = noise,mask=mask)


        model_out = self.model(x_t, t, cond=cond, mask=mask)
        

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        
        
        loss_elementwise = self.loss_fn(model_out, target, reduction='none')

        loss_t_ele = loss_elementwise[..., :1] # Shape: [B, L, 1]

        loss_diff_t = (loss_t_ele * mask).sum() / mask.sum().clamp(min=1.)


        loss_s_ele = loss_elementwise[..., 1:]
        


        loss_s_masked = loss_s_ele * mask 
        



        denom_s = mask.sum() * loss_s_ele.shape[-1]
        
        loss_diff_s = loss_s_masked.sum() / denom_s.clamp(min=1.)

        return loss_diff_t, loss_diff_s
   
    def _prior_bpd(self, x_start, mask=None):
        """
        Get the prior KL term for the variational lower-bound, measured in nats.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        

        if mask is not None:
            kl_prior = kl_prior * mask



        prior_all = kl_prior.sum(dim=(1, 2)) / np.log(np.e)
        prior_temporal = kl_prior[:,:,:1].sum(dim=(1, 2)) / np.log(np.e)
        prior_spatial = kl_prior[:,:,1:].sum(dim=(1, 2)) / np.log(np.e)

        return prior_all, prior_temporal, prior_spatial
    

    def NLL_cal(self, img, cond, mask=None, noise=None):
        x_start = normalize_to_neg_one_to_one(img)
        b, c, n, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))
        vb_all, vb_temporal_all, vb_spatial_all = [], [], []
        if mask is not None:
            mask = mask.unsqueeze(-1) if mask.dim() == 2 else mask
            mask = mask.type_as(x_start)
            noise = noise * mask

        for tt in list(range(self.num_timesteps))[::-1]:
            t = torch.tensor([tt]).expand(b).long().to(device)
            x = self.q_sample(x_start=x_start, t=t, noise=noise, mask=mask)
            

            vb, vb_temporal, vb_spatial, _ = self._vb_terms_bpd(
                x_start, x, t, clip_denoised=True, cond=cond, mask=mask
            )
            
            vb_all.append(vb.unsqueeze(dim=1))
            vb_temporal_all.append(vb_temporal.unsqueeze(dim=1))
            vb_spatial_all.append(vb_spatial.unsqueeze(dim=1))


        vb_all_sum = torch.sum(torch.cat(vb_all, dim=-1), dim=-1) # [Batch]
        vb_temporal_sum = torch.sum(torch.cat(vb_temporal_all, dim=-1), dim=-1)
        vb_spatial_sum = torch.sum(torch.cat(vb_spatial_all, dim=-1), dim=-1)



        prior_bpd, prior_t, prior_s = self._prior_bpd(x_start, mask=mask)


        total_bpd = vb_all_sum + prior_bpd
        total_temporal = vb_temporal_sum + prior_t
        total_spatial = vb_spatial_sum + prior_s


        return total_bpd.sum().item(), total_temporal.sum().item(), total_spatial.sum().item()
    def forward(self, img, cond,mask=None, *args, **kwargs):
        """
        Forward pass used during diffusion training.
        """
        mask_tensor = None
        if mask is not None:
            mask_tensor = mask.unsqueeze(-1) if mask.dim() == 2 else mask
            mask_tensor = mask_tensor.type_as(img)
        img = normalize_to_neg_one_to_one(img)
        if mask_tensor is not None:
            img = img * mask_tensor + (-1.) * (1 - mask_tensor)
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        return self.p_losses(img, t, cond=cond, mask=mask_tensor, *args, **kwargs)





class ST_Diffusion(nn.Module):
    def __init__(self, n_steps, dim, num_units=64, self_condition=False, 
                 condition=True, cond_dim=0, interaction_start=4, num_heads=4, **kwargs):
        super(ST_Diffusion, self).__init__()
        self.channels = 1
        self.self_condition = self_condition
        self.condition = condition
        self.cond_dim = cond_dim
        actual_cond_dim = cond_dim * 3
        self.num_heads = num_heads

        self.cond_all = nn.Sequential(
            nn.Linear(cond_dim * 3, num_units), 
            nn.GELU(),
            nn.Linear(num_units, num_units)
        )

        self.input_time_mlp = nn.Sequential(
            nn.Linear(1, num_units),
            nn.GELU(),
            nn.Linear(num_units, num_units),
            nn.GELU(),
            nn.Linear(num_units, num_units)
        )
        
        self.spatial_input_mlp = nn.Sequential(
            nn.Linear(dim - 1, num_units),
            nn.GELU(),
            nn.Linear(num_units, num_units)
        )
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_units),
            nn.Linear(num_units, num_units * 4),
            nn.GELU(),
            nn.Linear(num_units * 4, num_units)
        )
        
        self.num_blocks = 6
        self.interaction_start = interaction_start
        
        self.temporal_blocks = nn.ModuleList([
            DiTBlock(num_units, num_heads=num_heads, mlp_ratio=4, dropout=0.1)
            for _ in range(self.num_blocks)
        ])

        self.spatial_blocks = nn.ModuleList([
            DiTBlock(num_units, num_heads=num_heads, mlp_ratio=4, dropout=0.1)
            for _ in range(self.num_blocks)
        ])
        num_interaction_layers = self.num_blocks - self.interaction_start
        self.t2s_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(num_units * 2, num_units), nn.Sigmoid())
            for _ in range(num_interaction_layers)
        ])
        self.s2t_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(num_units * 2, num_units), nn.Sigmoid())
            for _ in range(num_interaction_layers)
        ])
        
       
        self.interaction_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_units * 2, num_units),
                nn.Sigmoid()
            )
            for _ in range(num_interaction_layers)
        ])
        
        self.temporal_refine = nn.Sequential(
            nn.Linear(num_units, num_units * 2),
            nn.GELU(),
            nn.Linear(num_units * 2, num_units),
        )

        self.alpha_t = nn.Sequential(
            nn.Linear(num_units * 3, num_units),
            nn.Sigmoid()
        )
        
        self.alpha_s = nn.Sequential(
            nn.Linear(num_units * 3, num_units),
            nn.Sigmoid()
        )

        self.temporal_output_proj = nn.Linear(num_units, 1)
        self.spatial_output_proj = nn.Linear(num_units, dim - 1)
        
        nn.init.zeros_(self.temporal_output_proj.weight)
        nn.init.zeros_(self.temporal_output_proj.bias)
        nn.init.zeros_(self.spatial_output_proj.weight)
        nn.init.zeros_(self.spatial_output_proj.bias)
        
     
        self.joint_mean_predictor = ImprovedJointMeanPredictor(
            cond_dim=actual_cond_dim,
            hidden_dim=num_units,
            spatial_dim=dim - 1,
            num_heads=4
        )
        
        
        self.mean_pred_temporal = None
        self.mean_pred_spatial = None
        
        self._last_temporal_aux = None
        self._last_spatial_aux = None
        
    def forward(self, x, t, x_self_cond=None, cond=None, mask=None):
        x_temporal = x[: , :, :1]
        x_spatial = x[:, :, 1:]
      
        cond_global = self.cond_all(cond) 
        t_emb = self.time_mlp(t).unsqueeze(1)
        modulation = cond_global + t_emb
        
        h_t = self.input_time_mlp(x_temporal) 
        h_s = self.spatial_input_mlp(x_spatial)
    
        h_t = h_t + modulation
        h_s = h_s + modulation

        interaction_idx = 0
        for i in range(self.num_blocks):
            h_t = self.temporal_blocks[i](h_t, cond=modulation, mask=mask)
            h_s = self.spatial_blocks[i](h_s, cond=modulation, mask=mask)
            
            if i >= self.interaction_start:


                h_t_orig = h_t
                h_s_orig = h_s
                
                gate_t2s = self.t2s_gates[interaction_idx](torch.cat([h_s_orig, h_t_orig], dim=-1))
                gate_s2t = self.s2t_gates[interaction_idx](torch.cat([h_t_orig, h_s_orig], dim=-1))

                h_t = h_t_orig + gate_s2t * h_s_orig
                h_s = h_s_orig + gate_t2s * h_t_orig
                interaction_idx += 1
        
        h_t = h_t + self.temporal_refine(h_t) * 0.1

        input_gate_t = torch.cat([h_t, h_s, modulation], dim=-1) 
        gate_t = self.alpha_t(input_gate_t) 
        
        input_gate_s = torch.cat([h_s, h_t, modulation], dim=-1)
        gate_s = self.alpha_s(input_gate_s)
        
        h_t_final = h_t + h_s * gate_t
        h_s_final = h_s + h_t * gate_s
        

        out_t = self.temporal_output_proj(h_t_final)
        out_s = self.spatial_output_proj(h_s_final)
        
        if self.training:

            self._last_temporal_aux, self._last_spatial_aux = self.joint_mean_predictor(
                cond=cond.detach(),
                mask=mask
            )
        else:

            self._last_temporal_aux, self._last_spatial_aux = self.joint_mean_predictor(
                cond=cond,
                mask=mask
            )
        
        return torch.cat([out_t, out_s], dim=-1)
    
    def get_attn(self, x, t, x_self_cond=None, cond=None):
        return None
    
class Model_all(nn.Module):
    """Wrapper that combines the transformer encoder and diffusion model."""
    def __init__(self, transformer, diffusion):
        super(Model_all, self).__init__()
        self.transformer = transformer
        self.diffusion = diffusion
