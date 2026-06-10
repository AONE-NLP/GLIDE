import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import DSTPP.Constants as Constants
from DSTPP.Layers import EncoderLayer
from typing import Sequence, Tuple, Optional
from einops import rearrange, repeat

def get_non_pad_mask(seq):
    """ 
    Return a mask that marks non-padding positions.
    """
    assert seq.dim() == 2



    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ 
    Build the key padding mask used by attention.
    """

    len_q = seq_q.size(1)

    padding_mask = seq_k.eq(Constants.PAD)

    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq, dim=2):
    """ 
    Build the causal mask that hides future positions.
    """
    sz_b, len_s = seq.size()[:2]

    subsequent_mask = torch.triu(
        torch.ones((dim, len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1).permute(1,2,0)

    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1,-1)  # b x ls x ls
    return subsequent_mask



 
       
def temporal_enc(self, time, non_pad_mask):
    """
    Compute sinusoidal temporal encodings.
    """

    result = time.unsqueeze(-1) / self.position_vec

    result[:, :, 0::2] = torch.sin(result[:, :, 0::2])

    result[:, :, 1::2] = torch.cos(result[:, :, 1::2])

    return result * non_pad_mask


def forward(self, event_loc, event_time, non_pad_mask):
    """Encode an event sequence with masked self-attention."""



    slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=self.loc_dim)
    slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc)
    slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)


    slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)


    tem_enc = self.temporal_enc(event_time, non_pad_mask)
    enc_output = self.event_emb(event_loc)
    
    slf_attn_mask = slf_attn_mask[:,:,:,0]


    for enc_layer in self.layer_stack:

        enc_output += tem_enc
        enc_output, _ = enc_layer(
            enc_output,
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask)
    return enc_output


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, device):
        t = torch.arange(max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb
class FourierSpatialEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, num_freqs=64, learnable=True):
        super().__init__()
        self.num_freqs = num_freqs

        self.freqs = nn.Parameter(torch.randn(input_dim, num_freqs), requires_grad=learnable)

        self.proj = nn.Linear(num_freqs * 2, d_model)
        
    def forward(self, x):
        # x: [B, L, input_dim]
        # freqs: [input_dim, num_freqs]
        x_proj = 2 * np.pi * x @ self.freqs # [B, L, num_freqs]
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # [B, L, num_freqs * 2]
        return self.proj(out)



class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, device):
        t = torch.arange(max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())



class RNN_layers(nn.Module):
    """
    Optional recurrent layer stacked on top of transformer features.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):

        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()

        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)

        temp = self.rnn(pack_enc_output)[0]

        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]


        out = self.projection(out)
        return out



class RoPESelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, freqs, mask=None):
        b, n, d = x.shape
        h = self.num_heads


        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=h)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=h)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=h)

        q = apply_rotary_pos_emb(q, freqs)
        k = apply_rotary_pos_emb(k, freqs)


        # ------------------------------------------------


        

        if mask is not None:
            mask_bool = mask.view(b, 1, 1, n)
        else:
            mask_bool = torch.ones((b, 1, 1, n), device=q.device, dtype=torch.bool)
            

        # shape: [1, 1, N, N]
        causal_mask = torch.triu(
            torch.ones((1, 1, n, n), device=q.device, dtype=torch.bool), 
            diagonal=1
        )
        
       
        

        padding_mask_broadcast = mask_bool.expand(-1, -1, n, -1)
        

        attn_mask = torch.zeros((b, 1, n, n), device=q.device, dtype=q.dtype)
        

        attn_mask = attn_mask.masked_fill(padding_mask_broadcast == 0, -float('inf'))

        attn_mask = attn_mask.masked_fill(causal_mask, -float('inf'))

        # 3. Flash Attention
        # ------------------------------------------------
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=0.1 if self.training else 0.0,
            is_causal=False
        )

        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))

class Transformer_ST(nn.Module):

    def __init__(
            self, d_model=64, d_rnn=256,
            n_layers=6, n_head=8, dropout=0.1,spatial_dropout=0.1, device=None, loc_dim=2,
            time_scales=None, k_neighbors=None, connect_backward=True,input_dim=None,num_time_harmonics=4):
        super().__init__()
        self.num_time_harmonics = num_time_harmonics
        self.d_model = d_model
        self.device = device
        time_input_dim = 1 + 1 + num_time_harmonics * 2

        self.temporal_input_proj = nn.Sequential(
            nn.Linear(time_input_dim, d_model), 
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.rope = RotaryEmbedding(d_model // n_head)
        
        self.temporal_layers = nn.ModuleList([])
        for _ in range(n_layers): 
            self.temporal_layers.append(nn.ModuleList([
                RMSNorm(d_model),
                RoPESelfAttention(d_model, n_head, dropout=dropout),
                LayerScale(d_model),
                RMSNorm(d_model),
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                ),
                LayerScale(d_model)
            ]))

        
        self.spatial_encoder = GATEncoder(
             d_model=d_model,
             n_layers=n_layers, 
             n_head=n_head, 
             dropout=spatial_dropout,
             device=device,
             loc_dim=loc_dim,
             input_dim=input_dim
         )
        

      
        self.fusion_norm = RMSNorm(d_model)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)
      #  self.rnn = RNN_layers(d_model * 3, d_rnn)
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 3),
            nn.GELU(),
            nn.Linear(d_model * 3, d_model * 3)
        )
        
    def forward(self, graph_batch):
        node_features = graph_batch.x
        history_time_abs = node_features[:, 0]
        history_time_diff_zscore = node_features[:, 1]
        
        dense_time_abs, non_pad_mask = to_dense_batch(history_time_abs, graph_batch.batch)
        dense_time_diff, _ = to_dense_batch(history_time_diff_zscore, graph_batch.batch)
        


        feat_zscore = dense_time_diff.unsqueeze(-1)  # [B, L, 1]
        

        feat_softplus = F.softplus(dense_time_diff).unsqueeze(-1)  # [B, L, 1]
        

        freqs = torch.arange(1, self.num_time_harmonics + 1, device=dense_time_abs.device).float()
        angles = 2 * np.pi * dense_time_abs.unsqueeze(-1) * freqs
        feat_sin = torch.sin(angles)
        feat_cos = torch.cos(angles)
        
        dense_time_enhanced = torch.cat([
            feat_zscore,    # [B, L, 1]
            feat_softplus,  # [B, L, 1]  
            feat_sin,       # [B, L, num_harmonics]
            feat_cos        # [B, L, num_harmonics]
        ], dim=-1)    

        t_emb = self.temporal_input_proj(dense_time_enhanced)
        
        seq_len = t_emb.shape[1]
        freqs = self.rope(seq_len, t_emb.device)
        freqs = freqs.unsqueeze(0).unsqueeze(0) # [1, 1, SeqLen, HeadDim]
        for norm1, attn, ls1, norm2, ff, ls2 in self.temporal_layers:
            x_norm = norm1(t_emb)
            

            attn_out = attn(x_norm, freqs=freqs, mask=non_pad_mask)
            
            t_emb = t_emb + ls1(attn_out)
            t_emb = t_emb + ls2(ff(norm2(t_emb)))
            
        time_out = t_emb


        spatial_out, _ = self.spatial_encoder(graph_batch)
        
        combined_feat = torch.cat([spatial_out, time_out], dim=-1) # [B, L, 2*D]
        fused_out = self.fusion_proj(combined_feat)
        final_features = torch.cat([spatial_out, time_out, fused_out], dim=-1)
        


        mask_for_rnn = non_pad_mask.unsqueeze(-1)
        

       # enc_output = self.rnn(final_features, mask_for_rnn)   
        enc_output = self.output_mlp(final_features)

        enc_output = enc_output * mask_for_rnn    
        return enc_output, mask_for_rnn

class GATEncoder(nn.Module):
    """GAT encoder with Fourier embeddings and a deeper projection network."""
    def __init__(self, d_model, n_layers, n_head, dropout, device, loc_dim,input_dim=None):
        super().__init__()
        self.d_model = d_model
        self.loc_dim = loc_dim
        self.device = device
        self.n_head = n_head
        self.edge_feat_dim = 3 

        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_feat_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.edge_feat_dim)
        )
        
        if input_dim is None:

             feat_dim = 1 + loc_dim * 2 
        else:
             feat_dim = input_dim
        self.input_emb = nn.Linear(feat_dim, d_model)


        self.fourier_emb = FourierSpatialEmbedding(loc_dim, d_model, num_freqs=64, learnable=True)
        self.loc_proj = nn.Linear(loc_dim, d_model)
        



        head_dim = d_model // n_head
        gat_out_dim = head_dim * n_head
        
        self.gat_layers = nn.ModuleList([
            GATv2Conv(d_model, head_dim, heads=n_head, dropout=dropout, edge_dim=self.edge_feat_dim)
            for _ in range(n_layers)
        ])
        

        
        self.output_projs = nn.ModuleList([nn.Linear(gat_out_dim, d_model) for _ in range(n_layers)])
        

        self.residual_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.layer_norm = RMSNorm(d_model)

    def temporal_enc(self, time, non_pad_mask):
        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, graph_batch):
        node_features = graph_batch.x
        event_time = node_features[:, 0]
        event_loc = node_features[:, 2 : 2 + self.loc_dim]
        
        
        dense_time, non_pad_mask_dense = to_dense_batch(event_time, graph_batch.batch)
        non_pad_mask = non_pad_mask_dense.unsqueeze(-1).float()

        enhanced_edge_attr = self.edge_mlp(graph_batch.edge_attr) + graph_batch.edge_attr
      

        h = self.input_emb(node_features) + self.fourier_emb(event_loc) * 1.5
        for i in range(len(self.gat_layers)):
            h_res = h
            h = self.gat_layers[i](h, graph_batch.edge_index, enhanced_edge_attr)
            h = F.elu(h)
            h = self.output_projs[i](h)      
            h = self.residual_norms[i](h + h_res)
            
        output, _ = to_dense_batch(h, graph_batch.batch)
        
        return self.layer_norm(output), non_pad_mask
