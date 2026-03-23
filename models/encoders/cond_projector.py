import torch
import torch.nn as nn
import torch.nn.functional as F

class TextProjectorMVarMScaleMStep(nn.Module):
    def __init__(self, n_var, n_scale, n_steps, n_stages, dim_in=128, dim_out=128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_var = n_var
        self.seg_size = n_steps // n_stages + 1
        self.var_emb = nn.Parameter(torch.zeros((1, n_var, dim_in)))
        self.scale_emb = nn.Parameter(torch.zeros((1, n_scale, dim_in)))
        self.step_emb = nn.Parameter(torch.zeros((1, n_stages, dim_in)))
        var_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.var_cross_attn = nn.TransformerDecoder(var_cross_attn_layer, num_layers=2)
        scale_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.scale_cross_attn = nn.TransformerDecoder(scale_cross_attn_layer, num_layers=2)
        step_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.step_cross_attn = nn.TransformerDecoder(step_cross_attn_layer, num_layers=2)
        self.proj_out = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, attr, diffusion_step):
        # attr.shape == (batch_size, seq_len, dim)
        B = attr.shape[0]
        var_emb = self.var_emb.expand([B,-1,-1]) # (B, n_var, dim_in)
        mvar_attr = self.var_cross_attn(tgt=var_emb, memory=attr) # (B, n_var, dim_in)
        mvar_attr = mvar_attr[:,:,None,:] # (B, n_var, 1, dim_in)

        scale_emb = self.scale_emb.expand([B,-1,-1])
        mscale_attr = self.scale_cross_attn(tgt=scale_emb, memory=attr)
        mscale_attr = mscale_attr[:,None,:,:].expand([-1,self.n_var,-1,-1]) # (B, 1, n_scale, dim_in)

        step_emb = self.step_emb.expand([B,-1,-1])
        mstep_attr = self.step_cross_attn(tgt=step_emb, memory=attr)
        indices = diffusion_step // self.seg_size
        indices = indices[:,None,None]
        mstep_attr = torch.gather(mstep_attr, dim=1, index=indices.expand([-1, -1, mstep_attr.shape[-1]]))
        mstep_attr = mstep_attr[:,None,:,:].expand([-1, self.n_var, -1, -1])

        mix_attr = mvar_attr + mscale_attr + mstep_attr
        out = self.proj_out(mix_attr)
        return out


class AttrProjectorAvg(nn.Module):
    def __init__(self, dim_in=128, dim_hid=128, dim_out=128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.proj_out = nn.Linear(self.dim_hid, self.dim_out)

    def forward(self, attr):
        # input project
        B = attr.shape[0]
        h = torch.mean(attr, dim=1, keepdim=True)  # (B,1,d)
        h = h[:,None,:,:] # (B,1,1,d)
        # out project
        out = self.proj_out(h)
        return out





class QwenProjector(nn.Module):
    def __init__(self, n_var, n_scale, n_steps, n_stages, dim_in=128, dim_out=128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_var = n_var
        self.seg_size = n_steps // n_stages + 1
        self.var_emb = nn.Parameter(torch.zeros((1, n_var, dim_in)))
        self.scale_emb = nn.Parameter(torch.zeros((1, n_scale, dim_in)))
        self.step_emb = nn.Parameter(torch.zeros((1, n_stages, dim_in)))
        var_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.var_cross_attn = nn.TransformerDecoder(var_cross_attn_layer, num_layers=2)
        scale_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.scale_cross_attn = nn.TransformerDecoder(scale_cross_attn_layer, num_layers=2)
        step_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.step_cross_attn = nn.TransformerDecoder(step_cross_attn_layer, num_layers=2)
        self.proj_out = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, attr, diffusion_step):
        # attr.shape = (batch_size, n_var, n_seg, seq_len, dim)
        B, n_vars, n_segments, seq_len, dim_attr = attr.shape

        var_emb = self.var_emb.expand([B,-1,-1]).view(B*n_vars, 1,  -1) # (B*n_var, 1, dim_in)

        mvar_attr = self.var_cross_attn(tgt=var_emb, memory=attr.view(B*n_vars, n_segments*seq_len, dim_attr)) # (B*n_var, 1, dim_in)
        mvar_attr = mvar_attr.view(B, n_vars, 1, -1)# (B, n_var, 1, dim_in)

        scale_emb = self.scale_emb.expand([B,-1,-1])#(B, n_scale, dim_in)
        mscale_attr = self.scale_cross_attn(tgt=scale_emb, memory=attr.reshape(B, -1, dim_attr))
        mscale_attr = mscale_attr[:,None,:,:].expand([-1,self.n_var,-1,-1])#(B, n_vars,n_scale, dim_in)

        step_emb = self.step_emb.expand([B,-1,-1])#()
        mstep_attr = self.step_cross_attn(tgt=step_emb, memory=attr.reshape(B, -1, dim_attr))
        indices = diffusion_step // self.seg_size
        indices = indices[:,None,None]
        mstep_attr = torch.gather(mstep_attr, dim=1, index=indices.expand([-1, -1, mstep_attr.shape[-1]]))
        mstep_attr = mstep_attr[:,None,:,:].expand([-1, self.n_var, -1, -1])

        mix_attr = mvar_attr + mscale_attr + mstep_attr
        out = self.proj_out(mix_attr)
        return out




class QwenV3Projector(nn.Module):
    def __init__(self, n_steps, n_stages, dim_in=128, dim_out=128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.seg_size = n_steps // n_stages + 1
        self.step_emb = nn.Parameter(torch.zeros((1, n_stages, dim_in)))
        step_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.step_cross_attn = nn.TransformerDecoder(step_cross_attn_layer, num_layers=2)
        self.proj_out = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, attr, diffusion_step):
        B, n_vars, n_segments, dim_attr = attr.shape

        step_emb = self.step_emb.expand([B,-1,-1])#(B, n_stages, dim_in)
        mstep_attr = self.step_cross_attn(tgt=step_emb, memory=attr.reshape(B, -1, dim_attr))
        indices = diffusion_step // self.seg_size
        indices = indices[:,None,None]
        mstep_attr = torch.gather(mstep_attr, dim=1, index=indices.expand([-1, -1, mstep_attr.shape[-1]]))
        mstep_attr = mstep_attr[:,None,:,:].expand([-1, n_vars, -1, -1])

        mix_attr = attr+ mstep_attr
        out = self.proj_out(mix_attr)
        return out
