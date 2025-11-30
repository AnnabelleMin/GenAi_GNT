import numpy as np
import torch
import torch.nn as nn


class RotaryPositionalEmbeddings(nn.Module):
    """Computes rotary embeddings following the RoFormer formulation."""

    def __init__(self, d: int, base: int = 10000):
        super().__init__()
        if d % 2 != 0:
            raise ValueError("RoPE requires an even dimension")
        self.d = d
        self.base = base
        #compute [theta1....theta d/2]
        theta = base ** (-2 * torch.arange(0, d // 2).float() / d)
        self.register_buffer("theta", theta)
        self.cache = {}

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len not in self.cache:
            positions = torch.arange(seq_len, device=device).float()
            matrix = torch.einsum("m,d->md", positions, self.theta)
            cos_C = torch.cos(matrix)
            sin_C = torch.sin(matrix)
            cos_C = torch.cat([cos_C, cos_C], dim=-1)
            sin_C = torch.cat([sin_C, sin_C], dim=-1)
            self.cache[seq_len] = (cos_C, sin_C)
        cos_C, sin_C = self.cache[seq_len]
        cos_C=cos_C.to(device=device, dtype=dtype)
        sin_C=sin_C.to(device=device, dtype=dtype)
        return cos_C, sin_C

    def forward(self, Y: torch.Tensor, positions: torch.Tensor = None):
        b, h, t, d = Y.shape
        if positions is None:
            cos_C, sin_C = self._build_cache(t, Y.device, Y.dtype)
            cos_C = cos_C.view(1, 1, t, d)
            sin_C = sin_C.view(1, 1, t, d)
        else:
            positions = positions.to(device=Y.device, dtype=Y.dtype)
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)
            freqs = positions[..., None] * self.theta.to(Y.device, Y.dtype)
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
            cos_C = torch.cat([cos, cos], dim=-1)[:, None, :, :]
            sin_C = torch.cat([sin, sin], dim=-1)[:, None, :, :]
        x1 = Y[..., : d // 2]
        x2 = Y[..., d // 2 :]
        rotated_x1 = x1 * cos_C[..., : d // 2] - x2 * sin_C[..., : d // 2]
        rotated_x2 = x2 * cos_C[..., d // 2 :] + x1 * sin_C[..., d // 2 :]
        return torch.cat([rotated_x1, rotated_x2], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


# TESTING FUNCTION FOR ROPE
def test_rope_detailed():
    d = 64
    seq_len = 5
    rope = RotaryPositionalEmbeddings(d)
    
    print("Testing RoPE implementation...")
    
    # basic testing
    x = torch.randn(1, 2, seq_len, d)
    out = rope(x)
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {out.shape}")
    
    # sensitivity testing
    pos1 = torch.arange(seq_len)
    pos2 = torch.arange(seq_len).flip(0)
    
    out1 = rope(x, pos1)
    out2 = rope(x, pos2)
    
    diff = torch.abs(out1 - out2).mean()
    print(f"✓ Position sensitivity test - mean difference: {diff.item():.6f}")
    
    if diff > 1e-6:
        print("✓ RoPE is position-sensitive (good!)")
    else:
        print("✗ RoPE might not be position-sensitive")
    
    # same position consistency testing
    out3 = rope(x, pos1)
    same_pos_diff = torch.abs(out1 - out3).mean()
    print(f"✓ Same position consistency - mean difference: {same_pos_diff.item():.6f}")
    
    if same_pos_diff < 1e-6:
        print("✓ Same positions produce same output (good!)")
    else:
        print("✗ Same positions produce different output")
    
    # range testing
    print(f"✓ Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")

    print("All tests completed!")


# Subtraction-based efficient attention
class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate):
        super(Attention2D, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(k)

        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        return x


# View Transformer
class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate):
        super(Transformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention2D(dim, attn_dp_rate)

    def forward(self, q, k, pos, mask=None):
        residue = q
        x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        return x


# attention module for self attention.
# contains several adaptations to incorportate positional information (NOT IN PAPER)
#   - qk (default) -> only (q.k) attention.
#   - pos -> replace (q.k) attention with position attention.
#   - gate -> weighted addition of  (q.k) attention and position attention.
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dp_rate, attn_mode="qk", pos_dim=None, rotary_emb=None):
        super(Attention, self).__init__()
        if attn_mode in ["qk", "gate"]:
            self.q_fc = nn.Linear(dim, dim, bias=False)
            self.k_fc = nn.Linear(dim, dim, bias=False)
        if attn_mode in ["pos", "gate"]:
            self.pos_fc = nn.Sequential(
                nn.Linear(pos_dim, pos_dim), nn.ReLU(), nn.Linear(pos_dim, dim // 8)
            )
            self.head_fc = nn.Linear(dim // 8, n_heads)
        if attn_mode == "gate":
            self.gate = nn.Parameter(torch.ones(n_heads))
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.n_heads = n_heads
        self.attn_mode = attn_mode
        #added rotary emb for q and k
        self.rotary_emb = rotary_emb

    def forward(self, x, pos=None, rope_pos=None, ret_attn=False):
        if self.attn_mode in ["qk", "gate"]:
            q = self.q_fc(x)
            q = q.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
            k = self.k_fc(x)
            k = k.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
            # apply rotary emb for q and k
            if self.rotary_emb is not None:
                q = self.rotary_emb(q, rope_pos)
                k = self.rotary_emb(k, rope_pos)
        v = self.v_fc(x)
        v = v.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        if self.attn_mode in ["qk", "gate"]:
            attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
            attn = torch.softmax(attn, dim=-1)
        elif self.attn_mode == "pos":
            pos = self.pos_fc(pos)
            attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            attn = torch.softmax(attn, dim=-1)
        if self.attn_mode == "gate":
            pos = self.pos_fc(pos)
            pos_attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            pos_attn = torch.softmax(pos_attn, dim=-1)
            gate = self.gate.view(1, -1, 1, 1)
            attn = (1.0 - torch.sigmoid(gate)) * attn + torch.sigmoid(gate) * pos_attn
            attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.dp(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(x.shape[0], x.shape[1], -1)
        out = self.dp(self.out_fc(out))
        if ret_attn:
            return out, attn
        else:
            return out


# Ray Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        ff_hid_dim,
        ff_dp_rate,
        n_heads,
        attn_dp_rate,
        attn_mode="qk",
        pos_dim=None,
        # set rope here to True to use ROPE
        use_rope=False,
    ):
        super(Transformer, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        rotary_emb = None
        if use_rope:
            head_dim = dim // n_heads
            rotary_emb = RotaryPositionalEmbeddings(head_dim)
        self.attn = Attention(dim, n_heads, attn_dp_rate, attn_mode, pos_dim, rotary_emb)
        self.use_rope = use_rope

    def forward(self, x, pos=None, rope_pos=None, ret_attn=False):
        residue = x
        x = self.attn_norm(x)
        rope_inputs = rope_pos if self.use_rope else None
        x = self.attn(x, pos, rope_inputs, ret_attn)
        if ret_attn:
            x, attn = x
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        if ret_attn:
            return x, attn.mean(dim=1)[:, 0]
        else:
            return x


class GNT(nn.Module):
    def __init__(self, args, in_feat_ch=32, posenc_dim=3, viewenc_dim=3, ret_alpha=False):
        super(GNT, self).__init__()
        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(in_feat_ch + 3, args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        )

        # NOTE: Apologies for the confusing naming scheme, here view_crosstrans refers to the view transformer, while the view_selftrans refers to the ray transformer
        self.view_selftrans = nn.ModuleList([])
        self.view_crosstrans = nn.ModuleList([])
        self.q_fcs = nn.ModuleList([])
        for i in range(args.trans_depth):
            # view transformer
            view_trans = Transformer2D(
                dim=args.netwidth,
                ff_hid_dim=int(args.netwidth * 4),
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_crosstrans.append(view_trans)
            # ray transformer
            ray_trans = Transformer(
                dim=args.netwidth,
                ff_hid_dim=int(args.netwidth * 4),
                n_heads=4,
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
                use_rope=True,
            )
            self.view_selftrans.append(ray_trans)
            # mlp
            if i % 2 == 0:
                q_fc = nn.Sequential(
                    nn.Linear(args.netwidth, args.netwidth),
                    nn.ReLU(),
                    nn.Linear(args.netwidth, args.netwidth),
                )
            else:
                q_fc = nn.Identity()
            self.q_fcs.append(q_fc)

        self.ret_alpha = ret_alpha
        self.norm = nn.LayerNorm(args.netwidth)
        self.rgb_fc = nn.Linear(args.netwidth, 3)
        self.relu = nn.ReLU()

    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d):
        # project rgb features to netwidth
        rgb_feat = self.rgbfeat_fc(rgb_feat)
        # q_init -> maxpool
        q = rgb_feat.max(dim=2)[0]

        # rotary positions along each ray (sequence = samples per ray)
        ray_dir = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
        ray_positions = torch.sum(pts * ray_dir[:, None, :], dim=-1)
        ray_positions = ray_positions - ray_positions[:, :1]

        # transformer modules
        for i, (crosstrans, q_fc, selftrans) in enumerate(
            zip(self.view_crosstrans, self.q_fcs, self.view_selftrans)
        ):
            # view transformer to update q
            q = crosstrans(q, rgb_feat, ray_diff, mask)
            # embed positional information
            if i % 2 == 0:
                q = q_fc(q)
            # ray transformer with rope positional embedding
            q = selftrans(q, rope_pos=ray_positions, ret_attn=self.ret_alpha)
            # 'learned' density
            if self.ret_alpha:
                q, attn = q
        # normalize & rgb
        h = self.norm(q)
        outputs = self.rgb_fc(h.mean(dim=1))
        if self.ret_alpha:
            return torch.cat([outputs, attn], dim=1)
        else:
            return outputs
