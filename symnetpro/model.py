"""SymNetPro network; parameter names match the trained Lightning checkpoints."""

import torch
import torch.nn as nn
import lightning as L
from timm.layers.patch_embed import PatchEmbed
from timm.models.vision_transformer import Block
from timm.layers.pos_embed_sincos import build_sincos2d_pos_embed
from symnet.layers import (
    SkipNet,
    CrossAttentionBlock,
    convert_layer_norm_to_dynamic_tanh,
)


class BiasSelfAttention(nn.Module):

    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        attn = q @ k.transpose(-2, -1) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias[:, None, :, :]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class BiasBlock(nn.Module):

    def __init__(
        self, dim, num_heads=4, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = BiasSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, attn_bias=None):
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(L.LightningModule):

    def __init__(
        self,
        img_size=256,
        patch_size=8,
        in_chans=2,
        embed_dim=192,
        depth=6,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super(VisionTransformer, self).__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.Hp = img_size // patch_size
        self.Wp = img_size // patch_size
        self.patch_embed1 = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed2 = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.encoder1_block1 = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=norm_layer,
        )
        self.encoder2_block1 = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=norm_layer,
        )
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.bias_block1 = BiasBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=norm_layer,
        )
        self.bias_block2 = BiasBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=norm_layer,
        )
        self.attn_bias_scale1 = nn.Parameter(torch.tensor(0.01))
        self.attn_bias_scale2 = nn.Parameter(torch.tensor(0.01))
        self.predloc = nn.Linear(embed_dim, patch_size**2, bias=True)
        self.predhtm = nn.Linear(embed_dim, patch_size**2, bias=True)
        self.cross_attn = CrossAttentionBlock(embed_dim, num_heads)
        self.encoder_blocks_h2 = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth - 1)
            ]
        )
        self.encoder_blocks_h3 = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth - 1)
            ]
        )
        self.normh4 = norm_layer(embed_dim)
        self.normh3 = norm_layer(embed_dim)
        self.normc3 = norm_layer(embed_dim)
        self.normc4 = norm_layer(embed_dim)
        self.encoder_blocks_c2 = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.encoder_blocks_c3 = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.skipnet = SkipNet(in_channels=2)
        self.skip3 = SkipNet(in_channels=3)
        self.skip4 = SkipNet(in_channels=3)
        self.initialize_weights()
        self.save_hyperparameters(ignore=["norm_layer"])
        self.patch_bias_alpha = 2.0
        self.patch_bias_beta = 1.0
        self.patch_bias_temp = 2.0

    def initialize_weights(self):
        patch_embed_weights = self.patch_embed1.proj.weight.data
        nn.init.xavier_uniform_(
            patch_embed_weights.view([patch_embed_weights.shape[0], -1])
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_patch_bias(self, graph_list, device=None, num_tokens=None):
        bias_list = []
        valid_mask_list = []
        eps = 1e-06
        alpha = float(getattr(self, "patch_bias_alpha", 2.0))
        beta = float(getattr(self, "patch_bias_beta", 1.0))
        temp = float(getattr(self, "patch_bias_temp", 2.0))
        for g in graph_list:
            patch_type = g["patch_type"].to(device).long()
            edge_index = g["patch_edge_index"].to(device).long()
            edge_attr = g["patch_edge_attr"].to(device).float()
            N = patch_type.shape[0] if num_tokens is None else num_tokens
            valid_mask = patch_type > 0
            B = torch.zeros((N, N), device=device, dtype=torch.float32)
            if edge_index.numel() > 0:
                src = edge_index[0]
                dst = edge_index[1]
                d = edge_attr[:, 0]
                los = edge_attr[:, 1]
                scaled_score = (-alpha * d - beta * (1.0 - los)) / max(temp, eps)
                row_max = torch.full(
                    (N,), -float("inf"), device=device, dtype=scaled_score.dtype
                )
                row_max.scatter_reduce_(
                    0, dst, scaled_score, reduce="amax", include_self=True
                )
                exp_score = torch.exp(scaled_score - row_max[dst])
                row_sum = torch.zeros((N,), device=device, dtype=scaled_score.dtype)
                row_sum.scatter_add_(0, dst, exp_score)
                row_bias = (
                    scaled_score - row_max[dst] - torch.log(row_sum[dst].clamp_min(eps))
                )
                B[dst, src] = row_bias
            B = B - B.mean(dim=-1, keepdim=True)
            bias_list.append(B.unsqueeze(0))
            valid_mask_list.append(valid_mask.unsqueeze(0))
        bias_mat = torch.cat(bias_list, dim=0)
        valid_mask = torch.cat(valid_mask_list, dim=0)
        return (bias_mat, valid_mask)

    def forward(self, y, graph=None):
        x = y.clone()
        y_skip = self.skipnet(y)
        y_skip = y_skip.squeeze(1)
        x[:, 0] = y_skip
        b = x.clone()
        building = b[:, 1].unsqueeze(1)
        x_clone1 = x.clone()
        x_clone2 = x.clone()
        x1 = self.patch_embed1(x_clone1)
        x1 = self.pos_embed(x1)
        x2 = self.patch_embed2(x_clone2)
        x2 = self.pos_embed(x2)
        if graph is not None:
            bias_mat, valid_mask = self._build_patch_bias(
                graph, device=x1.device, num_tokens=x1.size(1)
            )
            vm = valid_mask.float()
            bias_mat = bias_mat * vm[:, :, None] * vm[:, None, :]
            x1 = self.bias_block1(x1, attn_bias=self.attn_bias_scale1 * bias_mat)
            x2 = self.bias_block2(x2, attn_bias=self.attn_bias_scale2 * bias_mat)
        x1 = self.encoder1_block1(x1)
        x1 = self.norm1(x1)
        x2 = self.encoder2_block1(x2)
        x2 = self.norm2(x2)
        h2 = x1
        c2 = x2
        h3 = self.encoder_blocks_h2(h2)
        c3 = self.encoder_blocks_c2(c2)
        h3 = self.normh3(h3)
        c3 = self.normc3(c3)
        h3 = self.encoder_blocks_h3(h3)
        h3 = self.normh4(h3)
        c3 = self.encoder_blocks_c3(c3)
        c3 = self.normc4(c3)
        h4 = self.cross_attn(h3, c3)
        c4 = self.cross_attn(c3, h3)
        h1 = self.predhtm(h4)
        c1 = self.predloc(c4)
        h1 = self.unpatchify(h1)
        c1 = self.unpatchify(c1)
        input_3 = torch.cat((h1, c1, building), dim=1)
        c1 = self.skip3(input_3)
        h1 = self.skip4(input_3)
        return (x1, x2, c1, h1)

    def pos_embed(self, x):
        return x + build_sincos2d_pos_embed(
            self.patch_embed1.grid_size, dim=x.size(-1), device=x.device
        )

    def unpatchify(self, x):
        p = self.patch_embed1.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return imgs

    def configure_optimizers(self):
        multipler = 0.5
        optimizer = torch.optim.Adam(
            [
                {"params": self.skipnet.parameters(), "lr": 0.0005 * multipler},
                {"params": self.skip3.parameters(), "lr": 0.0005 * multipler},
                {"params": self.skip4.parameters(), "lr": 0.0005 * multipler},
                {
                    "params": [
                        param
                        for name, param in self.named_parameters()
                        if "gnn" in name
                    ],
                    "lr": 0.0005 * multipler,
                },
                {
                    "params": [
                        param
                        for name, param in self.named_parameters()
                        if "skipnet" not in name
                        and "skip" not in name
                        and ("skip3" not in name)
                        and ("gnn" not in name)
                    ],
                    "lr": 0.001 * multipler,
                },
            ]
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        losses = {}
        for branch in ("full", "drop"):
            _, _, loc, signal = self(
                batch[f"data_{branch}"].float(), batch[f"graphs_{branch}"]
            )
            target = batch[f"target_{branch}"].float()
            heatmap = batch[f"atts_{branch}"].float()
            flag = batch[f"{branch}_flag"].float()
            signal_mse = ((signal.squeeze(1) - target) ** 2).mean(dim=(1, 2))
            loc_mse = ((loc.squeeze(1) - heatmap) ** 2).mean(dim=(1, 2))
            losses[branch] = (
                (signal_mse * flag).sum() / flag.numel(),
                (loc_mse * flag).sum() / flag.numel(),
            )
        sig_full, loc_full = losses["full"]
        sig_drop, loc_drop = losses["drop"]
        # The historical full/drop localization weights are intentionally different.
        loss = 0.25 * ((sig_full + (1 / 2.5) * loc_full) + (sig_drop + loc_drop))
        self.log(
            "train_loss",
            loss.sqrt(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(flag),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, loc, signal = self(batch["data_full"].float(), batch["graphs_full"])
        target, heatmap = batch["target_full"].float(), batch["atts_full"].float()
        flag = batch["full_flag"].float()
        sig = ((signal.squeeze(1) - target) ** 2).mean(dim=(1, 2))
        loc = (((loc.squeeze(1) - heatmap) ** 2) * heatmap).mean(dim=(1, 2))
        sig = (sig * flag).sum() / flag.sum().clamp_min(1.0)
        loc = loc.mean()
        for name, value in (
            ("val_loss", sig + loc / 2.5),
            ("val_rmse_signal", sig),
            ("val_rmse_att", loc),
        ):
            self.log(
                name,
                value.sqrt(),
                on_step=False,
                on_epoch=True,
                prog_bar=name == "val_loss",
                sync_dist=True,
                batch_size=len(flag),
            )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["symnetpro_config"] = self.run_config


MODEL_CONFIG = dict(
    img_size=256,
    patch_size=8,
    in_chans=2,
    embed_dim=192,
    depth=6,
    num_heads=16,
    mlp_ratio=4.0,
)


def build_model():
    return convert_layer_norm_to_dynamic_tanh(VisionTransformer(**MODEL_CONFIG))


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("symnetpro_config")
    if config is None:
        raise ValueError(
            "Use the packaged SymNetPro checkpoint, which includes its graph configuration."
        )
    if config["model"] != MODEL_CONFIG:
        raise ValueError("Checkpoint architecture does not match this model.")
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), config
