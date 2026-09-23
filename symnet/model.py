from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.patch_embed import PatchEmbed
from timm.layers.pos_embed_sincos import build_sincos2d_pos_embed
from timm.models.vision_transformer import Block

from .layers import CrossAttentionBlock, SkipNet, convert_layer_norm_to_dynamic_tanh


class SymNet(L.LightningModule):
    """Joint reconstruction/localization model with checkpoint-compatible layer names."""

    def __init__(
        self,
        img_size=256,
        patch_size=8,
        in_chans=3,
        embed_dim=192,
        depth=6,
        num_heads=16,
        mlp_ratio=4.0,
        use_dynamic_tanh=True,
        skipnet_learning_rate=5e-4,
        other_learning_rate=1e-3,
        optimizer_name="AdamW",
        weight_decay=0.01,
        radio_map_loss_weight=1.0,
        localization_loss_weight=1.0,
    ):
        super().__init__()
        if in_chans != 3:
            raise ValueError("SymNet expects sampled signal, sampling/building, and DNB channels")
        if optimizer_name not in {"Adam", "AdamW"}:
            raise ValueError("optimizer_name must be Adam or AdamW")
        if min(radio_map_loss_weight, localization_loss_weight) < 0 or (
            radio_map_loss_weight + localization_loss_weight <= 0
        ):
            raise ValueError("Loss weights must be nonnegative, with at least one positive")
        self.save_hyperparameters()
        self.patch_embed1 = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed2 = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        block_args = dict(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
        )
        self.encoder_blocks1 = nn.Sequential(*(Block(**block_args) for _ in range(depth)))
        self.encoder_blocks2 = nn.Sequential(*(Block(**block_args) for _ in range(depth)))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.predloc = nn.Linear(embed_dim, patch_size**2)
        self.predhtm = nn.Linear(embed_dim, patch_size**2)
        self.cross_attn = CrossAttentionBlock(embed_dim, num_heads)
        self.encoder_blocks_h2 = nn.Sequential(*(Block(**block_args) for _ in range(depth)))
        self.encoder_blocks_c2 = nn.Sequential(*(Block(**block_args) for _ in range(depth)))
        self.normh3 = nn.LayerNorm(embed_dim)
        self.normc3 = nn.LayerNorm(embed_dim)
        self.skipnet = SkipNet(in_channels=3)
        self.skip3 = SkipNet(in_channels=3)
        self.skip4 = SkipNet(in_channels=3)
        self._initialize_weights()
        if use_dynamic_tanh:
            convert_layer_norm_to_dynamic_tanh(self)

    def _initialize_weights(self):
        for patch_embed in (self.patch_embed1, self.patch_embed2):
            weight = patch_embed.proj.weight.data
            nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _pos_embed(self, x):
        position = build_sincos2d_pos_embed(
            self.patch_embed1.grid_size, dim=x.shape[-1], device=x.device
        )
        return x + position

    def _unpatchify(self, x):
        patch = self.patch_embed1.patch_size[0]
        height = width = int(x.shape[1] ** 0.5)
        if height * width != x.shape[1]:
            raise ValueError("Token count is not a square patch grid")
        x = x.reshape(x.shape[0], height, width, patch, patch, 1)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], 1, height * patch, width * patch)

    def forward(self, inputs):
        """Return (localization_heatmap, radio_map), both shaped [B, 1, H, W]."""
        refined = inputs.clone()
        refined[:, 0] = self.skipnet(inputs).squeeze(1)
        building = refined[:, 1:2]

        branch_h = self.norm1(self.encoder_blocks1(self._pos_embed(self.patch_embed1(refined))))
        branch_c = self.norm2(self.encoder_blocks2(self._pos_embed(self.patch_embed2(refined))))
        fused_h = self.cross_attn(branch_h, branch_c)
        fused_c = self.cross_attn(branch_c, branch_h)
        hidden_h = self.normh3(self.encoder_blocks_h2(fused_h))
        hidden_c = self.normc3(self.encoder_blocks_c2(fused_c))
        coarse_radio = self._unpatchify(self.predhtm(hidden_h))
        coarse_localization = self._unpatchify(self.predloc(hidden_c))
        decoder_input = torch.cat((coarse_radio, coarse_localization, building), dim=1)
        localization = self.skip3(decoder_input)
        radio_map = self.skip4(decoder_input)
        return localization, radio_map

    def training_step(self, batch, batch_idx):
        localization, radio_map = self(batch["input"].float())
        radio_loss = F.mse_loss(radio_map.squeeze(1), batch["radio_map"].float())
        localization_loss = F.mse_loss(
            localization.squeeze(1), batch["localization_heatmap"].float()
        )
        loss = (self.hparams.radio_map_loss_weight * radio_loss
                + self.hparams.localization_loss_weight * localization_loss)
        self.log_dict(
            {
                "train_loss": torch.sqrt(loss),
                "train_radio_rmse": torch.sqrt(radio_loss),
                "train_localization_rmse": torch.sqrt(localization_loss),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input"].shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        localization, radio_map = self(batch["input"].float())
        radio_loss = F.mse_loss(radio_map.squeeze(1), batch["radio_map"].float())
        localization_loss = F.mse_loss(
            localization.squeeze(1), batch["localization_heatmap"].float()
        )
        loss = (self.hparams.radio_map_loss_weight * radio_loss
                + self.hparams.localization_loss_weight * localization_loss)
        predicted_index = localization.flatten(1).argmax(1)
        target_index = batch["localization_heatmap"].flatten(1).argmax(1)
        width = localization.shape[-1]
        row_error = (predicted_index // width - target_index // width).float()
        col_error = (predicted_index % width - target_index % width).float()
        localization_error = torch.sqrt(row_error.square() + col_error.square()).mean()
        self.log_dict(
            {"val_loss": torch.sqrt(loss), "val_localization_error": localization_error},
            on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch["input"].shape[0],
        )
        return loss

    def configure_optimizers(self):
        skip_parameters = list(self.skipnet.parameters()) + list(self.skip3.parameters()) + list(
            self.skip4.parameters()
        )
        skip_ids = {id(parameter) for parameter in skip_parameters}
        other_parameters = [parameter for parameter in self.parameters() if id(parameter) not in skip_ids]
        optimizer = torch.optim.AdamW if self.hparams.optimizer_name == "AdamW" else torch.optim.Adam
        return optimizer(
            [
                {"params": skip_parameters, "lr": self.hparams.skipnet_learning_rate},
                {"params": other_parameters, "lr": self.hparams.other_learning_rate},
            ],
            weight_decay=self.hparams.weight_decay,
        )


def load_symnet_checkpoint(path, map_location="cpu") -> SymNet:
    """Strictly load the portable release checkpoint without importing legacy code."""
    checkpoint: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=True)
    hyperparameters = checkpoint.get("hyper_parameters", {})
    allowed = {
        "img_size", "patch_size", "in_chans", "embed_dim", "depth", "num_heads", "mlp_ratio",
        "use_dynamic_tanh", "skipnet_learning_rate", "other_learning_rate", "optimizer_name",
        "weight_decay",
        "radio_map_loss_weight", "localization_loss_weight",
    }
    model_args = {key: value for key, value in hyperparameters.items() if key in allowed}
    model = SymNet(**model_args)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model
