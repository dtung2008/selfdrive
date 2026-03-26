"""Attention-based world model.

Each vehicle (ego + neighbours) is a token. Self-attention captures
inter-vehicle interactions. Given current state + ego action → predicts
next state for all vehicles.

Architecture:
  1. Per-vehicle embedding (linear projection of features).
  2. Action embedding concatenated to the ego token.
  3. Transformer encoder (self-attention layers).
  4. Per-vehicle prediction head → next-state delta.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VehicleEmbedding(nn.Module):
    """Embed a vehicle's features into a token vector."""

    def __init__(self, feature_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, feature_dim) -> (batch, embed_dim)"""
        return self.proj(x)


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, embed_dim)
        mask: (batch, seq_len) — True for valid tokens, False for padding
        Returns: (batch, seq_len, embed_dim)
        """
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, S, S)

        if mask is not None:
            # mask: (B, S) -> (B, 1, 1, S) for broadcasting
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            attn = attn.masked_fill(~attn_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # handle all-masked rows

        out = torch.matmul(attn, v)  # (B, H, S, hd)
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int = None):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class AttentionWorldModel(nn.Module):
    """Predicts next observation given current observation + action.

    Input representation:
      - Ego token:     [ego_speed, ego_lane, ego_x, action_onehot]
      - Neighbor i:    [rel_x, rel_lane, rel_speed, exists]

    Output:
      - Predicted next observation (same format as input obs).
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, max_vehicles: int = 7,
                 num_actions: int = 9):
        super().__init__()
        self.max_vehicles = max_vehicles
        self.num_actions = num_actions
        self.embed_dim = embed_dim

        # Ego features: speed, lane, x + action onehot
        ego_feat_dim = 3 + num_actions
        # Neighbor features: rel_x, rel_lane, rel_speed, exists
        neighbor_feat_dim = 4

        self.ego_embed = VehicleEmbedding(ego_feat_dim, embed_dim)
        self.neighbor_embed = VehicleEmbedding(neighbor_feat_dim, embed_dim)

        # Learned position/type embedding
        self.type_embed = nn.Embedding(2, embed_dim)  # 0=ego, 1=neighbor

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Prediction heads
        # Ego head: predict (delta_speed, delta_lane, delta_x)
        self.ego_head = nn.Linear(embed_dim, 3)
        # Neighbor head: predict (delta_rel_x, delta_rel_lane, delta_rel_speed)
        self.neighbor_head = nn.Linear(embed_dim, 3)

    def forward(self, obs: torch.Tensor,
                action_idx: torch.Tensor) -> torch.Tensor:
        """Predict full next observation (legacy interface).

        Args:
            obs: (batch, obs_dim) — flat observation vector
            action_idx: (batch,) — integer action indices

        Returns:
            pred_next_obs: (batch, obs_dim) — predicted next observation
        """
        B = obs.shape[0]
        device = obs.device

        ego_feats, neighbor_feats, tokens, mask = self._encode(obs, action_idx)
        k = neighbor_feats.shape[1]

        # Transformer
        x = tokens
        for layer in self.layers:
            x = layer(x, mask)

        # Predictions
        ego_delta = self.ego_head(x[:, 0, :])
        neighbor_delta = self.neighbor_head(x[:, 1:, :])

        # Build predicted next obs (residual prediction)
        pred_ego = ego_feats + ego_delta
        pred_neighbors = neighbor_feats.clone()
        pred_neighbors[:, :, :3] = neighbor_feats[:, :, :3] + neighbor_delta

        pred_obs = torch.cat([
            pred_ego,
            pred_neighbors.reshape(B, -1),
        ], dim=-1)

        return pred_obs

    def forward_npc_deltas(self, obs: torch.Tensor,
                           action_idx: torch.Tensor) -> torch.Tensor:
        """Predict only NPC absolute deltas (delta_abs_x, delta_abs_speed per neighbor).

        Used by the hybrid planner: ego state comes from kinematics,
        NPC state changes come from this method.

        The model predicts how each NPC's absolute position and speed
        change, independent of ego movement.

        Args:
            obs: (batch, obs_dim) — flat observation vector (can be in any space)
            action_idx: (batch,) — integer action indices

        Returns:
            npc_abs_deltas: (batch, k, 2) — (delta_abs_x, delta_abs_speed) per neighbor
        """
        B = obs.shape[0]

        ego_feats, neighbor_feats, tokens, mask = self._encode(obs, action_idx)
        k = neighbor_feats.shape[1]

        x = tokens
        for layer in self.layers:
            x = layer(x, mask)

        # Only use neighbor head output
        neighbor_delta = self.neighbor_head(x[:, 1:, :])  # (B, k, 3)
        # Return delta_rel_x and delta_rel_speed (indices 0 and 2)
        # These approximate absolute NPC deltas when ego action effect
        # is removed by the planner
        return neighbor_delta  # (B, k, 3): delta_rel_x, delta_rel_lane, delta_rel_speed

    def _encode(self, obs, action_idx):
        """Shared encoding: parse obs, build tokens and mask."""
        B = obs.shape[0]
        device = obs.device

        ego_feats = obs[:, :3]
        k = (obs.shape[1] - 3) // 4
        neighbor_feats = obs[:, 3:].reshape(B, k, 4)

        action_oh = F.one_hot(action_idx.long(),
                              self.num_actions).float()

        ego_input = torch.cat([ego_feats, action_oh], dim=-1)
        ego_tok = self.ego_embed(ego_input)
        ego_tok = ego_tok + self.type_embed(
            torch.zeros(B, dtype=torch.long, device=device))

        n_tok = self.neighbor_embed(
            neighbor_feats.reshape(B * k, 4)).reshape(B, k, -1)
        n_tok = n_tok + self.type_embed(
            torch.ones(B, dtype=torch.long, device=device)).unsqueeze(1)

        tokens = torch.cat([ego_tok.unsqueeze(1), n_tok], dim=1)

        neighbor_mask = neighbor_feats[:, :, 3] > 0.5
        ego_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        mask = torch.cat([ego_mask, neighbor_mask], dim=1)

        return ego_feats, neighbor_feats, tokens, mask
