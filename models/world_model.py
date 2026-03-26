"""Attention-based world model for multi-vehicle driving simulation.

Each vehicle (ego + neighbours) is represented as a *token*. Self-attention
layers capture pairwise inter-vehicle interactions (e.g. car-following,
yielding, merging). Given the current state of all vehicles plus the ego
vehicle's chosen action, the model predicts the next state for every vehicle
as a *residual* (delta) on top of the current state.

Architecture overview:
  1. **Per-vehicle embedding** -- a learned linear projection maps each
     vehicle's raw features into a shared embedding space of dimension
     ``embed_dim``.
  2. **Action injection** -- the ego action is one-hot encoded and
     concatenated to the ego vehicle's raw features *before* embedding,
     so the action information is baked into the ego token from the start.
  3. **Type embedding** -- a learned 2-class embedding (ego vs. neighbour)
     is *added* to every token so the transformer can distinguish the
     ego vehicle from surrounding traffic.
  4. **Transformer encoder** -- a stack of standard pre-norm transformer
     blocks (multi-head self-attention + feed-forward) processes the
     token sequence. Padding masks ensure that absent neighbour slots
     do not influence the attention distribution.
  5. **Per-vehicle prediction head** -- separate linear heads for the ego
     token and neighbour tokens project each transformer output back to a
     low-dimensional delta vector (change in speed, lane, position, etc.).
     The final prediction is ``current_state + predicted_delta`` (residual
     connection at the output level).

Typical usage:
    model = AttentionWorldModel(embed_dim=64, num_heads=4, num_layers=2)
    predicted_next_obs = model(obs_batch, action_idx_batch)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VehicleEmbedding(nn.Module):
    """Linear projection that maps raw vehicle features into a token vector.

    This is the simplest possible embedding: a single affine transformation
    (``nn.Linear``) from the feature space to the shared token space. Both
    ego and neighbour vehicles use their own instance of this class because
    their feature spaces differ in dimensionality.

    Attributes:
        proj (nn.Linear): Learnable affine projection.
    """

    def __init__(self, feature_dim: int, embed_dim: int):
        """Initialise the vehicle embedding layer.

        Args:
            feature_dim: Number of raw input features per vehicle
                (e.g. 12 for ego: 3 state features + 9 action one-hot;
                4 for neighbour: rel_x, rel_lane, rel_speed, exists).
            embed_dim: Dimensionality of the output token vector.
        """
        super().__init__()
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw features into the embedding space.

        Args:
            x: Input tensor of shape ``(batch, feature_dim)``.

        Returns:
            Embedded tensor of shape ``(batch, embed_dim)``.
        """
        return self.proj(x)


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention.

    All three projections (query, key, value) are computed with a single
    fused linear layer (``self.qkv``) for efficiency, then split into
    per-head slices. The output of all heads is concatenated and passed
    through a final linear projection (``self.out_proj``).

    Attributes:
        embed_dim (int): Total embedding dimensionality (across all heads).
        num_heads (int): Number of parallel attention heads.
        head_dim (int): Dimensionality of each individual head
            (``embed_dim // num_heads``).
        qkv (nn.Linear): Fused projection that produces Q, K, V in one shot.
            Output size is ``3 * embed_dim``.
        out_proj (nn.Linear): Final linear projection applied after
            concatenating all head outputs.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """Initialise multi-head self-attention.

        Args:
            embed_dim: Total embedding dimension. Must be divisible by
                ``num_heads`` so that each head gets an equal slice.
            num_heads: Number of parallel attention heads.

        Raises:
            AssertionError: If ``embed_dim`` is not divisible by ``num_heads``.
        """
        super().__init__()
        assert embed_dim % num_heads == 0  # heads must divide evenly
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # per-head dimensionality

        # Fused QKV projection: one matrix multiply instead of three,
        # producing query, key, and value concatenated along the last dim.
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)

        # Output projection applied after concatenating all heads.
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: Input token sequence of shape ``(batch, seq_len, embed_dim)``.
            mask: Optional boolean mask of shape ``(batch, seq_len)``.
                ``True`` marks valid (real) tokens; ``False`` marks padding.
                Padded positions receive ``-inf`` attention scores so they
                contribute nothing after softmax.

        Returns:
            Output tensor of shape ``(batch, seq_len, embed_dim)`` with
            the same sequence length as the input.
        """
        B, S, D = x.shape  # batch, sequence length, embedding dim

        # Compute Q, K, V in one fused linear pass, then reshape so that
        # the "3" (for q/k/v) and the head dimension are explicit axes.
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        # Permute to (3, B, num_heads, S, head_dim) so we can index
        # the first axis to separate Q, K, V cleanly.
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention: softmax(Q K^T / sqrt(d_k)) V
        # The scale factor prevents dot products from growing too large
        # in magnitude, which would push softmax into saturated regions
        # with near-zero gradients.
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, S, S)

        if mask is not None:
            # Expand mask from (B, S) to (B, 1, 1, S) so it broadcasts
            # across the head and query dimensions. Padded *key* positions
            # are set to -inf so softmax assigns them zero weight.
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            attn = attn.masked_fill(~attn_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # If an entire row is masked (all keys are padding), softmax
        # produces NaN (log-sum-exp of all -inf). Replace NaN with 0
        # to avoid propagating NaN through the network.
        attn = torch.nan_to_num(attn, nan=0.0)

        # Weighted sum of values using the attention weights.
        out = torch.matmul(attn, v)  # (B, H, S, hd)

        # Concatenate heads: permute back to (B, S, H, hd) then flatten
        # the last two dims to recover the original embed_dim.
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single pre-norm transformer encoder block.

    Follows the "pre-norm" variant (LayerNorm *before* attention / FF),
    which tends to train more stably than the original post-norm design.

    Structure (with residual connections):
        x_out = x + Attention(LayerNorm(x))
        x_out = x_out + FeedForward(LayerNorm(x_out))

    The feed-forward sub-layer uses a GELU activation, which is smoother
    than ReLU and has become standard in modern transformer architectures.

    Attributes:
        attn (MultiHeadSelfAttention): Self-attention sub-layer.
        ln1 (nn.LayerNorm): Layer norm applied before attention.
        ff (nn.Sequential): Two-layer feed-forward network with GELU.
        ln2 (nn.LayerNorm): Layer norm applied before the feed-forward.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int = None):
        """Initialise a transformer encoder block.

        Args:
            embed_dim: Token embedding dimensionality.
            num_heads: Number of attention heads.
            ff_dim: Hidden dimensionality of the feed-forward sub-layer.
                Defaults to ``4 * embed_dim`` (the standard transformer
                expansion factor).
        """
        super().__init__()
        # Default FF hidden size is 4x the embedding dim, following the
        # original "Attention Is All You Need" convention.
        ff_dim = ff_dim or 4 * embed_dim

        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)  # pre-attention norm

        # Position-wise feed-forward: expand -> GELU -> contract.
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)  # pre-FF norm

    def forward(self, x, mask=None):
        """Apply one transformer block with residual connections.

        Args:
            x: Token tensor of shape ``(batch, seq_len, embed_dim)``.
            mask: Optional boolean mask of shape ``(batch, seq_len)``.

        Returns:
            Transformed tokens of the same shape ``(batch, seq_len, embed_dim)``.
        """
        # --- Sub-layer 1: self-attention with pre-norm residual ---
        # The residual (skip) connection adds the original input to the
        # attention output, enabling gradient flow through deep stacks.
        x = x + self.attn(self.ln1(x), mask)

        # --- Sub-layer 2: feed-forward with pre-norm residual ---
        x = x + self.ff(self.ln2(x))
        return x


class AttentionWorldModel(nn.Module):
    """Transformer-based world model that predicts next observation.

    The model treats each vehicle (ego + up to ``max_vehicles - 1``
    neighbours) as a token in a short sequence. Self-attention allows
    every vehicle's prediction to be conditioned on the full traffic
    context, capturing interactions like following distance adjustments
    and lane-change reactions.

    Input representation (flat observation vector):
      - **Ego token features** (first 3 values): ``[ego_speed, ego_lane, ego_x]``
      - **Neighbour i features** (4 values each): ``[rel_x, rel_lane, rel_speed, exists]``
        where ``exists`` is a binary indicator (1.0 if the neighbour slot
        is occupied, 0.0 if padding).

    The ego action is injected by concatenating its one-hot encoding to
    the ego features *before* the embedding layer, so the transformer
    sees action-conditioned ego tokens.

    Output:
      - Predicted next observation in the same flat format as the input.
        Predictions are *residual*: the model predicts deltas which are
        added to the current state.

    Attributes:
        max_vehicles (int): Maximum number of vehicle tokens (ego + neighbours).
        num_actions (int): Size of the discrete action space.
        embed_dim (int): Dimensionality of token embeddings.
        ego_embed (VehicleEmbedding): Embedding layer for the ego token.
        neighbor_embed (VehicleEmbedding): Embedding layer for neighbour tokens.
        type_embed (nn.Embedding): Learned type embedding with 2 classes
            (index 0 = ego, index 1 = neighbour) added to tokens so the
            transformer can distinguish vehicle roles.
        layers (nn.ModuleList): Stack of TransformerBlock layers.
        ego_head (nn.Linear): Prediction head for the ego vehicle's delta.
        neighbor_head (nn.Linear): Prediction head for each neighbour's delta.
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, max_vehicles: int = 7,
                 num_actions: int = 9):
        """Initialise the attention-based world model.

        Args:
            embed_dim: Dimensionality of the token embedding space.
            num_heads: Number of attention heads in each transformer block.
            num_layers: Number of stacked transformer encoder blocks.
            max_vehicles: Maximum total vehicles (1 ego + N neighbours).
                Determines the sequence length for the transformer.
            num_actions: Number of discrete actions. The ego action is
                one-hot encoded and appended to the ego feature vector.
        """
        super().__init__()
        self.max_vehicles = max_vehicles
        self.num_actions = num_actions
        self.embed_dim = embed_dim

        # --- Feature dimensions ---
        # Ego raw features: 3 state values + one-hot action vector.
        ego_feat_dim = 3 + num_actions
        # Neighbour raw features: relative position, lane offset, relative
        # speed, and an existence flag (1 if real, 0 if padding).
        neighbor_feat_dim = 4

        # Separate embedding layers because ego and neighbour feature
        # spaces have different dimensionalities.
        self.ego_embed = VehicleEmbedding(ego_feat_dim, embed_dim)
        self.neighbor_embed = VehicleEmbedding(neighbor_feat_dim, embed_dim)

        # Learned type embedding: index 0 for ego, index 1 for neighbour.
        # Added (not concatenated) to each token so the transformer can
        # differentiate vehicle roles without increasing the token dimension.
        self.type_embed = nn.Embedding(2, embed_dim)  # 0=ego, 1=neighbor

        # Stack of transformer encoder blocks.
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # --- Prediction heads (output residual deltas) ---
        # Ego head outputs 3 values: (delta_speed, delta_lane, delta_x).
        self.ego_head = nn.Linear(embed_dim, 3)
        # Neighbour head outputs 3 values per neighbour:
        # (delta_rel_x, delta_rel_lane, delta_rel_speed).
        self.neighbor_head = nn.Linear(embed_dim, 3)

    def forward(self, obs: torch.Tensor,
                action_idx: torch.Tensor) -> torch.Tensor:
        """Predict the full next observation (legacy interface).

        This is the primary forward pass used during training. It predicts
        deltas for both the ego and neighbour vehicles, then adds those
        deltas to the current state (residual prediction).

        Args:
            obs: Flat observation tensor of shape ``(batch, obs_dim)``
                where ``obs_dim = 3 + 4*k`` (3 ego features + 4 features
                per neighbour).
            action_idx: Integer action indices of shape ``(batch,)``.

        Returns:
            Predicted next observation of shape ``(batch, obs_dim)`` in
            the same flat format as the input.
        """
        B = obs.shape[0]
        device = obs.device

        # Encode observation into tokens with padding mask.
        ego_feats, neighbor_feats, tokens, mask = self._encode(obs, action_idx)
        k = neighbor_feats.shape[1]  # number of neighbour slots

        # Pass the token sequence through the transformer stack.
        x = tokens
        for layer in self.layers:
            x = layer(x, mask)

        # --- Extract predictions from transformer output tokens ---
        # Token 0 is always the ego vehicle; tokens 1..k are neighbours.
        ego_delta = self.ego_head(x[:, 0, :])       # (B, 3)
        neighbor_delta = self.neighbor_head(x[:, 1:, :])  # (B, k, 3)

        # --- Residual prediction: next_state = current_state + delta ---
        # For the ego vehicle, add predicted delta to current ego features.
        pred_ego = ego_feats + ego_delta

        # For neighbours, only the first 3 channels (rel_x, rel_lane,
        # rel_speed) are predicted as deltas. The 4th channel ("exists")
        # is kept unchanged because vehicle presence is not predicted.
        pred_neighbors = neighbor_feats.clone()
        pred_neighbors[:, :, :3] = neighbor_feats[:, :, :3] + neighbor_delta

        # Re-flatten into the same observation format: [ego(3), neighbours(k*4)].
        pred_obs = torch.cat([
            pred_ego,
            pred_neighbors.reshape(B, -1),
        ], dim=-1)

        return pred_obs

    def forward_npc_deltas(self, obs: torch.Tensor,
                           action_idx: torch.Tensor) -> torch.Tensor:
        """Predict only NPC (non-player character) deltas.

        Used by the hybrid planner where the ego vehicle's next state is
        computed via a kinematic model rather than the neural network.
        Only neighbour/NPC state changes are needed from this world model.

        The returned deltas are in *relative* coordinates as predicted by
        the neighbour head. The planner is responsible for removing the
        ego motion component to recover absolute NPC deltas.

        Args:
            obs: Flat observation tensor of shape ``(batch, obs_dim)``.
                Can be in any coordinate frame as long as it matches the
                training distribution.
            action_idx: Integer action indices of shape ``(batch,)``.

        Returns:
            NPC delta tensor of shape ``(batch, k, 3)`` containing
            ``(delta_rel_x, delta_rel_lane, delta_rel_speed)`` per
            neighbour slot.
        """
        B = obs.shape[0]

        # Reuse the shared encoding (same tokenisation as full forward).
        ego_feats, neighbor_feats, tokens, mask = self._encode(obs, action_idx)
        k = neighbor_feats.shape[1]

        # Run the transformer stack.
        x = tokens
        for layer in self.layers:
            x = layer(x, mask)

        # Only extract the neighbour deltas (ignore the ego token output).
        neighbor_delta = self.neighbor_head(x[:, 1:, :])  # (B, k, 3)

        # Return all 3 delta channels. Indices: 0 = delta_rel_x,
        # 1 = delta_rel_lane, 2 = delta_rel_speed. The caller (planner)
        # subtracts ego motion to obtain absolute NPC deltas.
        return neighbor_delta  # (B, k, 3)

    def _encode(self, obs, action_idx):
        """Shared encoding: parse flat obs, build token sequence and mask.

        This helper is called by both ``forward`` and ``forward_npc_deltas``
        to avoid duplicating the tokenisation logic.

        Steps:
          1. Slice the flat observation into ego features and neighbour features.
          2. One-hot encode the ego action and concatenate it to ego features.
          3. Project ego and neighbour features through their respective
             embedding layers.
          4. Add learned type embeddings (ego=0, neighbour=1) to each token.
          5. Concatenate all tokens into a single sequence.
          6. Build a boolean mask from the neighbour "exists" flags so that
             padding slots are ignored by attention.

        Args:
            obs: Flat observation of shape ``(batch, obs_dim)``.
            action_idx: Integer action indices of shape ``(batch,)``.

        Returns:
            A tuple of four tensors:
              - ego_feats: ``(batch, 3)`` -- raw ego state features.
              - neighbor_feats: ``(batch, k, 4)`` -- raw neighbour features.
              - tokens: ``(batch, 1+k, embed_dim)`` -- full token sequence
                with ego at position 0 and neighbours at positions 1..k.
              - mask: ``(batch, 1+k)`` -- boolean mask (True = valid token).
        """
        B = obs.shape[0]
        device = obs.device

        # --- Parse flat observation ---
        # First 3 values are ego features: [speed, lane, x].
        ego_feats = obs[:, :3]
        # Remaining values are k neighbours, each with 4 features.
        k = (obs.shape[1] - 3) // 4
        neighbor_feats = obs[:, 3:].reshape(B, k, 4)

        # --- Build ego token ---
        # One-hot encode the action and concatenate to ego features so
        # the embedding layer receives [speed, lane, x, action_0, ..., action_8].
        action_oh = F.one_hot(action_idx.long(),
                              self.num_actions).float()

        ego_input = torch.cat([ego_feats, action_oh], dim=-1)  # (B, 3+num_actions)
        ego_tok = self.ego_embed(ego_input)  # (B, embed_dim)
        # Add type embedding for "ego" (index 0).
        ego_tok = ego_tok + self.type_embed(
            torch.zeros(B, dtype=torch.long, device=device))

        # --- Build neighbour tokens ---
        # Reshape to (B*k, 4) for batch embedding, then back to (B, k, embed_dim).
        n_tok = self.neighbor_embed(
            neighbor_feats.reshape(B * k, 4)).reshape(B, k, -1)
        # Add type embedding for "neighbour" (index 1). The unsqueeze(1)
        # broadcasts the same type vector across all k neighbour slots.
        n_tok = n_tok + self.type_embed(
            torch.ones(B, dtype=torch.long, device=device)).unsqueeze(1)

        # --- Assemble token sequence ---
        # Ego token is always at position 0, neighbours follow.
        tokens = torch.cat([ego_tok.unsqueeze(1), n_tok], dim=1)  # (B, 1+k, embed_dim)

        # --- Build attention mask ---
        # A neighbour is "real" when its exists flag (4th feature) > 0.5.
        # Padding slots (exists=0) will be masked out in attention.
        neighbor_mask = neighbor_feats[:, :, 3] > 0.5  # (B, k)
        # Ego is always present, so its mask entry is always True.
        ego_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        mask = torch.cat([ego_mask, neighbor_mask], dim=1)  # (B, 1+k)

        return ego_feats, neighbor_feats, tokens, mask
