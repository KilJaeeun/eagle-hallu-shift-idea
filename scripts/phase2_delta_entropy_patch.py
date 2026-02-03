"""
Phase 2: Layer Delta Prediction + Attention Entropy Conditioning
================================================================
This module extends Phase 1 with:
1. Layer Delta Prediction Head - predicts target's layer-to-layer change
2. Attention Entropy Conditioning - uses previous tokens' attention entropy

Usage:
    from phase2_delta_entropy_patch import patch_eagle3_for_phase2
    model = patch_eagle3_for_phase2(
        model,
        lambda_consistency=0.1,
        lambda_delta=0.1,
        use_attention_entropy=True,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict


class LayerDeltaPredictionHead(nn.Module):
    """
    Predicts target model's layer-to-layer delta (hidden[2] - hidden[1]).
    This teaches the draft to understand how representations evolve in target.
    """

    def __init__(self, hidden_size: int, lambda_weight: float = 0.1):
        super().__init__()
        self.lambda_weight = lambda_weight
        # Simple linear head to predict delta
        self.delta_head = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        draft_output: torch.Tensor,     # [batch, seq, hidden]
        target_delta: torch.Tensor,     # [batch, seq, hidden]
        position_mask: torch.Tensor,    # [batch, seq, 1]
    ) -> torch.Tensor:
        """
        Compute delta prediction loss.
        """
        predicted_delta = self.delta_head(draft_output)

        # MSE loss on predicted vs actual delta
        loss = F.mse_loss(
            predicted_delta * position_mask,
            target_delta * position_mask,
            reduction='mean'
        )

        return self.lambda_weight * loss


class AttentionEntropyEncoder(nn.Module):
    """
    Encodes attention entropy information for conditioning.
    Uses previous tokens' attention entropy (which is available during inference).
    """

    def __init__(self, hidden_size: int, num_layers: int = 3):
        super().__init__()
        # Project entropy to hidden_size dimension
        self.entropy_proj = nn.Sequential(
            nn.Linear(num_layers, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )

    def forward(
        self,
        attention_entropies: torch.Tensor,  # [batch, seq, num_layers]
    ) -> torch.Tensor:
        """
        Encode attention entropy into hidden representation.

        Returns:
            Entropy encoding of shape [batch, seq, hidden_size]
        """
        return self.entropy_proj(attention_entropies)


def compute_attention_entropy(
    attention_weights: torch.Tensor,  # [batch, heads, seq, seq]
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Compute attention entropy for each position.

    Returns:
        Entropy of shape [batch, seq]
    """
    # Clamp to avoid log(0)
    attn = attention_weights.clamp(min=eps)
    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(attn * torch.log(attn), dim=-1)
    # Average over heads
    entropy = entropy.mean(dim=1)  # [batch, seq]
    return entropy


class Phase2Loss(nn.Module):
    """
    Combined loss for Phase 2:
    - Layer consistency loss (from Phase 1)
    - Layer delta prediction loss
    - Optional attention entropy conditioning
    """

    def __init__(
        self,
        hidden_size: int,
        lambda_consistency: float = 0.1,
        lambda_delta: float = 0.1,
        use_attention_entropy: bool = True,
    ):
        super().__init__()
        self.lambda_consistency = lambda_consistency
        self.lambda_delta = lambda_delta
        self.use_attention_entropy = use_attention_entropy

        # Delta prediction head
        self.delta_head = LayerDeltaPredictionHead(hidden_size, lambda_delta)

        # Attention entropy encoder (optional)
        if use_attention_entropy:
            self.entropy_encoder = AttentionEntropyEncoder(hidden_size, num_layers=3)

    def compute_consistency_loss(
        self,
        draft_input: torch.Tensor,
        draft_output: torch.Tensor,
        target_consistency: torch.Tensor,
        position_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Layer consistency loss (same as Phase 1)."""
        draft_consistency = F.cosine_similarity(draft_input, draft_output, dim=-1)
        loss = F.mse_loss(
            draft_consistency * position_mask.squeeze(-1),
            target_consistency * position_mask.squeeze(-1),
            reduction='mean'
        )
        return self.lambda_consistency * loss

    def compute_delta_loss(
        self,
        draft_output: torch.Tensor,
        target_delta: torch.Tensor,
        position_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Layer delta prediction loss."""
        return self.delta_head(draft_output, target_delta, position_mask)

    def forward(
        self,
        draft_input: torch.Tensor,
        draft_output: torch.Tensor,
        target_consistency: torch.Tensor,
        target_delta: torch.Tensor,
        position_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all Phase 2 losses.

        Returns:
            Dict with 'consistency_loss', 'delta_loss', 'total_loss'
        """
        consistency_loss = self.compute_consistency_loss(
            draft_input, draft_output, target_consistency, position_mask
        )
        delta_loss = self.compute_delta_loss(
            draft_output, target_delta, position_mask
        )

        return {
            'consistency_loss': consistency_loss,
            'delta_loss': delta_loss,
            'total_loss': consistency_loss + delta_loss,
        }


def patched_dataprepare_phase2(self, input_ids, attention_mask, loss_mask):
    """
    Modified dataprepare that returns:
    - hidden_states (concatenated)
    - target logits
    - loss_mask
    - input_ids
    - target_consistency (avg cosine similarity between layers)
    - target_delta (hidden[2] - hidden[1])
    - attention_entropies (optional)
    """
    device = input_ids.device
    outs = self.target_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=self.use_attention_entropy,
    )

    hidden_states0 = outs.hidden_states[0]
    hidden_states1 = outs.hidden_states[1]
    hidden_states2 = outs.hidden_states[2]
    hidden_states = torch.cat((hidden_states0, hidden_states1, hidden_states2), dim=-1)

    # Compute layer consistency
    cos_sim_01 = F.cosine_similarity(hidden_states0, hidden_states1, dim=-1)
    cos_sim_12 = F.cosine_similarity(hidden_states1, hidden_states2, dim=-1)
    avg_consistency = (cos_sim_01 + cos_sim_12) / 2

    # Compute layer delta
    target_delta = hidden_states2 - hidden_states1

    # Compute attention entropies (if enabled)
    attention_entropies = None
    if self.use_attention_entropy and hasattr(outs, 'attentions') and outs.attentions is not None:
        # Get attention from first 3 layers
        attn_entropies = []
        for i in range(min(3, len(outs.attentions))):
            entropy = compute_attention_entropy(outs.attentions[i])
            attn_entropies.append(entropy)
        attention_entropies = torch.stack(attn_entropies, dim=-1)  # [batch, seq, 3]

    # Padding for target
    from eagle.traineagle3.cnets import padding
    target = outs.logits
    target = padding(target, left=False)
    input_ids = padding(input_ids, left=False)

    if target is not None:
        target = target.to(device)
        loss_mask = loss_mask[..., None]
        loss_mask = loss_mask.to(device)

    return (
        hidden_states,
        target,
        loss_mask,
        input_ids,
        avg_consistency.detach(),
        target_delta.detach(),
        attention_entropies.detach() if attention_entropies is not None else None,
    )


def patched_forward_phase2(
    self,
    input_ids,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    loss_mask: Optional[torch.Tensor] = None,
):
    """
    Modified forward with Phase 2 losses (consistency + delta + entropy conditioning).
    """
    # Get data including layer dynamics
    result = self.dataprepare(input_ids, attention_mask, loss_mask)
    (hidden_states, target, loss_mask, input_ids,
     target_consistency, target_delta, attention_entropies) = result

    batch_size, seq_length, _ = hidden_states.shape
    seq_length_with_past = seq_length
    past_key_values_length = 0

    if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
        hidden_states.requires_grad = True

    # Apply FC projection
    hidden_states_input = self.fc(hidden_states)

    # Apply attention entropy conditioning (if enabled)
    if self.use_attention_entropy and attention_entropies is not None:
        entropy_encoding = self.phase2_loss.entropy_encoder(attention_entropies)
        hidden_states_input = hidden_states_input + entropy_encoding

    hidden_states = hidden_states_input

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    if position_ids is None:
        device = hidden_states.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
    )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    plosses = []
    vlosses = []
    acces = []
    phase2_losses = []
    cache_hidden = [[], []]

    for idx in range(self.length):
        last = idx == self.length - 1
        inputs_embeds = self.embed_tokens(input_ids)
        if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad = True
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, None, output_attentions)
                return custom_forward

            layer_outputs, cache_hidden = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.midlayer),
                inputs_embeds,
                hidden_states,
                cache_hidden,
                attention_mask,
                position_ids,
            )
        else:
            layer_outputs, cache_hidden = self.midlayer(
                input_emb=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=True,
            )

        hidden_states_out = layer_outputs[0]

        with torch.no_grad():
            target_head = target
            target_max_token = target_head.argmax(-1)
            self.t2d = self.t2d.to(target_max_token.device)
            target_mask = self.t2d[target_max_token]
            target_mask = target_mask[..., None].int()
            position_mask = target_mask * loss_mask
            target_head = target_head[..., self.t2d]
            target_head = target_head.float()
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()

        hidden_states = hidden_states_out
        hidden_states_out_norm = self.norm(hidden_states_out)

        logits = self.lm_head(hidden_states_out_norm)
        logits = logits.float()
        out_logp = nn.LogSoftmax(dim=2)(logits)
        plogp = target_p * out_logp
        loss = -torch.sum(position_mask * plogp, 2).mean()
        plosses.append(loss)

        # Compute Phase 2 losses
        if hasattr(self, 'phase2_loss'):
            from eagle.traineagle3.cnets import padding
            if idx > 0:
                target_consistency_padded = padding(target_consistency.unsqueeze(-1), left=False).squeeze(-1)
                target_delta_padded = padding(target_delta, left=False)
                hidden_states_input_padded = padding(hidden_states_input, left=False)
            else:
                target_consistency_padded = target_consistency
                target_delta_padded = target_delta
                hidden_states_input_padded = hidden_states_input

            p2_loss = self.phase2_loss(
                draft_input=hidden_states_input_padded,
                draft_output=hidden_states_out,
                target_consistency=target_consistency_padded,
                target_delta=target_delta_padded,
                position_mask=position_mask,
            )
            phase2_losses.append(p2_loss)

        with torch.no_grad():
            acces.append(((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)).sum().item() / (
                    loss_mask.sum().item() + 1e-6))

        if not last:
            from eagle.traineagle3.cnets import padding
            input_ids = padding(input_ids, left=False)
            target = padding(target, left=False)
            loss_mask = padding(loss_mask, left=False)
            target_consistency = padding(target_consistency.unsqueeze(-1), left=False).squeeze(-1)
            target_delta = padding(target_delta, left=False)
            hidden_states_input = padding(hidden_states_input, left=False)

    return plosses, vlosses, acces, phase2_losses


def patch_eagle3_for_phase2(
    model,
    lambda_consistency: float = 0.1,
    lambda_delta: float = 0.1,
    use_attention_entropy: bool = True,
):
    """
    Patch Eagle3 model for Phase 2 training.

    Args:
        model: Eagle3 training model (EModel)
        lambda_consistency: Weight for consistency loss
        lambda_delta: Weight for delta prediction loss
        use_attention_entropy: Whether to use attention entropy conditioning
    """
    # Get hidden size from model
    hidden_size = model.config.hidden_size

    # Add Phase 2 loss module
    model.phase2_loss = Phase2Loss(
        hidden_size=hidden_size,
        lambda_consistency=lambda_consistency,
        lambda_delta=lambda_delta,
        use_attention_entropy=use_attention_entropy,
    )

    # Store configuration
    model.use_attention_entropy = use_attention_entropy

    # Store original methods
    model._original_dataprepare = model.dataprepare
    model._original_forward = model.forward

    # Patch methods
    import types
    model.dataprepare = types.MethodType(patched_dataprepare_phase2, model)
    model.forward = types.MethodType(patched_forward_phase2, model)

    print(f"[Phase 2] Patched Eagle3:")
    print(f"  - Lambda consistency: {lambda_consistency}")
    print(f"  - Lambda delta: {lambda_delta}")
    print(f"  - Attention entropy: {use_attention_entropy}")

    return model


def unpatch_eagle3_phase2(model):
    """Remove Phase 2 patch from Eagle3 model."""
    if hasattr(model, '_original_dataprepare'):
        model.dataprepare = model._original_dataprepare
        model.forward = model._original_forward
        delattr(model, '_original_dataprepare')
        delattr(model, '_original_forward')
        if hasattr(model, 'phase2_loss'):
            delattr(model, 'phase2_loss')
        if hasattr(model, 'use_attention_entropy'):
            delattr(model, 'use_attention_entropy')
        print("[Phase 2] Removed patch from Eagle3")
    return model


if __name__ == "__main__":
    print("Phase 2: Delta + Entropy Patch Module")
    print("=" * 50)
    print("This module provides:")
    print("  - LayerDeltaPredictionHead: Predicts layer delta")
    print("  - AttentionEntropyEncoder: Encodes attention entropy")
    print("  - Phase2Loss: Combined loss module")
    print("  - patch_eagle3_for_phase2(): Apply Phase 2 patch")
    print()
    print("Usage:")
    print("  from phase2_delta_entropy_patch import patch_eagle3_for_phase2")
    print("  model = patch_eagle3_for_phase2(")
    print("      model,")
    print("      lambda_consistency=0.1,")
    print("      lambda_delta=0.1,")
    print("      use_attention_entropy=True,")
    print("  )")
