"""
Phase 1: Layer Consistency Loss Patch for Eagle3
=================================================
This module patches Eagle3's training to add layer consistency auxiliary loss.

Usage:
    from phase1_cnets_patch import patch_eagle3_for_phase1
    patch_eagle3_for_phase1(model, lambda_consistency=0.1)

The patch modifies:
1. dataprepare() - compute target layer consistency
2. forward() - add consistency loss to total loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class LayerConsistencyLoss(nn.Module):
    """
    Auxiliary loss for layer consistency.

    Encourages draft model's input->output consistency to match
    target model's layer-to-layer consistency.
    """

    def __init__(self, lambda_weight: float = 0.1):
        super().__init__()
        self.lambda_weight = lambda_weight

    def forward(
        self,
        draft_input: torch.Tensor,    # [batch, seq, hidden]
        draft_output: torch.Tensor,   # [batch, seq, hidden]
        target_consistency: torch.Tensor,  # [batch, seq]
        position_mask: torch.Tensor,  # [batch, seq, 1]
    ) -> torch.Tensor:
        """
        Compute layer consistency loss.

        Args:
            draft_input: Input embeddings to draft model
            draft_output: Output hidden states from draft model
            target_consistency: Target model's average layer consistency
            position_mask: Mask for valid positions

        Returns:
            Weighted consistency loss
        """
        # Compute draft consistency (input -> output)
        draft_consistency = F.cosine_similarity(draft_input, draft_output, dim=-1)

        # MSE loss between draft and target consistency
        loss = F.mse_loss(
            draft_consistency * position_mask.squeeze(-1),
            target_consistency * position_mask.squeeze(-1),
            reduction='mean'
        )

        return self.lambda_weight * loss


def compute_target_layer_consistency(
    hidden_states0: torch.Tensor,
    hidden_states1: torch.Tensor,
    hidden_states2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute target model's layer consistency metrics.

    Returns:
        Tuple of (cos_sim_01, cos_sim_12, avg_consistency)
    """
    cos_sim_01 = F.cosine_similarity(hidden_states0, hidden_states1, dim=-1)
    cos_sim_12 = F.cosine_similarity(hidden_states1, hidden_states2, dim=-1)
    avg_consistency = (cos_sim_01 + cos_sim_12) / 2

    return cos_sim_01, cos_sim_12, avg_consistency


def patched_dataprepare(self, input_ids, attention_mask, loss_mask):
    """
    Modified dataprepare that also returns layer consistency.
    """
    device = input_ids.device
    outs = self.target_model(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states0 = outs.hidden_states[0]
    hidden_states1 = outs.hidden_states[1]
    hidden_states2 = outs.hidden_states[2]
    hidden_states = torch.cat((hidden_states0, hidden_states1, hidden_states2), dim=-1)

    # NEW: Compute layer consistency
    cos_sim_01, cos_sim_12, avg_consistency = compute_target_layer_consistency(
        hidden_states0, hidden_states1, hidden_states2
    )

    # Padding for target
    from eagle.traineagle3.cnets import padding
    target = outs.logits
    target = padding(target, left=False)
    input_ids = padding(input_ids, left=False)

    if target is not None:
        target = target.to(device)
        loss_mask = loss_mask[..., None]
        loss_mask = loss_mask.to(device)

    # Return original + layer consistency
    return hidden_states, target, loss_mask, input_ids, avg_consistency.detach()


def patched_forward(
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
    Modified forward that includes layer consistency loss.
    """
    # Get data including layer consistency
    hidden_states, target, loss_mask, input_ids, target_consistency = \
        self.dataprepare(input_ids, attention_mask, loss_mask)

    batch_size, seq_length, _ = hidden_states.shape
    seq_length_with_past = seq_length
    past_key_values_length = 0

    if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
        hidden_states.requires_grad = True

    # Store input for consistency loss
    hidden_states_input = self.fc(hidden_states)
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
    consistency_losses = []  # NEW: track consistency losses
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

        # NEW: Compute consistency loss
        if hasattr(self, 'consistency_loss_fn'):
            from eagle.traineagle3.cnets import padding
            # Pad target_consistency and hidden_states_input for alignment
            if idx > 0:
                target_consistency_padded = padding(target_consistency.unsqueeze(-1), left=False).squeeze(-1)
                hidden_states_input_padded = padding(hidden_states_input, left=False)
            else:
                target_consistency_padded = target_consistency
                hidden_states_input_padded = hidden_states_input

            consistency_loss = self.consistency_loss_fn(
                draft_input=hidden_states_input_padded,
                draft_output=hidden_states_out,
                target_consistency=target_consistency_padded,
                position_mask=position_mask,
            )
            consistency_losses.append(consistency_loss)

        with torch.no_grad():
            acces.append(((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)).sum().item() / (
                    loss_mask.sum().item() + 1e-6))

        if not last:
            from eagle.traineagle3.cnets import padding
            input_ids = padding(input_ids, left=False)
            target = padding(target, left=False)
            loss_mask = padding(loss_mask, left=False)
            target_consistency = padding(target_consistency.unsqueeze(-1), left=False).squeeze(-1)
            hidden_states_input = padding(hidden_states_input, left=False)

    # Return consistency losses along with original returns
    return plosses, vlosses, acces, consistency_losses


def patch_eagle3_for_phase1(model, lambda_consistency: float = 0.1):
    """
    Patch an Eagle3 model for Phase 1 training with layer consistency loss.

    Args:
        model: Eagle3 training model (EModel)
        lambda_consistency: Weight for consistency loss
    """
    # Add consistency loss module
    model.consistency_loss_fn = LayerConsistencyLoss(lambda_weight=lambda_consistency)

    # Store original methods
    model._original_dataprepare = model.dataprepare
    model._original_forward = model.forward

    # Patch methods
    import types
    model.dataprepare = types.MethodType(patched_dataprepare, model)
    model.forward = types.MethodType(patched_forward, model)

    print(f"[Phase 1] Patched Eagle3 with layer consistency loss (λ={lambda_consistency})")
    return model


def unpatch_eagle3(model):
    """
    Remove Phase 1 patch from Eagle3 model.
    """
    if hasattr(model, '_original_dataprepare'):
        model.dataprepare = model._original_dataprepare
        model.forward = model._original_forward
        delattr(model, '_original_dataprepare')
        delattr(model, '_original_forward')
        delattr(model, 'consistency_loss_fn')
        print("[Phase 1] Removed patch from Eagle3")
    return model


# Test code
if __name__ == "__main__":
    print("Phase 1 Patch Module")
    print("=" * 40)
    print("This module provides:")
    print("  - LayerConsistencyLoss: Auxiliary loss class")
    print("  - patch_eagle3_for_phase1(): Apply patch to model")
    print("  - unpatch_eagle3(): Remove patch from model")
    print()
    print("Usage:")
    print("  from phase1_cnets_patch import patch_eagle3_for_phase1")
    print("  model = patch_eagle3_for_phase1(model, lambda_consistency=0.1)")
