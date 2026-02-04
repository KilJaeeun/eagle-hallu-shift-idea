"""
Phase 0: Eagle3 Inference with Layer Dynamics Logging
=====================================================
This script patches Eagle3's inference to collect layer dynamics
for accepted and rejected tokens at each position.

Usage:
    python scripts/phase0_inference_hook.py \
        --base_model meta-llama/Llama-2-7b-chat-hf \
        --ea_model yuhuili/EAGLE-llama2-chat-7B \
        --data_path data/mt_bench_prompts.json \
        --output_dir phase0_results
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Add EAGLE to path
sys.path.insert(0, str(Path(__file__).parent.parent / "EAGLE"))


class LayerDynamicsCollector:
    """
    Collector for layer dynamics during Eagle3 inference.
    Patches into the evaluation process to collect statistics.
    """

    def __init__(self):
        self.accepted_dynamics: List[Dict] = []
        self.rejected_dynamics: List[Dict] = []
        self.current_hidden_states: Optional[List[torch.Tensor]] = None
        self.current_position: int = 0

    def reset(self):
        self.current_hidden_states = None
        self.current_position = 0

    def set_hidden_states(self, hidden_states: List[torch.Tensor]):
        """Store current hidden states from target model."""
        self.current_hidden_states = [h.detach().clone() for h in hidden_states[:3]]

    def compute_dynamics(self, position_idx: int) -> Dict:
        """Compute layer dynamics for a specific position."""
        if self.current_hidden_states is None:
            return {}

        h0 = self.current_hidden_states[0][:, position_idx]
        h1 = self.current_hidden_states[1][:, position_idx]
        h2 = self.current_hidden_states[2][:, position_idx]

        # Compute features
        cos_sim_01 = F.cosine_similarity(h0, h1, dim=-1).item()
        cos_sim_12 = F.cosine_similarity(h1, h2, dim=-1).item()
        delta_01 = torch.norm(h1 - h0, dim=-1).item()
        delta_12 = torch.norm(h2 - h1, dim=-1).item()

        def compute_entropy(h):
            h_abs = torch.abs(h)
            h_norm = h_abs / (h_abs.sum(dim=-1, keepdim=True) + 1e-9)
            return -torch.sum(h_norm * torch.log(h_norm + 1e-9), dim=-1).item()

        return {
            "cos_sim_01": cos_sim_01,
            "cos_sim_12": cos_sim_12,
            "delta_01": delta_01,
            "delta_12": delta_12,
            "entropy_0": compute_entropy(h0),
            "entropy_1": compute_entropy(h1),
            "entropy_2": compute_entropy(h2),
            "position": self.current_position,
        }

    def record_acceptance(self, position_indices: List[int], is_accepted: List[bool]):
        """Record whether each position was accepted or rejected."""
        for pos_idx, accepted in zip(position_indices, is_accepted):
            dynamics = self.compute_dynamics(pos_idx)
            dynamics["draft_position"] = pos_idx

            if accepted:
                self.accepted_dynamics.append(dynamics)
            else:
                self.rejected_dynamics.append(dynamics)

        self.current_position += 1

    def get_results(self) -> Tuple[List[Dict], List[Dict]]:
        return self.accepted_dynamics, self.rejected_dynamics

    def save_results(self, output_path: str):
        results = {
            "accepted": self.accepted_dynamics,
            "rejected": self.rejected_dynamics,
            "stats": {
                "n_accepted": len(self.accepted_dynamics),
                "n_rejected": len(self.rejected_dynamics),
            }
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)


def patch_evaluate_posterior(collector: LayerDynamicsCollector):
    """
    Create a patched version of evaluate_posterior that logs dynamics.
    """
    from eagle.model import utils as eagle_utils

    original_evaluate_posterior = eagle_utils.evaluate_posterior

    def patched_evaluate_posterior(logits, candidates, logits_processor):
        result = original_evaluate_posterior(logits, candidates, logits_processor)
        best_candidate, accept_length, sample_p = result

        # Log which positions were accepted/rejected
        # Positions 0 to accept_length were accepted
        # Positions accept_length+1 onwards were rejected
        n_candidates = candidates.shape[1] - 1  # Exclude the initial token

        is_accepted = [i <= accept_length.item() for i in range(n_candidates)]
        position_indices = list(range(n_candidates))

        # Record dynamics
        if collector.current_hidden_states is not None:
            collector.record_acceptance(position_indices, is_accepted)

        return result

    return patched_evaluate_posterior


def patch_tree_decoding(collector: LayerDynamicsCollector):
    """
    Create a patched version of tree_decoding that captures hidden states.
    """
    from eagle.model import utils as eagle_utils

    original_tree_decoding = eagle_utils.tree_decoding

    def patched_tree_decoding(model, tree_candidates, past_key_values,
                               tree_position_ids, input_ids, retrieve_indices):
        result = original_tree_decoding(
            model, tree_candidates, past_key_values,
            tree_position_ids, input_ids, retrieve_indices
        )
        logits, hidden_state, outputs = result

        # Capture hidden states from target model
        if "hidden_states" in outputs and len(outputs["hidden_states"]) >= 3:
            collector.set_hidden_states(outputs["hidden_states"])

        return result

    return patched_tree_decoding


def load_prompts(data_path: str, max_samples: int = 100) -> List[str]:
    """Load prompts from JSON file or create default prompts."""
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            prompts = data[:max_samples]
        elif isinstance(data, dict) and "prompts" in data:
            prompts = data["prompts"][:max_samples]
        else:
            prompts = list(data.values())[:max_samples]
        return prompts

    # Default prompts if file not found
    return [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to sort a list of numbers.",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis.",
        "How does a computer processor work?",
    ] * 20  # Repeat to get more samples


def run_inference_with_collection(
    base_model_path: str,
    ea_model_path: str,
    prompts: List[str],
    collector: LayerDynamicsCollector,
    max_new_tokens: int = 128,
    device: str = "cuda",
):
    """
    Run Eagle3 inference with layer dynamics collection.
    """
    try:
        from eagle.model.ea_model import EaModel
    except ImportError as e:
        print(f"Error: Could not import Eagle3 model. Make sure EAGLE is in the path.")
        print(f"Import error details: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Loading base model: {base_model_path}")
    print(f"Loading EA model: {ea_model_path}")

    # Load model
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Get tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Patch functions
    from eagle.model import utils as eagle_utils
    eagle_utils.evaluate_posterior = patch_evaluate_posterior(collector)
    eagle_utils.tree_decoding = patch_tree_decoding(collector)

    print(f"\nRunning inference on {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        collector.reset()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        try:
            with torch.no_grad():
                outputs = model.eagenerate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,  # Greedy for reproducibility
                )
        except Exception as e:
            print(f"Error on prompt {i}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(prompts)} prompts")
            print(f"  Accepted: {len(collector.accepted_dynamics)}, "
                  f"Rejected: {len(collector.rejected_dynamics)}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: Eagle3 Inference with Layer Dynamics Collection"
    )
    parser.add_argument(
        "--base_model", type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Base model path"
    )
    parser.add_argument(
        "--ea_model", type=str,
        default="yuhuili/EAGLE-llama2-chat-7B",
        help="Eagle model path"
    )
    parser.add_argument(
        "--data_path", type=str,
        default="data/prompts.json",
        help="Path to prompts JSON file"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./phase0_results",
        help="Output directory"
    )
    parser.add_argument(
        "--max_samples", type=int,
        default=100,
        help="Maximum number of prompts to process"
    )
    parser.add_argument(
        "--max_new_tokens", type=int,
        default=128,
        help="Maximum new tokens per prompt"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 0: Layer Dynamics Collection")
    print("=" * 60)

    # Create collector
    collector = LayerDynamicsCollector()

    # Load prompts
    prompts = load_prompts(args.data_path, args.max_samples)
    print(f"Loaded {len(prompts)} prompts")

    # Run inference
    run_inference_with_collection(
        base_model_path=args.base_model,
        ea_model_path=args.ea_model,
        prompts=prompts,
        collector=collector,
        max_new_tokens=args.max_new_tokens,
    )

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output = output_dir / f"dynamics_raw_{timestamp}.json"
    collector.save_results(str(raw_output))
    print(f"\nRaw dynamics saved to: {raw_output}")

    # Run analysis
    print("\nRunning statistical analysis...")
    from phase0_validation import analyze_results, compute_statistics

    accepted, rejected = collector.get_results()
    if len(accepted) > 0 and len(rejected) > 0:
        results = analyze_results(accepted, rejected)

        # Save analysis results
        analysis_output = output_dir / f"analysis_{timestamp}.json"
        with open(analysis_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Analysis saved to: {analysis_output}")

        # Print summary
        summary = results["summary"]
        print("\n" + "=" * 60)
        print("DECISION")
        print("=" * 60)
        print(f"Significant features: {summary['significant_features']}/{summary['total_features']}")

        if summary["go_decision"]:
            print("\n🟢 GO: Proceed to Phase 1")
        else:
            print("\n🔴 NO-GO: Consider pivoting")
    else:
        print("Not enough data for analysis.")
        print(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")


if __name__ == "__main__":
    main()
