"""
Phase 0: Validation Study
=========================
Goal: Validate that layer dynamics differ between accepted and rejected tokens.

This script:
1. Runs Eagle3 inference with logging
2. Collects layer dynamics for each draft token (accepted/rejected)
3. Performs statistical analysis (t-test, effect size)

Success Criteria:
- p < 0.05 AND Cohen's d > 0.2 → Proceed to Phase 1
- Otherwise → Pivot or stop
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm


def compute_layer_dynamics(hidden_states: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Compute layer dynamics features from hidden states.

    Args:
        hidden_states: List of hidden states from different layers
                      Each tensor shape: [batch, seq_len, hidden_size]

    Returns:
        Dictionary containing:
        - cos_sim_01: Cosine similarity between layer 0 and 1
        - cos_sim_12: Cosine similarity between layer 1 and 2
        - delta_01: L2 norm of difference between layer 0 and 1
        - delta_12: L2 norm of difference between layer 1 and 2
        - entropy_0: Representation entropy of layer 0
        - entropy_1: Representation entropy of layer 1
        - entropy_2: Representation entropy of layer 2
    """
    h0, h1, h2 = hidden_states[0], hidden_states[1], hidden_states[2]

    # Cosine similarity between adjacent layers
    # Shape: [batch, seq_len]
    cos_sim_01 = F.cosine_similarity(h0, h1, dim=-1)
    cos_sim_12 = F.cosine_similarity(h1, h2, dim=-1)

    # Delta (L2 norm of difference)
    delta_01 = torch.norm(h1 - h0, dim=-1)
    delta_12 = torch.norm(h2 - h1, dim=-1)

    # Representation entropy (normalize and compute entropy)
    def compute_repr_entropy(h):
        # Normalize to probability-like distribution
        h_abs = torch.abs(h)
        h_norm = h_abs / (h_abs.sum(dim=-1, keepdim=True) + 1e-9)
        # Compute entropy
        entropy = -torch.sum(h_norm * torch.log(h_norm + 1e-9), dim=-1)
        return entropy

    entropy_0 = compute_repr_entropy(h0)
    entropy_1 = compute_repr_entropy(h1)
    entropy_2 = compute_repr_entropy(h2)

    return {
        "cos_sim_01": cos_sim_01,
        "cos_sim_12": cos_sim_12,
        "delta_01": delta_01,
        "delta_12": delta_12,
        "entropy_0": entropy_0,
        "entropy_1": entropy_1,
        "entropy_2": entropy_2,
    }


def collect_dynamics_from_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run inference and collect layer dynamics for accepted/rejected tokens.

    Returns:
        Tuple of (accepted_dynamics, rejected_dynamics)
        Each is a list of dicts with features for each position
    """
    accepted_dynamics = []
    rejected_dynamics = []

    for prompt in tqdm(prompts, desc="Processing prompts"):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Run inference with logging enabled
        # This requires modifying Eagle3's inference code to return dynamics
        # For now, we'll use a mock structure

        # TODO: Integrate with actual Eagle3 inference
        # The actual implementation would:
        # 1. Call model.generate() with modifications
        # 2. At each step, log:
        #    - draft tokens (positions 0-6)
        #    - target model's hidden states at each layer
        #    - which tokens were accepted/rejected

        pass

    return accepted_dynamics, rejected_dynamics


def compute_statistics(
    accepted: List[float],
    rejected: List[float],
    feature_name: str
) -> Dict:
    """
    Compute statistical comparison between accepted and rejected groups.

    Returns:
        Dict with t-statistic, p-value, Cohen's d, means, stds
    """
    accepted = np.array(accepted)
    rejected = np.array(rejected)

    # T-test
    t_stat, p_value = stats.ttest_ind(accepted, rejected)

    # Cohen's d (effect size)
    pooled_std = np.sqrt(
        ((len(accepted) - 1) * np.std(accepted, ddof=1)**2 +
         (len(rejected) - 1) * np.std(rejected, ddof=1)**2) /
        (len(accepted) + len(rejected) - 2)
    )
    cohens_d = (np.mean(accepted) - np.mean(rejected)) / (pooled_std + 1e-9)

    return {
        "feature": feature_name,
        "accepted_mean": float(np.mean(accepted)),
        "accepted_std": float(np.std(accepted)),
        "rejected_mean": float(np.mean(rejected)),
        "rejected_std": float(np.std(rejected)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "n_accepted": int(len(accepted)),
        "n_rejected": int(len(rejected)),
        "significant": bool(p_value < 0.05 and abs(cohens_d) > 0.2),
    }


def analyze_results(
    accepted_dynamics: List[Dict],
    rejected_dynamics: List[Dict],
) -> Dict:
    """
    Perform statistical analysis on collected dynamics.
    """
    features = [
        "cos_sim_01", "cos_sim_12",
        "delta_01", "delta_12",
        "entropy_0", "entropy_1", "entropy_2",
    ]

    results = {"features": {}, "summary": {}}

    for feature in features:
        accepted_values = [d[feature] for d in accepted_dynamics if feature in d]
        rejected_values = [d[feature] for d in rejected_dynamics if feature in d]

        if len(accepted_values) > 0 and len(rejected_values) > 0:
            stats_result = compute_statistics(accepted_values, rejected_values, feature)
            results["features"][feature] = stats_result

    # Summary
    significant_features = [
        f for f, r in results["features"].items() if r["significant"]
    ]

    results["summary"] = {
        "total_features": int(len(features)),
        "significant_features": int(len(significant_features)),
        "significant_feature_names": significant_features,
        "go_decision": bool(len(significant_features) >= 1),  # At least 1 significant feature
    }

    return results


def create_mock_data_for_testing() -> Tuple[List[Dict], List[Dict]]:
    """
    Create mock data to test the analysis pipeline.
    In real implementation, this will be replaced by actual inference data.
    """
    np.random.seed(42)
    n_accepted = 500
    n_rejected = 300

    # Simulate that accepted tokens have higher cosine similarity (more stable)
    accepted = [
        {
            "cos_sim_01": np.random.normal(0.85, 0.05),
            "cos_sim_12": np.random.normal(0.82, 0.06),
            "delta_01": np.random.normal(1.2, 0.3),
            "delta_12": np.random.normal(1.5, 0.4),
            "entropy_0": np.random.normal(5.0, 0.5),
            "entropy_1": np.random.normal(5.2, 0.5),
            "entropy_2": np.random.normal(5.3, 0.5),
            "position": np.random.randint(0, 7),
        }
        for _ in range(n_accepted)
    ]

    # Simulate that rejected tokens have lower cosine similarity (less stable)
    rejected = [
        {
            "cos_sim_01": np.random.normal(0.75, 0.08),  # Lower similarity
            "cos_sim_12": np.random.normal(0.70, 0.10),  # Lower similarity
            "delta_01": np.random.normal(1.8, 0.5),      # Higher delta
            "delta_12": np.random.normal(2.2, 0.6),      # Higher delta
            "entropy_0": np.random.normal(5.5, 0.6),     # Higher entropy
            "entropy_1": np.random.normal(5.8, 0.7),
            "entropy_2": np.random.normal(6.0, 0.8),
            "position": np.random.randint(0, 7),
        }
        for _ in range(n_rejected)
    ]

    return accepted, rejected


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Validation Study")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to Eagle3 model")
    parser.add_argument("--target_model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                       help="Target model name")
    parser.add_argument("--output_dir", type=str, default="./phase0_results",
                       help="Output directory for results")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock data for testing the pipeline")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of prompts to process")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 0: Validation Study")
    print("=" * 60)
    print(f"Goal: Validate layer dynamics ↔ acceptance correlation")
    print(f"Success Criteria: p < 0.05 AND Cohen's d > 0.2")
    print("=" * 60)

    if args.mock:
        print("\n[Using mock data for pipeline testing]")
        accepted_dynamics, rejected_dynamics = create_mock_data_for_testing()
    else:
        # TODO: Load model and run actual inference
        print("\n[Real inference mode - not yet implemented]")
        print("Please use --mock flag for now to test the pipeline")
        return

    # Analyze results
    print("\nAnalyzing results...")
    results = analyze_results(accepted_dynamics, rejected_dynamics)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for feature, stats in results["features"].items():
        sig = "✓" if stats["significant"] else "✗"
        print(f"\n{feature}:")
        print(f"  Accepted: {stats['accepted_mean']:.4f} ± {stats['accepted_std']:.4f} (n={stats['n_accepted']})")
        print(f"  Rejected: {stats['rejected_mean']:.4f} ± {stats['rejected_std']:.4f} (n={stats['n_rejected']})")
        print(f"  t = {stats['t_statistic']:.4f}, p = {stats['p_value']:.6f}")
        print(f"  Cohen's d = {stats['cohens_d']:.4f}")
        print(f"  Significant: {sig}")

    print("\n" + "=" * 60)
    print("DECISION")
    print("=" * 60)

    summary = results["summary"]
    print(f"Significant features: {summary['significant_features']}/{summary['total_features']}")
    print(f"Features: {summary['significant_feature_names']}")

    if summary["go_decision"]:
        print("\n🟢 GO: Proceed to Phase 1")
        print("Layer dynamics show significant correlation with acceptance!")
    else:
        print("\n🔴 NO-GO: Consider pivoting")
        print("Layer dynamics do not show significant correlation.")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"validation_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Also save a summary for quick reference
    summary_file = output_dir / "latest_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Phase 0 Validation Study Results\n")
        f.write(f"================================\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Decision: {'GO' if summary['go_decision'] else 'NO-GO'}\n")
        f.write(f"Significant features: {summary['significant_feature_names']}\n")

    return results


if __name__ == "__main__":
    main()
