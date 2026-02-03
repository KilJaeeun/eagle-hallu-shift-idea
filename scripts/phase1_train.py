"""
Phase 1: Training with Layer Consistency Loss
==============================================
Modified Eagle3 training script with layer consistency auxiliary loss.

Usage:
    deepspeed --num_gpus=8 scripts/phase1_train.py \
        --base_model meta-llama/Llama-2-7b-chat-hf \
        --data_path data/sharegpt_train.json \
        --lambda_consistency 0.1 \
        --output_dir checkpoints/phase1

Based on Eagle3's training script with modifications for layer consistency loss.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "EAGLE"))
sys.path.insert(0, str(Path(__file__).parent))

from phase1_cnets_patch import patch_eagle3_for_phase1, LayerConsistencyLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Layer Consistency Training")

    # Model arguments
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model path (e.g., meta-llama/Llama-2-7b-chat-hf)")
    parser.add_argument("--draft_model", type=str, default=None,
                       help="Pretrained draft model to fine-tune (optional)")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--test_data_path", type=str, default=None,
                       help="Path to test data")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Phase 1 specific arguments
    parser.add_argument("--lambda_consistency", type=float, default=0.1,
                       help="Weight for layer consistency loss")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase1")
    parser.add_argument("--wandb_project", type=str, default="eagle-hallushift")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # DeepSpeed arguments
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed training
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = args.local_rank

    # Set seed
    set_seed(42)

    # Setup device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(f"[Rank {rank}] Initializing Phase 1 training...")
    print(f"[Rank {rank}] Lambda consistency: {args.lambda_consistency}")

    # Initialize wandb on rank 0
    if rank == 0:
        run_name = args.wandb_run_name or f"phase1_lambda{args.lambda_consistency}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Load model
    print(f"[Rank {rank}] Loading model...")
    from eagle.traineagle3.cnets import EModel
    model = EModel(
        base_model_path=args.base_model,
        use_flash_attn=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Apply Phase 1 patch
    model = patch_eagle3_for_phase1(model, lambda_consistency=args.lambda_consistency)

    # Load dataset
    print(f"[Rank {rank}] Loading dataset...")
    from eagle.traineagle3.getdata import get_dataset
    train_dataset = get_dataset(args.data_path, tokenizer)

    if args.test_data_path:
        test_dataset = get_dataset(args.test_data_path, tokenizer)
    else:
        # Use 10% of train as test
        train_size = int(0.9 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, test_size]
        )

    # Create data loaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        collate_fn=test_dataset.collate_fn if hasattr(test_dataset, 'collate_fn') else None,
    )

    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": args.batch_size * world_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_steps
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"},
            "contiguous_gradients": True,
            "overlap_comm": True
        }
    }

    # Initialize DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    # Training loop
    print(f"[Rank {rank}] Starting training...")

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        epoch_plosses = [[] for _ in range(model.length)]
        epoch_closses = []  # Consistency losses
        epoch_acces = [[] for _ in range(model.length)]

        for batch_idx, data in enumerate(tqdm(train_loader, disable=rank != 0)):
            model_engine.zero_grad()

            # Forward pass (returns consistency losses too now)
            outputs = model_engine(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                loss_mask=data["loss_mask"],
            )

            # Unpack outputs (modified for Phase 1)
            if len(outputs) == 4:
                plosses, vlosses, acces, consistency_losses = outputs
            else:
                # Fallback for non-patched model
                plosses, vlosses, acces = outputs
                consistency_losses = []

            # Compute prediction loss
            ploss_weight = [0.8 ** i for i in range(len(plosses))]
            ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])

            # Add consistency loss
            if consistency_losses:
                closs = sum(consistency_losses) / len(consistency_losses)
                total_loss = ploss + closs
                epoch_closses.append(closs.item())
            else:
                total_loss = ploss

            # Backward pass
            model_engine.backward(total_loss)
            model_engine.step()

            # Logging
            if rank == 0:
                logdict = {
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/total_loss": total_loss.item(),
                    "train/ploss": ploss.item(),
                }
                if consistency_losses:
                    logdict["train/consistency_loss"] = closs.item()

                for i in range(len(plosses)):
                    logdict[f"train/ploss_{i}"] = plosses[i].item()
                for i in range(len(acces)):
                    logdict[f"train/acc_{i}"] = acces[i]

                wandb.log(logdict)

            for i in range(len(acces)):
                epoch_acces[i].append(acces[i])
                epoch_plosses[i].append(plosses[i].item())

        # Epoch summary
        if rank == 0:
            print(f"\n=== Epoch {epoch + 1}/{args.num_epochs} Summary ===")
            for i in range(len(epoch_acces)):
                avg_acc = sum(epoch_acces[i]) / len(epoch_acces[i])
                avg_loss = sum(epoch_plosses[i]) / len(epoch_plosses[i])
                print(f"  Position {i}: Acc={avg_acc:.4f}, Loss={avg_loss:.4f}")
                wandb.log({
                    f"epoch/acc_{i}": avg_acc,
                    f"epoch/loss_{i}": avg_loss,
                })

            if epoch_closses:
                avg_closs = sum(epoch_closses) / len(epoch_closses)
                print(f"  Consistency Loss: {avg_closs:.4f}")
                wandb.log({"epoch/consistency_loss": avg_closs})

        # Evaluation
        model.eval()
        test_acces = [[] for _ in range(model.length)]

        with torch.no_grad():
            for data in tqdm(test_loader, disable=rank != 0, desc="Evaluating"):
                outputs = model_engine(
                    input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                    loss_mask=data["loss_mask"],
                )
                if len(outputs) == 4:
                    _, _, acces, _ = outputs
                else:
                    _, _, acces = outputs

                for i in range(len(acces)):
                    test_acces[i].append(acces[i])

        # Sync and log test metrics
        for i in range(len(test_acces)):
            if test_acces[i]:
                avg_acc = sum(test_acces[i]) / len(test_acces[i])
                acc_tensor = torch.tensor(avg_acc).cuda()
                dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)

                if rank == 0:
                    print(f"  Test Position {i}: Acc={acc_tensor.item():.4f}")
                    wandb.log({f"test/acc_{i}": acc_tensor.item()})

        # Save checkpoint
        if rank == 0:
            checkpoint_dir = Path(args.output_dir) / f"epoch_{epoch + 1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model_engine.save_checkpoint(str(checkpoint_dir))
            print(f"  Saved checkpoint to {checkpoint_dir}")

            # Upload to HuggingFace
            try:
                os.system(f"huggingface-cli upload kje2952/eagle-hallu-shift {checkpoint_dir} --repo-type model")
                print(f"  Uploaded to HuggingFace")
            except Exception as e:
                print(f"  Failed to upload to HuggingFace: {e}")

    # Final summary
    if rank == 0:
        wandb.finish()
        print("\n=== Training Complete ===")
        print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
