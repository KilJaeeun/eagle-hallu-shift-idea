"""
Phase 2: Training with Delta Prediction + Attention Entropy
===========================================================
Extended training script with all three auxiliary losses:
1. Layer consistency loss (Phase 1)
2. Layer delta prediction loss (Phase 2)
3. Attention entropy conditioning (Phase 2)

Usage:
    deepspeed --num_gpus=8 scripts/phase2_train.py \
        --base_model meta-llama/Llama-2-7b-chat-hf \
        --data_path data/sharegpt_train.json \
        --lambda_consistency 0.1 \
        --lambda_delta 0.1 \
        --use_attention_entropy \
        --output_dir checkpoints/phase2
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

from phase2_delta_entropy_patch import patch_eagle3_for_phase2


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Delta + Entropy Training")

    # Model arguments
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--draft_model", type=str, default=None)

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, default=None)

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Phase 2 specific arguments
    parser.add_argument("--lambda_consistency", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=0.1)
    parser.add_argument("--use_attention_entropy", action="store_true")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase2")
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

    set_seed(42)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(f"[Rank {rank}] Initializing Phase 2 training...")
    print(f"[Rank {rank}] Lambda consistency: {args.lambda_consistency}")
    print(f"[Rank {rank}] Lambda delta: {args.lambda_delta}")
    print(f"[Rank {rank}] Use attention entropy: {args.use_attention_entropy}")

    # Initialize wandb
    if rank == 0:
        run_name = args.wandb_run_name or f"phase2_c{args.lambda_consistency}_d{args.lambda_delta}_e{int(args.use_attention_entropy)}"
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

    # Apply Phase 2 patch
    model = patch_eagle3_for_phase2(
        model,
        lambda_consistency=args.lambda_consistency,
        lambda_delta=args.lambda_delta,
        use_attention_entropy=args.use_attention_entropy,
    )

    # Load dataset
    print(f"[Rank {rank}] Loading dataset...")
    from eagle.traineagle3.getdata import get_dataset
    train_dataset = get_dataset(args.data_path, tokenizer)

    if args.test_data_path:
        test_dataset = get_dataset(args.test_data_path, tokenizer)
    else:
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
    print(f"[Rank {rank}] Starting Phase 2 training...")

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        epoch_plosses = [[] for _ in range(model.length)]
        epoch_consistency_losses = []
        epoch_delta_losses = []
        epoch_acces = [[] for _ in range(model.length)]

        for batch_idx, data in enumerate(tqdm(train_loader, disable=rank != 0)):
            model_engine.zero_grad()

            # Forward pass
            outputs = model_engine(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                loss_mask=data["loss_mask"],
            )

            # Unpack outputs
            if len(outputs) == 4:
                plosses, vlosses, acces, phase2_losses = outputs
            else:
                plosses, vlosses, acces = outputs
                phase2_losses = []

            # Compute prediction loss
            ploss_weight = [0.8 ** i for i in range(len(plosses))]
            ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])

            # Add Phase 2 losses
            total_loss = ploss
            if phase2_losses:
                consistency_loss = sum([p['consistency_loss'] for p in phase2_losses]) / len(phase2_losses)
                delta_loss = sum([p['delta_loss'] for p in phase2_losses]) / len(phase2_losses)
                total_loss = ploss + consistency_loss + delta_loss
                epoch_consistency_losses.append(consistency_loss.item())
                epoch_delta_losses.append(delta_loss.item())

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
                if phase2_losses:
                    logdict["train/consistency_loss"] = consistency_loss.item()
                    logdict["train/delta_loss"] = delta_loss.item()

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

            # Per-position metrics
            for i in range(len(epoch_acces)):
                avg_acc = sum(epoch_acces[i]) / len(epoch_acces[i])
                avg_loss = sum(epoch_plosses[i]) / len(epoch_plosses[i])
                print(f"  Position {i}: Acc={avg_acc:.4f}, Loss={avg_loss:.4f}")
                wandb.log({f"epoch/acc_{i}": avg_acc, f"epoch/loss_{i}": avg_loss})

            # Phase 2 losses
            if epoch_consistency_losses:
                avg_closs = sum(epoch_consistency_losses) / len(epoch_consistency_losses)
                avg_dloss = sum(epoch_delta_losses) / len(epoch_delta_losses)
                print(f"  Consistency Loss: {avg_closs:.4f}")
                print(f"  Delta Loss: {avg_dloss:.4f}")
                wandb.log({
                    "epoch/consistency_loss": avg_closs,
                    "epoch/delta_loss": avg_dloss,
                })

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
            except Exception as e:
                print(f"  Failed to upload: {e}")

    if rank == 0:
        wandb.finish()
        print("\n=== Phase 2 Training Complete ===")


if __name__ == "__main__":
    main()
