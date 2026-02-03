# Infrastructure

---

## Compute

- **Platform**: Kubernetes
- **GPU**: H200
- **Storage**: NAS 사용 불가 (고장) → Pod 간 공유 저장소 없음

---

## Storage Strategy

⚠️ **중요**: NAS가 고장나서 Pod 간 저장소 공유가 불가능합니다.

### Workflow

1. **학습 완료 후 항상 HuggingFace에 업로드**
```bash
huggingface-cli upload kje2952/eagle-hallu-shift ./checkpoint --repo-type model
```

2. **다른 Pod에서 이어서 학습 시**
```bash
huggingface-cli download kje2952/eagle-hallu-shift --local-dir ./checkpoint
```

### 주의사항
- 학습 중간에 Pod이 죽으면 checkpoint 손실
- 주기적으로 HuggingFace에 업로드 (매 epoch 또는 N steps마다)

---

## Repositories

| Purpose | URL |
|---------|-----|
| **Code** | https://github.com/KilJaeeun/eagle-hallu-shift-idea |
| **Models** | https://huggingface.co/kje2952/eagle-hallu-shift |

---

## Experiment Tracking

- **WandB** for logging
- Project name: `eagle-hallushift` (예정)

---

## Environment Setup (Pod에서)

```bash
# Clone code
git clone https://github.com/KilJaeeun/eagle-hallu-shift-idea.git
cd eagle-hallu-shift-idea

# Login to services
export HF_TOKEN=<token>
export WANDB_API_KEY=<key>
huggingface-cli login --token $HF_TOKEN
wandb login

# Download checkpoint if resuming
huggingface-cli download kje2952/eagle-hallu-shift --local-dir ./checkpoint
```
