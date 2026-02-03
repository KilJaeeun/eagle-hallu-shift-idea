# ACTIVE - Current Work Status

**Last Updated**: 2026-02-03 23:45
**Current Phase**: Phase 0 & Phase 1 코드 완료, K8s 배포 대기

---

## Current Status

### What's Done
- [x] 연구 아이디어 정의 (Eagle3 + HalluShift layer dynamics)
- [x] 관련 논문 탐색 완료
- [x] 비판적 검토 완료 (4가지 주요 이슈 발견 및 해결책 도출)
- [x] 연구 계획 수립 (4 Phases)
- [x] GitHub 이슈에 분석 결과 기록 (#2, #3, #4, #5, #6)
- [x] Memory 시스템 구축 (docs/ 폴더)
- [x] **Phase 0 코드 완료**:
  - `scripts/phase0_validation.py` - 통계 분석
  - `scripts/phase0_inference_hook.py` - Eagle3 inference hook
- [x] **Phase 1 코드 완료**:
  - `scripts/phase1_cnets_patch.py` - Layer consistency loss patch
  - `scripts/phase1_train.py` - Training script
- [x] **K8s 배포 스크립트 완료**:
  - `k8s/phase0-job.yaml`
  - `k8s/phase1-job.yaml`
  - `k8s/secrets.yaml.template`
- [x] **Quick run script**: `run.sh`

### What's Ready to Run
```bash
# Local (mock test)
./run.sh mock

# K8s deployment
kubectl apply -f k8s/secrets.yaml  # (fill in your tokens first)
kubectl apply -f k8s/phase0-job.yaml
# After Phase 0 GO decision:
kubectl apply -f k8s/phase1-job.yaml
```

---

## Quick Commands

```bash
# Clone and setup
git clone https://github.com/KilJaeeun/eagle-hallu-shift-idea.git
cd eagle-hallu-shift-idea
chmod +x run.sh

# Mock test (no GPU)
./run.sh mock

# Phase 0 (GPU required)
./run.sh phase0

# Phase 1 (8 GPUs required)
./run.sh phase1

# K8s deployment
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/phase0-job.yaml
```

---

## Phase 1 Hyperparameter Grid

| Lambda | Description |
|--------|-------------|
| 0.01 | Minimal consistency weight |
| 0.1 | Recommended starting point |
| 0.5 | High consistency weight |
| 1.0 | Maximum consistency weight |

---

## Blockers

**GPU 필요**: 로컬에 GPU 없음. K8s H200 배포 필요.

---

## Key Decisions Made

1. **Primary Goal**: Later positions (pos 4-7)에서 acceptance +5%
2. **Attention entropy**: 사용 가능 (이전 토큰들의 entropy)
3. **Target Model**: LLaMA-2-7B-Chat

---

## Mock Data 결과 (Pipeline Validation)

| Feature | Cohen's d | Significant |
|---------|-----------|-------------|
| cos_sim_01 | 1.45 | ✓ |
| cos_sim_12 | 1.56 | ✓ |
| delta_01 | -1.57 | ✓ |
| delta_12 | -1.40 | ✓ |
| entropy_0 | -0.98 | ✓ |
| entropy_1 | -1.11 | ✓ |
| entropy_2 | -1.18 | ✓ |

**⚠️ Mock data 결과. 실제 결과는 다를 수 있음.**

---

## Quick Links

- **GitHub Issues**: https://github.com/KilJaeeun/eagle-hallu-shift-idea/issues
- **HuggingFace**: https://huggingface.co/kje2952/eagle-hallu-shift
- **계획 파일**: `/Users/kil/.claude/plans/deep-prancing-quasar.md`
