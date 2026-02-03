# QUICKSTART - Project Context

**Project**: Eagle-HalluShift
**Goal**: NeurIPS 2026 submission

---

## One-Line Summary

> Eagle3의 multi-layer approach에 HalluShift의 layer dynamics 정보를 통합하여 **later position acceptance rate +5%** 달성

---

## Core Idea (3 Axes)

1. **Layer Delta Prediction**: Draft가 target의 레이어 간 변화량 예측
2. **Attention Entropy Conditioning**: 이전 토큰들의 target attention entropy를 draft 입력에 추가
3. **Layer Consistency Loss**: Draft의 input→output consistency가 target의 layer consistency와 유사하도록

---

## Primary Goal

> **Later positions (pos 4-7)에서 acceptance rate +5%** vs Eagle3

Eagle3의 약점인 **error accumulation**을 layer dynamics supervision으로 해결

---

## Research Phases

| Phase | Duration | Goal |
|-------|----------|------|
| **0** | 2주 | Validation (GO/NO-GO) |
| **1** | 3주 | Layer consistency loss |
| **2** | 3주 | Delta + Attention entropy |
| **3** | 4주 | (Optional) Deeper layers |

---

## Key Files

```
EAGLE/eagle/traineagle3/cnets.py    # Training model (수정 대상)
EAGLE/eagle/traineagle3/main.py     # Training loop (수정 대상)
EAGLE/eagle/model/utils.py          # Acceptance evaluation (Phase 0 로깅)
```

---

## Quick Commands

```bash
# 학습 후 모델 업로드
huggingface-cli upload kje2952/eagle-hallu-shift ./checkpoint --repo-type model

# 모델 다운로드
huggingface-cli download kje2952/eagle-hallu-shift --local-dir ./checkpoint
```

---

## Current Status

**Phase 0 준비 중** - Validation study 코드 구현 필요
