# Experiments

---

## Planned Experiments

### Phase 0: Validation Study

**Goal**: Layer dynamics ↔ acceptance 상관관계 검증

| ID | Description | Status |
|----|-------------|--------|
| V0.1 | Eagle3 inference + logging (accepted/rejected + layer dynamics) | 🔲 Planned |
| V0.2 | Statistical analysis (t-test, effect size) | 🔲 Planned |

**Success Criteria**: p < 0.05 AND Cohen's d > 0.2

---

### Phase 1: Layer Consistency Loss

| ID | λ | Expected | Status |
|----|---|----------|--------|
| P1.1 | 0.01 | Baseline check | 🔲 Planned |
| P1.2 | 0.1 | Medium weight | 🔲 Planned |
| P1.3 | 0.5 | High weight | 🔲 Planned |
| P1.4 | 1.0 | Maximum weight | 🔲 Planned |

---

### Phase 2: Ablation Study

| ID | Consistency | Delta | Attn Entropy | Status |
|----|-------------|-------|--------------|--------|
| A (baseline) | - | - | - | 🔲 Planned |
| B | ✓ | - | - | 🔲 Planned |
| C | - | ✓ | - | 🔲 Planned |
| D | - | - | ✓ | 🔲 Planned |
| E | ✓ | ✓ | - | 🔲 Planned |
| F | ✓ | ✓ | ✓ | 🔲 Planned |

---

## Completed Experiments

(아직 없음)

---

## Baselines

| Name | Description |
|------|-------------|
| Eagle3 (vanilla) | 현재 SOTA |
| Eagle3 + longer training | 학습량 효과 분리 |
| Eagle3 + larger draft | Capacity 효과 분리 |

---

## Metrics

| Metric | Type | Target |
|--------|------|--------|
| Acceptance rate (pos 4-7) | Primary | +5% |
| Acceptance rate (overall) | Secondary | +2% |
| Wall-clock speedup | Secondary | ≥ baseline |
| Training overhead | Cost | <50% |
