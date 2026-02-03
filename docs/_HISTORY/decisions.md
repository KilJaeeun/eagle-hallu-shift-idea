# Key Decisions

---

## 2026-02-03

### Decision 1: Primary Goal
**Choice**: Later positions (pos 4-7)에서 acceptance rate +5%

**Rationale**:
- Eagle3의 알려진 약점이 error accumulation
- 전체 +2%보다 약점 구간 +5%가 더 인상적
- Layer dynamics가 이 문제를 직접 해결한다는 clear story

**Alternatives Considered**:
- Overall acceptance rate (+2%)
- Wall-clock speedup
- Composite goal

---

### Decision 2: Attention Entropy Conditioning
**Choice**: 유지 (Remove에서 변경)

**Rationale**:
- 초기에는 causality 문제로 제거하려 했음
- 하지만 **이전 토큰들의** attention entropy는 이미 알 수 있음
- Target이 context를 먼저 처리하므로

**Technical Note**:
- 현재 토큰(tn+1)의 entropy는 여전히 모름
- 이전 토큰들의 entropy pattern을 conditioning으로 활용

---

### Decision 3: Phase 0 Validation Study
**Choice**: 구현 전에 validation 먼저 진행

**Rationale**:
- "Layer dynamics → acceptance" 인과관계가 미검증
- 2주 투자로 30% 실패 확률을 사전에 확인 가능
- 실패 시 pivot 가능

**Success Criteria**:
- p < 0.05 AND Cohen's d > 0.2

---

### Decision 4: Target Model
**Choice**: LLaMA-2-7B-Chat

**Rationale**:
- Eagle3 공식 지원
- 빠른 iteration 가능
- 최종 검증은 70B로

---

## Template

### Decision N: [Title]
**Choice**: [What was decided]

**Rationale**: [Why this choice]

**Alternatives Considered**: [Other options]
