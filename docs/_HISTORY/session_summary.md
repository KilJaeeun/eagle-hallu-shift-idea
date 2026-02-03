# Session Summaries

---

## Session 2026-02-03 (Initial Planning)

### What Was Done
1. **연구 아이디어 정의**
   - Eagle3 + HalluShift layer dynamics 통합
   - 3가지 축: Layer delta, Attention entropy, Layer consistency

2. **관련 논문 탐색**
   - Eagle3 구현체 분석 (shallow layers 0,1,2 사용)
   - HalluShift features 분석 (Wasserstein, cosine sim, entropy)
   - Speculative decoding 최신 동향 (POSS, DREAM, AdaSD 등)

3. **비판적 검토 (NeurIPS reviewer 관점)**
   - Issue 1: Layer mismatch (shallow vs deep)
   - Issue 2: Causality → **Resolved** (이전 토큰 entropy 사용 가능)
   - Issue 3: Draft architecture constraint
   - Issue 4: Missing causal chain

4. **연구 계획 수립**
   - Phase 0: Validation (2주)
   - Phase 1: Consistency loss (3주)
   - Phase 2: Delta + Entropy (3주)
   - Phase 3: Deeper layers (4주, optional)

5. **GitHub 이슈 작성**
   - #2: Research Goal
   - #3: Eagle3 Analysis
   - #4: HalluShift Analysis
   - #5: Critical Review

6. **Memory 시스템 구축**
   - docs/ 폴더 구조 생성
   - 핵심 문서 작성

### Key Decisions
- Primary goal: Later positions +5%
- Attention entropy: 유지 (이전 토큰 entropy 사용)
- Target model: LLaMA-2-7B-Chat

### Next Steps
1. Phase 0 validation study 코드 구현
2. Eagle3 inference에 logging 추가
3. Accepted/Rejected tokens의 layer dynamics 수집

### Artifacts Created
- Plan file: `/Users/kil/.claude/plans/deep-prancing-quasar.md`
- GitHub Issues: #2, #3, #4, #5
- Memory docs: `docs/_HOT/`, `docs/_ENV/`, etc.
