# Code Index

---

## Key Files (수정 대상)

| File | Purpose | Lines to Modify |
|------|---------|-----------------|
| `EAGLE/eagle/traineagle3/cnets.py` | Training model definition | `dataprepare()` (714-731), `forward()` (733-868) |
| `EAGLE/eagle/traineagle3/main.py` | Training loop | loss aggregation (284-286) |
| `EAGLE/eagle/model/utils.py` | Acceptance evaluation | `evaluate_posterior()` (337-415) for Phase 0 logging |

---

## File Structure

```
EAGLE/
├── eagle/
│   ├── traineagle3/           # Training code (Eagle3)
│   │   ├── cnets.py           # ⭐ Main model definition
│   │   ├── main.py            # ⭐ Training loop
│   │   └── modeling_llama_kv.py  # LLaMA wrapper
│   │
│   ├── model/                 # Inference code
│   │   ├── cnets.py           # Inference model
│   │   └── utils.py           # ⭐ evaluate_posterior (acceptance)
│   │
│   └── ...
│
└── ...
```

---

## Key Functions

### `dataprepare()` in cnets.py
```python
# Lines 714-731
def dataprepare(self, input_ids, attention_mask, loss_mask):
    outs = self.target_model(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states0 = outs.hidden_states[0]
    hidden_states1 = outs.hidden_states[1]
    hidden_states2 = outs.hidden_states[2]
    hidden_states = torch.cat((hidden_states0, hidden_states1, hidden_states2), dim=-1)
    # ... 여기에 layer dynamics 추가 필요
```

### `evaluate_posterior()` in utils.py
```python
# Lines 337-415
# 여기에서 acceptance/rejection 결정
# Phase 0: 여기에 logging 추가하여 layer dynamics 수집
```

---

## Modification Plan

### Phase 0 (Validation)
- `evaluate_posterior()` 수정: accepted/rejected tokens의 layer dynamics 로깅

### Phase 1 (Consistency Loss)
- `dataprepare()` 수정: cosine similarity 계산
- `forward()` 수정: auxiliary loss 추가
- `main.py` 수정: loss aggregation

### Phase 2 (Delta + Entropy)
- `dataprepare()` 수정: delta, attention entropy 계산
- `__init__()` 수정: delta prediction head 추가
