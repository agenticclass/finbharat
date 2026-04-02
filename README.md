# FinBharat

**How Reliable is AI Financial Advice for India's 500 Million Non-English Speakers?**

FinBharat is a benchmark for evaluating LLM financial advisory reliability across six Indian languages: English, Hindi, Hinglish, Telugu, Bengali, and Tamil.

## Key Findings

- **Guardrail language gap**: Safety compliance drops from 90% (English) to 26% (Hindi). Models that refuse unauthorized investment advice in English freely give it in Indian languages.
- **Bengali catastrophe**: Qwen3.5 drops from 44% accuracy in English to under 10% in Bengali — a 34 percentage point collapse.
- **Hinglish advantage**: Code-switched Hindi-English consistently outperforms pure Hindi by 1-4% across all models.

## Dataset

180 expert-verified questions across 10 categories:

| Category | Questions | Topics |
|----------|-----------|--------|
| Income Tax | 25 | Section 80C, old/new regime, LTCG, STCG, HRA, NRI, crypto |
| Mutual Funds | 20 | ELSS, SIP, debt fund changes, exit loads |
| Stock Market | 20 | T+1, STT, F&O rules, corporate actions |
| Banking & RBI | 20 | Repo rate, FD, UPI, NBFC, digital lending |
| Recent Changes | 20 | Budget 2024-2025, SEBI reforms, RBI rate cuts |
| SEBI Regulations | 15 | Insider trading, TER, IPO, finfluencers |
| Insurance | 15 | Term vs endowment, PED, claim process |
| Retirement | 15 | EPF, PPF, NPS, gratuity |
| Guardrails | 15 | Refusal, compliance, prompt injection |
| Scenarios | 15 | Complex real-world multi-step cases |

Each question available in **6 languages** with verified ground truth from official Indian regulatory sources.

## Quick Start

```python
import json

# Load the benchmark
questions = []
for category in ['income_tax', 'mutual_funds', 'stock_market', 'banking_rbi',
                 'recent_changes', 'sebi_regulations', 'insurance', 'retirement',
                 'guardrails', 'scenarios']:
    with open(f'finbharat/data/questions_{category}.json') as f:
        questions.extend(json.load(f))

# Each question has:
# - question_en, question_hi, question_hinglish, question_te, question_bn, question_ta
# - answer (verified ground truth)
# - source (official regulatory reference)
# - common_llm_errors (known failure patterns)
# - danger_level (high/medium/low)
```

## Models Evaluated

| Model | Origin | Avg Accuracy |
|-------|--------|-------------|
| Mistral Large 3 | France | 42.7% |
| DeepSeek V3.2 | China | 36.0% |
| Qwen3.5-397B | China | 34.1% |
| Qwen3.5-27B | China | 33.0% |
| GPT-OSS-120B | USA | 28.3% |

## Requirements

```
pip install requests
```

## Citation

```bibtex
@article{panuganti2026finbharat,
  title={FinBharat: How Reliable is AI Financial Advice for India's 500 Million Non-English Speakers?},
  author={Panuganti, Rajkiran},
  year={2026}
}
```

## License

Dataset: CC-BY 4.0. Code: MIT.
