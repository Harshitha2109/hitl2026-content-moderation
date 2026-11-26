# HITL 2026 — Human-Assisted Content Moderation

Semester project for **Human-in-the-Loop Machine Learning (WiSe 26/27, HSBI)**.

Online platforms receive more user comments than moderators can check by hand. We study how human and machine effort can be combined to detect toxic comments, using the [Civil Comments](https://huggingface.co/datasets/google/civil_comments) dataset (binary label: `toxicity >= 0.5`).

## Team
- Harish Shanmugam
- Harshitha Kuppala

## Deliverables

| Deliverable | Main notebook | Content |
|---|---|---|
| Baseline | [`baseline/baseline.ipynb`](baseline/baseline.ipynb) | TF-IDF + Logistic Regression baseline, error analysis |
| **1 — Active Learning** | [**`active_learning/active_learning_experiments.ipynb`**](active_learning/active_learning_experiments.ipynb) | Random vs. uncertainty vs. diversity vs. representative sampling, 5 repetitions, learning curves, discussion |

## Repository structure
```
├── README.md
├── requirements.txt
├── baseline/
│   └── baseline.ipynb
├── active_learning/
│   ├── active_learning_experiments.ipynb   <- Deliverable 1 entry point
│   └── results/                            <- saved CSVs and plots
```

## Installation
```bash
git clone https://github.com/Harshitha2109/hitl2026-content-moderation.git
cd hitl2026-content-moderation
pip install -r requirements.txt
```

## Data
No data is stored in this repository. The notebook downloads Civil Comments automatically from the Hugging Face Hub:
```python
from datasets import load_dataset
ds = load_dataset("google/civil_comments")
```
We sample 25,000 training comments (the unlabeled pool) and 5,000 test comments with `seed=42`.

## Running the Active Learning experiment
- **Google Colab:** open `active_learning/active_learning_experiments.ipynb` in Colab (File → Open notebook → GitHub), then *Runtime → Run all*. A CPU runtime is enough.
- **Locally / VS Code:** `jupyter notebook active_learning/active_learning_experiments.ipynb`

The full experiment (5 strategies × 5 seeds × 20 rounds) takes roughly 10–20 minutes on CPU. Results are saved to `active_learning/results/al_results.csv`; set `RUN_EXPERIMENTS = False` to reload them without rerunning.
