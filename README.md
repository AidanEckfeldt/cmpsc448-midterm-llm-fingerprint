# CMPSC 448 Midterm Project: Who Wrote It?

Individual project by Aidan Eckfeldt for CMPSC 448, Fall 2026.

## Project question

Can a neural text classifier identify which LLM family generated a response?

This project classifies responses from three LLM families:

- GPT: `gpt-4o-2024-05-13`
- Claude: `claude-3-5-sonnet-20240620`
- Gemini: `gemini-1.5-pro-api-0514`

The required models are a convolutional neural network (CNN) and a recurrent neural network implemented as a bidirectional LSTM.

## Dataset

The project uses a curated subset of the public LMArena `arena-human-preference-100k` dataset. Only English, single-turn, non-refusal examples are retained.

The final experimental dataset contains 900 examples:

| Family | General | Code | Math | Total |
|---|---:|---:|---:|---:|
| GPT | 100 | 100 | 100 | 300 |
| Claude | 100 | 100 | 100 | 300 |
| Gemini | 100 | 100 | 100 | 300 |

The dataset-building notebook reconstructs this subset from the public source. Rows are grouped by `question_id` when creating train/validation/test splits to reduce prompt leakage.

## Research questions

- **RQ1:** Can the LLM be identified from the response output?
- **RQ2:** Does the user prompt help? Compare input-only, output-only, and input + output.
- **RQ3:** Do fingerprints generalize to an unseen task domain?
- **RQ4:** What response characteristics distinguish the models, and how much does formatting matter?

## Final results

### RQ1: Output-only classification

| Architecture | Accuracy | Macro F1 |
|---|---:|---:|
| CNN | 78.5% | 77.5% |
| LSTM | 74.1% | 72.9% |

Chance accuracy for three balanced classes is 33.3%.

### RQ2: Input vs output vs input + output

| Architecture | Input only | Output only | Input + output |
|---|---:|---:|---:|
| CNN | 34.1% | 78.5% | 78.5% |
| LSTM | 40.7% | 74.1% | 59.3% |

The final combined condition reserves 128 prompt tokens and 384 output tokens. This replaces an earlier combined run that truncated too much model output.

### RQ3: Cross-domain accuracy

| Held-out domain | CNN | LSTM |
|---|---:|---:|
| General | 71.3% | 62.3% |
| Code | 72.7% | 64.7% |
| Math | 81.0% | 74.7% |

### RQ4: Formatting ablation

CNN output-only accuracy fell from **78.5%** to **65.9%** after common Markdown formatting was stripped, a decrease of 12.6 percentage points.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── dataset_manifest.csv
├── notebooks/
│   ├── 01_build_dataset.ipynb
│   ├── 02_full_experiments.ipynb
│   └── 03_rq2_combined_rerun.ipynb
├── src/
│   ├── build_dataset.py
│   ├── full_experiments.py
│   └── rq2_combined_rerun.py
├── results/
│   ├── rq1_cnn_confusion_matrix.png
│   ├── rq1_lstm_confusion_matrix.png
│   ├── rq2_corrected_accuracy.png
│   ├── rq2_corrected_results.csv
│   ├── rq3_cross_domain_accuracy.png
│   ├── rq3_cross_domain_results.csv
│   ├── rq4_feature_means.csv
│   ├── rq4_formatting_ablation.csv
│   └── rq4_response_length.png
└── report/
    └── CMPSC448_Midterm_Report_Aidan_Eckfeldt.pdf
```

## Reproducing the project

The easiest workflow is Google Colab.

1. Run `notebooks/01_build_dataset.ipynb` to reconstruct the curated CSV from LMArena.
2. Run `notebooks/02_full_experiments.ipynb` with the curated CSV to reproduce RQ1, base RQ2, RQ3, and RQ4.
3. Run `notebooks/03_rq2_combined_rerun.ipynb` to reproduce the corrected input + output RQ2 condition.

A GPU runtime is recommended for the LSTM experiments.

## Source dataset

LMArena, `arena-human-preference-100k` on Hugging Face.

Related paper: W.-L. Chiang et al., *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*, arXiv:2403.04132, 2024.
