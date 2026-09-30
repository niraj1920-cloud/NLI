# Can NLI Models Handle Negation?
## A Cross-Lingual Study of English and German

This repository contains the code and experimental results on how explicit negation affects Natural Language Inference (NLI) performance in English and German.

## Research Questions

- **RQ1:** How does negation affect NLI performance?
- **RQ2:** Does negation robustness differ between English and German?
- **RQ3:** Which types of negation are most challenging for NLI models?

## Hypotheses

- **H1:** Negation will reduce NLI performance.
- **H2:** Negation robustness differs between English and German.
- **H3:** Performance varies by negation type.

The experiments provide mixed evidence for H1: the effect of negation depends on the model and language. H2 is supported by observed cross-lingual differences, particularly for mDeBERTa. H3 is supported with the qualification that some negation categories contain few examples.

## Dataset

The experiments use the **XNLI** dataset from Hugging Face.

- English: 5,010 test examples
- German: 5,010 test examples
- Combined dataset: 10,020 examples

Each example contains a premise, hypothesis, and NLI label.

### XNLI Labels

| Label ID | NLI label |
|---:|---|
| 0 | Entailment |
| 1 | Neutral |
| 2 | Contradiction |

### Balanced Evaluation Set

A balanced subset of **600 examples** is constructed.

For every combination of language, negation status, and NLI label, 50 examples are sampled:

**2 languages × 2 negation conditions × 3 labels × 50 examples = 600 examples**

A fixed random seed (`42`) is used for reproducibility.

## Negation Detection

The experiment focuses on **explicit lexical negation**.

### English markers

```text
not, no, never, nobody, nothing,
neither, nowhere, n't
```

### German markers

```text
nicht, kein, keine, keinen, keinem,
keiner, keines, nie, niemand, nichts,
weder, nirgendwo
```

An example is classified as **negated** if at least one relevant marker occurs in its premise or hypothesis.

Negation types are subsequently grouped into lexical categories.

This approach captures explicit lexical negation and does not capture all possible implicit or syntactic forms of negation.

## Models

### XLM-RoBERTa

```text
joeddav/xlm-roberta-large-xnli
```

### mDeBERTa

```text
MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
```

Both models are evaluated using batch inference. GPU acceleration is used when available.

## Evaluation

The following metrics are used:

- **Accuracy**
- **Macro-F1**

Performance is analyzed at four levels:

1. Overall model performance
2. Negated vs. non-negated examples
3. English vs. German
4. Individual negation types

## Main Results

### RQ1 — Effect of Negation

| Model | Negation | N | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| XLM-RoBERTa | Non-negated | 300 | 100.00 | 100.00 |
| XLM-RoBERTa | Negated | 300 | 100.00 | 100.00 |
| mDeBERTa | Non-negated | 300 | 84.33 | 84.30 |
| mDeBERTa | Negated | 300 | 85.33 | 85.29 |

The effect of negation is **not uniform**. It varies according to model and language. Therefore, H1 is **not uniformly supported**.

### RQ2 — Cross-Lingual Robustness

XLM-RoBERTa remains highly stable across English and German.

mDeBERTa shows a larger language difference, with lower performance on German examples than on English examples in the corresponding conditions.

These results provide evidence for **language-dependent NLI robustness**, particularly for mDeBERTa.

### RQ3 — Negation Types

For sufficiently represented categories, mDeBERTa shows variation across negation types.

| Language | Negation type | N | mDeBERTa Accuracy |
|---|---|---:|---:|
| English | `not` | 102 | 89.22% |
| English | `no` | 27 | 96.30% |
| English | `never` | 13 | 92.31% |
| German | `nicht` | 101 | 79.21% |
| German | `kein` | 27 | 85.19% |

XLM-RoBERTa achieved 100% accuracy across the observed negation categories in this evaluation subset.

Very small categories are interpreted cautiously because some contain only a few examples.

## Repository Structure

```text
.
├── NLI_final.ipynb
├── README.md
└── figures/
    ├── figure1_accuracy_language_negation.png
    ├── figure2_negation_types.png
    └── figure3_negation_effect.png
```

The notebook contains the complete experimental pipeline:

```text
XNLI loading
    ↓
Negation detection
    ↓
XNLI label mapping
    ↓
Balanced 600-example sampling
    ↓
Model loading
    ↓
Batch inference
    ↓
Overall evaluation
    ↓
RQ1
    ↓
RQ2
    ↓
Language × Negation
    ↓
RQ3
    ↓
Figures
```

## Requirements

```bash
pip install datasets transformers torch pandas numpy scikit-learn matplotlib
```

## Running the Experiment

1. Clone or download this repository.
2. Install the required packages.
3. Open `NLI_final.ipynb`.
4. Run the notebook from the beginning.
5. The notebook downloads the required XNLI data and pretrained models.
6. The balanced evaluation set is created using random seed `42`.
7. Both models are evaluated.
8. Tables and figures are generated automatically.

## Limitations

- Negation is detected using explicit lexical markers and may miss implicit or syntactic negation.
- The experiment uses a balanced subset of 600 XNLI test examples rather than the complete test set.
- Some negation-type categories contain relatively few examples, limiting the strength of category-level conclusions.
- Dataset overlap: XNLI-fine-tuned models may have seen parts of the evaluation data during training, limiting the validity of XNLI as an unseen test set.

