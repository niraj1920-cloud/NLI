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

H1: Not supported.
Negation did not reduce NLI performance. XLM-RoBERTa showed no change, while mDeBERTa achieved slightly higher performance on negated examples (+1.00 pp Accuracy).

### RQ2: Does negation robustness differ between English and German?

We compared NLI performance between English and German for both models.

| Model | Language | N | Accuracy (%) | Macro F1 (%) |
|---|---|---:|---:|---:|
| XLM-RoBERTa | English | 300 | 100.00 | 100.00 |
| XLM-RoBERTa | German | 300 | 100.00 | 100.00 |
| mDeBERTa | English | 300 | 89.33 | 89.35 |
| mDeBERTa | German | 300 | 80.33 | 80.33 |

#### RQ2 Finding

- XLM-RoBERTa achieved identical performance in English and German.
- mDeBERTa achieved higher performance in English (89.33%) than in German (80.33%).
- Overall, the results show that model performance differs between English and German.

**H2: Supported.** Performance differed between English and German, particularly for mDeBERTa, while XLM-RoBERTa showed identical performance across both languages.

### RQ3: Which types of negation are most challenging for NLI models?

We analyzed model performance across different explicit lexical negation types in English and German.

| Model | Language | Negation Type | N | Accuracy (%) | Macro F1 (%) |
|---|---|---|---:|---:|---:|
| XLM-RoBERTa | English | neither | 1 | 100.00 | 100.00 |
| mDeBERTa | English | neither | 1 | 0.00 | 0.00 |
| XLM-RoBERTa | English | never | 13 | 100.00 | 100.00 |
| mDeBERTa | English | never | 13 | 92.31 | 85.19 |
| XLM-RoBERTa | English | no | 27 | 100.00 | 100.00 |
| mDeBERTa | English | no | 27 | 96.30 | 96.89 |
| XLM-RoBERTa | English | not | 102 | 100.00 | 100.00 |
| mDeBERTa | English | not | 102 | 89.22 | 89.24 |
| XLM-RoBERTa | English | nothing | 7 | 100.00 | 100.00 |
| mDeBERTa | English | nothing | 7 | 100.00 | 100.00 |
| XLM-RoBERTa | German | kein | 27 | 100.00 | 100.00 |
| mDeBERTa | German | kein | 27 | 85.19 | 83.84 |
| XLM-RoBERTa | German | nicht | 101 | 100.00 | 100.00 |
| mDeBERTa | German | nicht | 101 | 79.21 | 79.19 |
| XLM-RoBERTa | German | nichts | 8 | 100.00 | 100.00 |
| mDeBERTa | German | nichts | 8 | 100.00 | 100.00 |
| XLM-RoBERTa | German | nie | 8 | 100.00 | 100.00 |
| mDeBERTa | German | nie | 8 | 62.50 | 58.57 |
| XLM-RoBERTa | German | niemand | 5 | 100.00 | 100.00 |
| mDeBERTa | German | niemand | 5 | 80.00 | 33.33 |
| XLM-RoBERTa | German | weder | 1 | 100.00 | 100.00 |
| mDeBERTa | German | weder | 1 | 0.00 | 0.00 |

#### RQ3 Findings

- **XLM-RoBERTa** achieved 100% accuracy and Macro F1 across all observed negation types.
- **mDeBERTa** showed substantial variation across negation types.
- For English, mDeBERTa performed lowest on **"not" (89.22%)** among categories with sufficient observations.
- For German, **"nie" (62.50%)** had the lowest accuracy among categories with more than one example.
- The categories **"neither"** and **"weder"** contain only one example each and therefore should not be interpreted as reliable evidence.

**H3: Supported** Performance varies across negation types, particularly for mDeBERTa.

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

