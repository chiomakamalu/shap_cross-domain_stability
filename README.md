# SHAP Cross-Domain Stability for Sentiment Classification

This repository contains the code and exported artifacts used to evaluate the cross-domain stability of SHAP explanations for sentiment classification. The experiments compare explanation behavior across multiple sentiment datasets and report predictive performance, cross-domain stability, and within-domain stability results.

## Repository Structure

- `code/pipeline.py`  
  Script for dataset loading, preprocessing, model inference, SHAP computation, aggregation, stability analysis, and figure generation.

- `results/`  
  Exported CSV files used for analysis and reporting.

- `images/`  
  Generated figures used to summarize the experimental results.

- `requirements.txt`  
  Python dependencies required to run the pipeline.

## Included Results

The repository includes the following main result files:

- `binary_label_counts.csv`  
  Dataset label counts after binary label harmonization.

- `predictive_metrics_with_ci.csv`  
  Accuracy and macro-F1 with bootstrap confidence intervals.

- `jaccard_k_full_small.csv`  
  Cross-domain top-k token overlap results.

- `spearman_full_small.csv`  
  Cross-domain rank-correlation results over shared tokens.

- `split_half_stability_full_small.csv`  
  Within-domain split-half stability results.

## Included Figures

The repository includes the following figures:

- `accuracy_ci.png`
- `jaccard_heatmap.png`
- `jaccard_k.png`

## Experimental Pipeline

The pipeline follows these main steps:

1. Load English and Nigerian sentiment datasets.
2. Standardize dataset schemas into a common format.
3. Harmonize labels into binary sentiment classes.
4. Create balanced evaluation subsets.
5. Run inference using fixed pretrained sentiment classifiers.
6. Compute predictive metrics and bootstrap confidence intervals.
7. Sample subsets for SHAP analysis.
8. Compute token-level SHAP values and save outputs incrementally.
9. Aggregate global token importance scores.
10. Compute explanation stability using Jaccard overlap, Spearman correlation, and split-half stability.
11. Export result tables and figures.

## Models

The experiments use the following pretrained models:

- `distilbert-base-uncased-finetuned-sst-2-english`
- `Davlan/naija-twitter-sentiment-afriberta-large`
- `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` (fallback)

## Reproducing the Results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python code/pipeline.py
```

## Output Locations

- CSV result tables → `results/`
- Figures → `images/`

## Reproducibility Notes

- Random seed = 42
- Balanced evaluation subsets
- SHAP computed on sampled data for efficiency
- SHAP stage is the main runtime cost

## Purpose

This repository supports reproducibility of experiments and findings on cross-domain explanation stability.
