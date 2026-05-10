# 🔍 Anomaly Detection from Sensor Readings

## Overview
Binary classification to detect anomalies from 5 sensor readings.
Competition: CEIP DS Lloyd Kaggle

## Dataset
- 1.64M training rows, 5 sensor features (X1–X5) + Date
- Severe class imbalance: only 0.86% anomalies

## Approach
- Feature Engineering: log transforms, interactions, temporal features
- Models: Logistic Regression, SVM, KNN, Decision Tree,
  Random Forest, XGBoost, LightGBM, CatBoost
- Imbalance: SMOTE + class_weight + threshold tuning
- Final: LightGBM ensemble, Macro F1 = 0.833

## Results
| Metric | Score |
|--------|-------|
| Macro F1 | 0.833 |
| Kaggle Rank | 23/N |

## Tech Stack
Python, Pandas, Scikit-learn, LightGBM, XGBoost, CatBoost
