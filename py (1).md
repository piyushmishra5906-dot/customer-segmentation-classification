# Multi-Class Customer Segmentation & Classification

## Overview
This repository contains an end-to-end machine learning pipeline to classify potential customers into 4 distinct segments (**A, B, C, D**) based on demographic and behavioral attributes.

---

## Dataset Description
- **Train.csv**: 8,068 rows containing demographic features and ground-truth `Segmentation` labels.
- **Test.csv**: 2,627 rows containing target customer attributes for prediction.
- **Key Features**: `Age`, `Gender`, `Ever_Married`, `Graduated`, `Profession`, `Work_Experience`, `Spending_Score`, `Family_Size`, `Var_1`.

---

## Modeling Approach
1. **Preprocessing & Missing Value Handling**: Imputed categorical missing values with dedicated categories and numerical missing values (`Work_Experience`, `Family_Size`) with medians.
2. **Feature Engineering**: Engineered `Age_Group` bins, `Family_Size_Large`, and `Is_Alone` binary indicators.
3. **Model Selection**: Trained a **HistGradientBoostingClassifier** within a 5-Fold Stratified Cross-Validation framework.
4. **Evaluation**: Evaluated performance using Stratified 5-Fold Out-of-Fold Accuracy and Macro F1-Score.

---

## Predicted Test Distribution

| Segment | Count | Proportion |
| :--- | :--- | :--- |
| **Segment D** | 851 | 32.4% |
| **Segment A** | 657 | 25.0% |
| **Segment C** | 619 | 23.6% |
| **Segment B** | 500 | 19.0% |

---

## Instructions to Run Locally

1. **Clone repository**:
   ```bash
   git clone [https://github.com/your-username/customer-segmentation-classification.git](https://github.com/your-username/customer-segmentation-classification.git)
   cd customer-segmentation-classification