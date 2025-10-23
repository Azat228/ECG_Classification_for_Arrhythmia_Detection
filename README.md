# Project CE_ARR: Final Report
<img width="420" height="208" alt="image" src="https://github.com/user-attachments/assets/7c88f19a-bdb3-411f-b29d-8c92c67b2c7f" />

## EE4C12 Machine Learning for Electrical Engineering Applications

**Azat Idayatov**  
aidayatov@tudelft.nl (Student ID: 6551505)  

**Giorgio Recchilongo**  
grecchilongo@tudelft.nl (Student ID: 6549632)

---

## Summary

This project aimed to classify different types of cardiac arrhythmias in patients through the analysis of ECG data and the creation of a machine learning model.  
Multiple model architectures were explored, including Multi-Layer Perceptrons (MLPs), Support Vector Machines (SVMs), Logistic Regression, and tree-based ensembles (Random Forest, LightGBM).  
Techniques such as class weighting and SMOTE oversampling were used to address class imbalance.  

After hyperparameter tuning, the LightGBM model emerged as the best performer, achieving:

- **Macro-F1 Score:** 0.76
- **Macro-Accuracy:** 0.98
- **Macro-Recall:** 0.75

on the test set. These results demonstrate the effectiveness of gradient boosting for this unbalanced classification task.

---

## ML Pipeline

![Pipeline of the ML algorithm development](<img width="834" height="247" alt="image" src="https://github.com/user-attachments/assets/5d6edcdd-498c-4ffb-bbdd-0179f213425d" />
)

**Pipeline Steps:**

1. **Data Preprocessing:**  
   Data acquired via ECG. QRS peak detection using Pan Tompkins Algorithm. Beat segmentation for single-beat samples. Train, validation, and test set creation. Data normalization (standard normalization fitted on training data).

2. **Feature Design:**  
   One-hot encoding of target labels to avoid model bias.

3. **Model Exploration:**  
   Baseline models trained and validated for initial comparison.

4. **Hyperparameter Tuning:**  
   Top models selected for further optimization and re-evaluated.

5. **Final Evaluation:**  
   Best model trained on the complete training set and assessed on the test set.

---

## Task 1

### Model Selection

- **Linear Models (Logistic Regression, LinearSVC):**
    - Computationally efficient baseline.
    - Test if data is linearly separable.
- **Non-Linear SVM (SVC with RBF kernel):**
    - Explores non-linear decision boundaries.
    - Computationally expensive for large datasets.
- **Tree-Based Ensembles (RandomForest, LightGBM):**
    - Test if tree-based models work well for ECG data.
    - Random Forest for robustness; LightGBM for efficiency.
- **Multi-Layer Perceptron (MLP):**
    - Deep learning baseline.
    - Class imbalance addressed via weighting, SMOTE oversampling, and undersampling.

### Performance Metrics

- **Emphasis on minimizing false negatives** due to clinical consequences.
- **F1 score** is the primary metric (balances precision and recall).
- **Recall** is used as a secondary metric if F1 is similar.

### Handling the Imbalanced Dataset

- Dataset is highly imbalanced; some classes are rare.
- Metrics are computed using **macro averaging** (all classes equally important).

![Sample class distribution](<img width="695" height="348" alt="image" src="https://github.com/user-attachments/assets/7071101c-107b-4924-aca0-551809595daf" />
)

---

### Results

| Model                        | Recall (Macro) | F1 (Macro) |
|------------------------------|:--------------:|:----------:|
| Simple MLP                   | 0.37           | 0.38       |
| Weighted MLP                 | 0.67           | 0.30       |
| SMOTE MLP                    | 0.62           | 0.45       |
| Downsampled MLP              | 0.76           | 0.52       |
| Random Forest                | 0.73           | 0.57       |
| Gradient Boosting (LightGBM) | 0.68           | 0.72       |
| SVC RBF                      | 0.86           | 0.61       |
| Downsampled SVC RBF          | 0.79           | 0.49       |
| SVC Linear Kernel            | 0.59           | 0.30       |
| Logistic Regression          | 0.69           | 0.35       |

- **Linear models** (Linear SVC, Logistic Regression) performed poorly (problem is not linearly separable).
- **SVC (RBF kernel)** did well, but was slow.
- **Tree-based models** (Random Forest, LightGBM) were efficient and accurate.
- **MLPs** improved with class balancing and SMOTE, with downsampling yielding best recall/F1 among MLPs.

---

#### Confusion Matrices

Simple MLP:
![Confusion matrix of the simple MLP model](Figs/conf-mat-simp-mlp.png)

Random Forest:
![Confusion matrix of Random Forest model](Figs/conf-mat-rand-for.png)

---

## Task 2: Model Selection and Optimization

The following models were selected for deeper analysis and optimization:

1. **Downsampled MLP**
2. **Random Forest**
3. **Gradient Boosting (LightGBM)**

### Performance Before and After Optimization

**Before Optimization:**

| Model                        | Accuracy | Recall (Macro) | F1 (Macro) |
|------------------------------|:--------:|:--------------:|:----------:|
| Downsampled MLP              | 0.90     | 0.80           | 0.55       |
| Random Forest                | 0.93     | 0.55           | 0.61       |
| Gradient Boosting (LightGBM) | 0.94     | 0.69           | 0.68       |

**After Optimization:**

| Model                        | Accuracy | Recall (Macro) | F1 (Macro) | Recall (Weighted) | F1 (Weighted) |
|------------------------------|:--------:|:--------------:|:----------:|:-----------------:|:-------------:|
| Downsampled MLP              | 0.88     | 0.76           | 0.50       | 0.85              | 0.88          |
| Random Forest                | 0.97     | 0.61           | 0.66       | 0.98              | 0.97          |
| Gradient Boosting (LightGBM) | 0.98     | 0.71           | 0.72       | 0.98              | 0.98          |

---

### Model Summaries

#### Downsampled MLP

- **Approach:** Multilayer Perceptron (MLP) with downsampling (majority class reduced to 10%).
- **SMOTE** used to address class imbalance.
- **Best configuration:**  
  - `hidden_layer_sizes=(20, 10)`
  - `activation='relu'`
  - `max_iter=50`
  - `random_state=42`
- **Best metrics (pre-optimization):**  
  - Accuracy: 0.90  
  - Macro recall: 0.80  
  - Macro F1: 0.55

Confusion matrix:  
![Confusion matrix of Downsampled MLP](Figs/conf_mat_downsamp.png)

---

#### Random Forest

- **Approach:** Ensemble of decision trees.
- **Best configuration:**  
  - `class_weight='balanced'`
  - `n_estimators=100`
  - `random_state=42`
- **Best metrics (after optimization):**  
  - Accuracy: 0.97  
  - Macro recall: 0.61  
  - Macro F1: 0.66

Confusion matrix:  
![Confusion matrix of Random Forest (after optimization)](Figs/Forest_search.png)

---

#### Gradient Boosting (LightGBM)

- **Approach:** Gradient boosting with LightGBM.
- **Best configuration:**  
  - `class_weight='balanced'`
  - `n_estimators=150`
  - `random_state=42`
  - `n_jobs=1`
- **Best metrics (after optimization):**  
  - Accuracy: 0.98  
  - Macro recall: 0.71  
  - Macro F1: 0.72  
  - Weighted recall/F1: 0.98

Confusion matrix:  
![Confusion matrix of Gradient Boosting (LightGBM)](Figs/Gradient_boosting.png)

---

## Conclusion

In this project, we systematically explored and compared a variety of machine learning models for classifying arrhythmias from ECG data. The primary challenge was a strong class imbalance: "no disease" class dominated, resulting in high accuracy but poor recall for disease classes.

- **Linear models** performed poorly, confirming the problem's non-linearity.
- **Tree-based ensembles** (Random Forest, LightGBM) and **downsampled MLPs** performed strongly.
- **LightGBM** was the top performer after optimization:  
  - Macro-F1: 0.72  
  - Macro-recall: 0.71  
  - Weighted metrics: 0.98

To conclude, addressing class imbalance (via SMOTE, weighting, and downsampling) and using advanced models (like gradient boosting) are crucial for effective disease detection in unbalanced datasets.

---
