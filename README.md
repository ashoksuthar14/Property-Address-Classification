# Property Address Classification System

A robust machine learning solution for categorizing property addresses into predefined categories using a hybrid hierarchical approach with weak supervision and deep learning fallback mechanisms.

## 📌 Problem Statement

Build a classifier that accurately categorizes unseen property addresses into one of five predefined categories:

- **flat**
- **houseorplot**
- **landparcel**
- **commercial unit**
- **others**

The input consists of raw text property addresses that are often noisy, inconsistent, and written in legal or registry-style formats. The challenge is to design a robust, reproducible ML solution that generalizes well to unseen data.

---

## 📂 Repository Structure

```
.
├── data/
│   ├── train.csv
│   ├── validation.csv
│
├── notebooks/
│   ├── exploration_and_modeling.ipynb
│
├── best_model/
│   ├── final_property_classifier.pkl
│
├── src/
│   ├── feature_engineering.py
│   ├── hierarchical_models.py
│   ├── cnn_fallback.py
│   ├── final_model_wrapper.py
│
├── requirements.txt
└── README.md
```

### Directory Overview

- **data/** - Training and validation datasets
- **best_model/** - Final saved model used for inference
- **src/** - Modular code for features, models, and final pipeline
- **notebooks/** - Experiments, evaluation, and visualizations

---

## 🧠 Solution Architecture

Instead of relying on a single flat classifier, the final solution uses a **multi-stage hybrid system** designed to address the ambiguity and noise present in real-world address data.

### System Pipeline

```
Raw Address
    ↓
Text Cleaning
    ↓
Weak Supervision Features (domain signals)
    ↓
Word + Character TF-IDF
    ↓
Hierarchical ML Classification
    ↓
Confidence-based Routing
    ↓
Character-CNN Fallback (for ambiguous cases)
    ↓
Final Prediction
```

This approach combines traditional ML with lightweight deep learning, prioritizing clarity, generalization, and correctness over brute-force complexity.

---

## 🔍 Key Design Decisions

### 1. Weak Supervision (Domain Signals)

Extract simple but powerful signals from the text:

- **Area units**: acre, hectare, gunta
- **Legal terms**: survey, khata, patta
- **Building indicators**: flat, tower, floor
- **Numeric density**: common in land records

These signals help reduce confusion between semantically similar classes (e.g., `houseorplot` vs `landparcel`).

### 2. Hierarchical Classification

Instead of directly predicting 5 classes, the task is decomposed into stages:

**Stage 1:** Residential vs Non-Residential

**Stage 2:**
- Residential → {flat, houseorplot, landparcel}
- Non-residential → {commercial unit, others}

This significantly reduces class overlap and improves recall for minority classes.

### 3. Confidence-Based Routing

- If the model is **confident**, the ML prediction is accepted
- If confidence is **low**, the input is routed to a **character-level CNN**, which better captures formatting patterns and registry-style text

This mimics how real-world production systems handle ambiguity.

---

## 📊 Evaluation Metrics

The model is evaluated on a held-out validation set using:

- Accuracy
- Precision / Recall / F1-score
- **Macro F1-score**
- Confusion Matrix

### Why Macro F1?

Macro F1 gives equal importance to each class, regardless of frequency. This is crucial because:

- Classes like `landparcel` and `others` are underrepresented
- Accuracy alone can hide poor performance on minority classes
- Macro F1 reflects how well the model performs across **all categories**, not just dominant ones

---

## 📈 Performance Results

### Validation Set Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | ~92–93% |
| **Macro F1** | ~0.94 |
| **Flat Recall** | ~91% |
| **Commercial Unit Recall** | ~91% |
| **Others Recall** | ~85% |
| **Landparcel Recall** | ~80% |

### Key Observations

- Strong diagonal dominance in confusion matrix indicates good overall classification
- Remaining errors are concentrated between:
  - `houseorplot` ↔ `landparcel`
  - `houseorplot` ↔ `others`
- These cases are inherently ambiguous, even for humans, due to lack of explicit usage information in the address text

### Interpretation

The model achieves **near-ceiling performance** for this dataset. Remaining errors are **data-limited, not model-limited**. Many misclassified addresses do not explicitly encode property usage, making perfect classification impossible without additional metadata.

Further improvements would require external signals such as:
- Zoning data
- Land-use codes
- Registry metadata

---

## 💾 Model Artifacts

The final model is saved as:

```
best_model/final_property_classifier.pkl
```

This single file encapsulates:
- Preprocessing logic
- Feature engineering
- Hierarchical ML models
- CNN fallback
- Confidence routing

---

## ▶️ Usage

### Installation

```bash
pip install -r requirements.txt
```

### Inference Example

```python
import joblib

# Load the trained model
model = joblib.load("best_model/final_property_classifier.pkl")

# Sample addresses
addresses = [
    "Plot No 23, Survey No 145, Village XYZ",
    "Flat No 504, Tower B, Residential Complex",
    "Commercial shop bearing No 12, Market Road"
]

# Make predictions
predictions = model.predict(addresses)
print(predictions)
```

### Expected Output

```python
['landparcel', 'flat', 'commercial unit']
```

---

## 🔬 Methodology Highlights

- **No prompt engineering or LLM-only shortcuts were used**
- The solution emphasizes sound reasoning, robustness, and reproducibility
- The design mirrors real-world ML system patterns
- The final performance is both strong and honest

---

## 🛠️ Technical Stack

- **Traditional ML**: Scikit-learn (TF-IDF, Logistic Regression, Random Forest)
- **Deep Learning**: Character-level CNN for fallback cases
- **Feature Engineering**: Custom domain-specific weak supervision
- **Model Persistence**: Joblib for serialization

---

## 📝 Future Improvements

- Incorporate external data sources (zoning maps, land registries)
- Experiment with transformer-based models for better context understanding
- Implement active learning for edge cases
- Deploy as a REST API for real-time inference

---

## 👤 Author

**Ashok Suthar**  
AI/ML Intern Candidate  
Focused on building practical, reliable ML systems for real-world data



---

**Note**: This system is designed for educational and demonstration purposes. For production deployment, additional validation, monitoring, and error handling should be implemented.
