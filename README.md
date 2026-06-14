# 🛍️ Product Return Risk Prediction

A binary classification project predicting whether a purchase will be returned, built on retail transaction data using Logistic Regression with cross-validation and feature importance analysis.

---

## 📂 Dataset

**Source:** `Export_Product_Return_Data.csv` (local)

| Property | Value |
|---|---|
| Rows | 5,000 |
| Columns (raw) | 11 |
| Features used | 9 |
| Target | `high_return_risk` (Yes / No) |

**Features:** Age, Gender, State, Category, Brand, Quantity, Price, Discount, Product Rating

**Target Distribution:**

| Class | Count | Share |
|---|---|---|
| Yes (High Risk) | 2,546 | 50.9% |
| No Risk | 2,454 | 49.1% |

The dataset is near-perfectly balanced — no oversampling or class weighting required.

---

## 🛠️ Tech Stack

- **Python 3.x**
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — EDA visualizations
- `scikit-learn` — preprocessing, modeling, evaluation

---

## 🔍 Exploratory Data Analysis

### Visualizations Produced
- **Target distribution** bar chart — confirmed balanced classes
- **Numeric feature histograms** — Age, Quantity, Price, Discount, Product Rating
- **Return rate by Category** — top 10 category/risk combinations
- **Price vs Return Risk** boxplot — minimal separation between classes
- **Discount vs Return Risk** boxplot — similarly flat across classes

These plots collectively confirmed that basic transactional features alone don't create clean decision boundaries for return prediction.

---

## 🔧 Preprocessing

| Step | Detail |
|---|---|
| Drop junk column | `Unnamed: 0` (timestamp artifact) dropped |
| Label Encoding | `Category`, `State`, `Brand`, `Gender` encoded with `LabelEncoder` |
| Target Encoding | `high_return_risk` → `No=0`, `Yes=1` via `LabelEncoder` |
| Feature Scaling | `StandardScaler` applied to all 9 features (fit on train, transform on test) |
| Train/Test Split | 80/20 with `stratify=y` and `random_state=10` |

> No null values were present in the dataset — zero imputation needed.

---

## 🧠 Model — Logistic Regression

```python
LogisticRegression(max_iter=1000, random_state=10)
```

Chosen for interpretability: coefficients directly reveal which features push toward high return risk.

---

## 📊 Results

### 5-Fold Cross-Validation (on training set)

| Fold | Accuracy |
|---|---|
| 1 | 0.5413 |
| 2 | 0.5100 |
| 3 | 0.5162 |
| 4 | 0.5325 |
| 5 | 0.4925 |
| **Mean** | **0.5185** |
| **Std Dev** | **0.0171** |

### Test Set Performance

| Metric | Value |
|---|---|
| Accuracy | 50.10% |
| ROC-AUC | 0.4991 |

### Classification Report

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Risk | 0.49 | 0.39 | 0.43 |
| High Risk | 0.51 | 0.61 | 0.55 |

---

## 📐 Feature Coefficients

| Feature | Coefficient | Direction |
|---|---|---|
| Product Rating | +0.079 | ↑ return risk |
| Discount | +0.069 | ↑ return risk |
| Category | +0.026 | ↑ return risk |
| Gender | +0.020 | ↑ return risk |
| Brand | +0.013 | ↑ return risk |
| Price | -0.026 | ↓ return risk |
| State | -0.028 | ↓ return risk |
| Quantity | -0.039 | ↓ return risk |
| Age | -0.051 | ↓ return risk |

All coefficients are very small and close to zero, which explains the near-random accuracy — no feature is a strong predictor of return risk in this dataset.

---

## 💡 Key Insights

- **~50% accuracy = essentially random** for a balanced dataset. The model cannot find a signal in these features.
- **Discount and Product Rating** are the weakest positive predictors — items with higher discounts and paradoxically low ratings show marginally higher return risk.
- **Age** is the strongest (still weak) negative predictor — older customers slightly less likely to return.
- **Root cause:** Return behavior is driven by factors absent from this dataset — product quality, sizing accuracy, delivery condition, misleading listings.

---

## 📁 Project Structure

```
📦 product-return-risk/
├── Product_Return_Risk_Analysis.ipynb   # Main notebook
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/<your-username>/product-return-risk.git
cd product-return-risk

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Place the dataset
# Add Export_Product_Return_Data.csv to the project root (or update path in cell 2)

# Launch notebook
jupyter notebook Product_Return_Risk_Analysis.ipynb
```

---

## 🔭 Next Steps

1. **Richer features** — return reason, delivery time, product reviews, past return history per customer
2. **Try tree-based models** — Random Forest or XGBoost may capture non-linear interactions
3. **Category-level targeting** — use model outputs as a risk-flagging signal for high-discount categories before dispatch
