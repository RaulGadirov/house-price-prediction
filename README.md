# 🏠 House Price Prediction

A regression project that predicts house sale prices from structural and locational features (area, rooms, amenities, etc.) using **Linear Regression** and **Random Forest Regressor**, with hyperparameter tuning performed via **Bayesian Optimization (BayesSearchCV)**.

> Project notebook: `RaulProject.ipynb` · Dataset: `Housing.csv`

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Data Preprocessing & Feature Engineering](#-data-preprocessing--feature-engineering)
- [Train/Test Split & Scaling](#-traintest-split--scaling)
- [Modeling](#-modeling)
- [Hyperparameter Tuning — Bayesian Search](#-hyperparameter-tuning--bayesian-search)
- [Model Evaluation](#-model-evaluation)
- [Results](#-results)
- [Conclusion](#-conclusion)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Possible Improvements](#-possible-improvements)

---

## 🎯 Project Overview

This is a **supervised regression problem**: given a set of house attributes, the goal is to predict the sale **price** as accurately as possible. Two model families were trained and compared — a simple linear baseline and a non-linear ensemble model — to see which one generalizes better on this dataset.

---

## 📊 Dataset

The dataset (`Housing.csv`) contains **545 records** and **13 columns** describing residential properties:

| Column | Type | Description |
|---|---|---|
| `price` | numeric (target) | Sale price of the house |
| `area` | numeric | Total plot/floor area (sq. ft.) |
| `bedrooms` | numeric | Number of bedrooms |
| `bathrooms` | numeric | Number of bathrooms |
| `stories` → `floors` | numeric | Number of floors (renamed during preprocessing) |
| `mainroad` | categorical (yes/no) | Whether the house faces a main road |
| `guestroom` | categorical (yes/no) | Whether a guest room is present |
| `basement` | categorical (yes/no) | Whether a basement is present |
| `hotwaterheating` | categorical (yes/no) | Whether hot water heating is installed |
| `airconditioning` | categorical (yes/no) | Whether AC is installed |
| `parking` | numeric | Number of parking spots |
| `prefarea` | categorical (yes/no) | Located in a "preferred" area |
| `furnishingstatus` | categorical | `furnished` / `semi-furnished` / `unfurnished` |

No missing values (`df.isnull().sum() == 0`) and no duplicated rows (`df.duplicated().sum() == 0`) were found — the data was clean out of the box, which simplified preprocessing.

---

## 🔍 Exploratory Data Analysis (EDA)

The following steps were used to understand the structure and relationships in the data before modeling:

1. **Structural inspection** — `df.head()`, `df.info()`, and `df.describe()` to check data types, ranges, and basic statistics (with float formatting applied for readability).
2. **Categorical value inspection** — iterated over all categorical columns (excluding `price` and `area`) and printed their unique values to confirm consistent yes/no encoding and clean category labels.
3. **Missing values & duplicates check** — confirmed the dataset required no cleaning for nulls or duplicates.
4. **Area vs. Price** — a scatter plot of `area` vs. `price`, colored by `furnishingstatus`, overlaid with a linear regression fit line (`sns.regplot`) to visualize the overall price trend as area increases.
5. **Bedrooms vs. Price (grouped by area)** — a bar plot comparing average price across bedroom counts, segmented by area, to check whether room count or square footage has a stronger relationship with price.

**Key EDA insight:**
> Across the bar plot, houses with **3 bedrooms but a large area** consistently show the **highest average price** — higher than houses with 4 or more bedrooms but a smaller area. In other words, adding bedrooms does **not** reliably increase price; what actually drives price up is **area**, regardless of how many bedrooms are squeezed into it.
>
> **Practical implication:** bedroom count should not be treated as the primary pricing lever. A more sensible pricing strategy is to price houses on a **per-area basis** (e.g., a fixed rate per m² / per sq. ft.), since price scales much more consistently with area than with room count. This observation also justified giving `area` strong weight as a feature in both models, rather than relying on `bedrooms` as a proxy for house size/value.

---

## 🛠 Data Preprocessing & Feature Engineering

1. **Renaming** — the `stories` column was renamed to `floors` for clarity.
2. **Binary encoding** — all yes/no categorical columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) were mapped to `1`/`0`.
3. **One-hot encoding** — `furnishingstatus` was one-hot encoded using `pd.get_dummies(..., drop_first=True)`, producing `furnishingstatus_semi-furnished` and `furnishingstatus_unfurnished` (with `furnished` as the reference category to avoid multicollinearity).
4. **Type casting** — the resulting dummy columns were cast to `int64` for consistency with the rest of the feature matrix.
5. **Feature/target split** — `X = df.drop("price", axis=1)`, `y = df["price"]`.

---

## ✂️ Train/Test Split & Scaling

- **Split:** `train_test_split(X, y, test_size=0.2, random_state=50)` → 80% train / 20% test.
- **Scaling:** `StandardScaler` was fit on `X_train` and applied to both `X_train` and `X_test`, since:
  - Linear Regression benefits from standardized features for stable coefficient estimation.
  - It keeps preprocessing consistent across both models for a fair comparison.

---

## 🤖 Modeling

Two regression algorithms were trained and compared on the same scaled features:

| Model | Library | Notes |
|---|---|---|
| **Linear Regression** | `sklearn.linear_model.LinearRegression` | Simple, interpretable baseline model |
| **Random Forest Regressor** | `sklearn.ensemble.RandomForestRegressor` | Non-linear ensemble model, tuned via Bayesian Search |

---

## 🎲 Hyperparameter Tuning — Bayesian Search

Rather than a brute-force grid search, **Bayesian Optimization** (`BayesSearchCV` from `scikit-optimize`) was used to efficiently tune the Random Forest's hyperparameters. Bayesian search builds a probabilistic surrogate model of the objective function and intelligently samples the search space — converging to good hyperparameters with far fewer iterations than an exhaustive grid search.

**Search configuration:**

```python
from skopt import BayesSearchCV

search = BayesSearchCV(
    RandomForestRegressor(random_state=50),
    {
        "n_estimators": (50, 200),
        "max_depth": (2, 10)
    },
    n_iter=10,
    cv=5,
    random_state=50
)
search.fit(X_train_scaled, y_train)
```

- **Search space:** `n_estimators` ∈ [50, 200], `max_depth` ∈ [2, 10]
- **Iterations:** 10
- **Cross-validation:** 5-fold
- **Best parameters found:** `n_estimators = 198`, `max_depth = 8`

The final Random Forest model was refit with these tuned parameters:

```python
random_model = RandomForestRegressor(n_estimators=198, max_depth=8, random_state=50)
```

---

## 📏 Model Evaluation

Both models were evaluated on the held-out test set using four standard regression metrics:

| Metric | Description |
|---|---|
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and actual price — robust to outliers, easy to interpret in currency units |
| **MSE** (Mean Squared Error) | Average of squared errors — penalizes large mistakes more heavily |
| **RMSE** (Root Mean Squared Error) | Square root of MSE — brings error back to the original price scale |
| **R²** (Coefficient of Determination) | Proportion of variance in price explained by the model (closer to 1 = better fit) |

---

## 📈 Results

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| **MAE** | 736,886.17 | 712,655.31 |
| **MSE** | 829,690,385,060.86 | 854,755,420,969.21 |
| **RMSE** | 910,873.42 | 924,529.84 |
| **R²** | 0.76 | 0.75 |

*(Metrics computed on the 20% held-out test set, currency units in the dataset's original price scale.)*

---

## ✅ Conclusion

> **Linear Regression > Random Forest** for this dataset.

Despite Random Forest achieving a slightly lower MAE, **Linear Regression outperformed Random Forest on RMSE and R²** — the two metrics most sensitive to overall fit quality and large errors. This is a reasonable outcome given:

- The dataset is relatively **small** (545 rows), which limits how much a complex ensemble model like Random Forest can leverage its non-linear capacity without overfitting.
- Several strong **linear relationships** exist in the data (most notably `area` vs. `price`, as seen during EDA), which favors a linear model.


---

## 🧰 Tech Stack

- **Python 3**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — modeling, preprocessing, metrics, train/test split
- `scikit-optimize` (`skopt`) — Bayesian hyperparameter search (`BayesSearchCV`)

---

## ▶️ How to Run

1. Clone the repository and place `Housing.csv` in the working directory.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scikit-optimize
   ```
3. Open and run `RaulProject.ipynb` (e.g., in Jupyter or Google Colab).

---

*Author: Raul Gadirov*
