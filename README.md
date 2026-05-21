# NYC Airbnb Price EDA

> Exploratory data analysis on New York City Airbnb listings — identifying the features that most influence listing price across all five boroughs.

---

## Problem

What factors drive the price of an Airbnb listing in New York City? This project performs a structured, end-to-end EDA on the NYC Airbnb Open Data to answer that question — cleaning the data, exploring distributions and relationships, handling outliers, and delivering a feature-selected dataset ready for downstream modeling.

## Dataset

- **Source:** [NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) — Kaggle
- **Key features analyzed:** `neighbourhood_group`, `neighbourhood`, `room_type`, `price`, `minimum_nights`, `number_of_reviews`, `calculated_host_listings_count`, `availability_365`
- **Dropped at cleaning:** `id`, `name`, `host_name`, `last_review`, `reviews_per_month` — not relevant to price prediction
- **Target variable:** `price`

## Approach

**Step 1 — Data cleaning**
Verified zero duplicate IDs and zero null values across all retained columns. Dropped five irrelevant identifier and metadata columns.

**Step 2 — Univariate analysis**
Plotted histograms and boxplots for all numerical features (`price`, `minimum_nights`, `number_of_reviews`, `calculated_host_listings_count`) and count distributions for categorical features (`neighbourhood_group`, `room_type`).

**Step 3 — Multivariate analysis**
Built regression plots and pairwise correlation heatmaps for numerical features against price. Plotted room type counts broken down by borough. Factorized categorical columns and generated a full feature correlation heatmap and pair plot.

**Step 4 — Outlier handling**
Identified three major outlier offenders from boxplot inspection:
- `price`: removed listings priced at $0
- `minimum_nights`: capped at 15 nights (IQR-based upper limit)
- `number_of_reviews`: documented via IQR analysis

**Step 5 — Feature scaling**
Applied `MinMaxScaler` to all numerical features: `number_of_reviews`, `minimum_nights`, `calculated_host_listings_count`, `availability_365`, `neighbourhood_group`, `room_type`.

**Step 6 — Feature selection**
Used `SelectKBest` with chi-square scoring to identify the 4 features most predictive of price from the scaled feature set. Saved the final train/test splits as `data/processed/clean_train_data.csv` and `clean_test_data.csv`, ready for modeling.

## Results

`SelectKBest` (chi-square, k=4) selected the following features as most predictive of price:

| Selected | Dropped |
|---|---|
| `room_type` | `minimum_nights` |
| `availability_365` | `neighbourhood_group` |
| `number_of_reviews` | |
| `calculated_host_listings_count` | |

The two dropped features — `minimum_nights` and `neighbourhood_group` — had the lowest chi-square scores, suggesting they contribute less independent signal to price after scaling. Notably, `room_type` outranking `neighbourhood_group` implies that *what* you're renting is a stronger price driver than *where* it is, at least within this feature set. The outlier-cleaned, feature-selected output is saved to `data/processed/` as train and test CSVs, ready for a regression model.

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn`

## Run it locally

```bash
git clone https://github.com/matthewkane-ml/EDA_Project.git
cd EDA_Project
pip install -r requirements.txt

# Add the raw dataset to data/raw/raw_data.csv, then run:
python src/app.py
```

## What I'd do next

- Build a regression model (Linear Regression or Random Forest Regressor) on the cleaned output to actually predict listing price
- Investigate geographic price patterns more deeply using latitude/longitude data and a map visualization (e.g., Folium or Plotly)
- Perform a deeper outlier analysis on `price` — many NYC listings are legitimately expensive, so a hard IQR cap may remove valid data points

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-kane-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)
