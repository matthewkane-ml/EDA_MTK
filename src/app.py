# # Step 1: Problem statement and data collection
# # Problem statement: What factors influence the price of an Airbnb listing in New York City?"
 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
 
# engine = db_connect()  # DB connection not needed; data loaded from CSV
 
# your code here
 
base_dir = os.path.dirname(os.path.abspath(__file__))
import_data = pd.read_csv(os.path.join(base_dir, "../data/raw/raw_data.csv"))
 
# testing to see if it worked (it did)
# print(import_data.head())
 
 
# %%
# # Step 2: Exploration and Data cleaning
# # We need to get rid of duplicates, null values, irrelevant columns, and potentially outliers (if relevant). We will be able to see
# # outliers easier after the graphing has been done so we will save that portion of the "data cleaning" step for later
 
print(import_data.shape)
 
print(import_data.info())
 
# find duplicates - most of the columns are allowed to have duplicates, with the possible exception of id
print(f"The number of duplicated ID records is: {import_data['id'].duplicated().sum()}")
 
# we need to get rid of columns that are really not related to the problem statement
import_data.drop(["id", "name", "host_name", "last_review", "reviews_per_month"], axis = 1, inplace = True)
import_data.head()
 
# here we have no null values to drop :)
print(f"Total null values: {import_data.isnull().sum().sort_values(ascending = False)}")
 
 
# %%
# Step 3: Analysis of univariate variables
# Analysis on categorical variables
 
fig, axis = plt.subplots(2, 3, figsize=(10, 7))
 
# Create Histograms for categorical variables - gives us a good overview of the information they contain
sns.histplot(ax = axis[0,0], data = import_data, x = "host_id")
sns.histplot(ax = axis[0,1], data = import_data, x = "neighbourhood_group")
sns.histplot(ax = axis[0,2], data = import_data, x = "neighbourhood")
sns.histplot(ax = axis[1,0], data = import_data, x = "room_type")
sns.histplot(ax = axis[1,1], data = import_data, x = "availability_365")
fig.delaxes(axis[1, 2])
 
plt.tight_layout()
plt.show()
 
# ------------------
 
fig, axis = plt.subplots(4, 2, figsize = (10, 14), gridspec_kw = {"height_ratios": [6, 1, 6, 1]})
 
# Create histograms and boxplots for numerical variables
sns.histplot(ax = axis[0, 0], data = import_data, x = "price")
sns.boxplot(ax = axis[1, 0], data = import_data, x = "price")
 
sns.histplot(ax = axis[2, 0], data = import_data, x = "number_of_reviews")
sns.boxplot(ax = axis[3, 0], data = import_data, x = "number_of_reviews")
 
sns.histplot(ax = axis[0, 1], data = import_data, x = "minimum_nights").set_xlim(0, 200)
sns.boxplot(ax = axis[1, 1], data = import_data, x = "minimum_nights")
 
sns.histplot(ax = axis[2,1], data = import_data, x = "calculated_host_listings_count")
sns.boxplot(ax = axis[3, 1], data = import_data, x = "calculated_host_listings_count")
 
plt.tight_layout()
plt.show()
 
 
# %%
# Step 4: Analysis of multivariate variables
 
# Numerical - Numerical Analysis
 
fig, axis = plt.subplots(4, 2, figsize = (10, 16))
 
sns.regplot(ax = axis[0, 0], data = import_data, x = "minimum_nights", y = "price")
sns.heatmap(import_data[["price", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)
 
sns.regplot(ax = axis[0, 1], data = import_data, x = "number_of_reviews", y = "price").set(ylabel = None)
sns.heatmap(import_data[["price", "number_of_reviews"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])
 
sns.regplot(ax = axis[2, 0], data = import_data, x = "calculated_host_listings_count", y = "price").set(ylabel = None)
sns.heatmap(import_data[["price", "calculated_host_listings_count"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0]).set(ylabel = None)
 
fig.delaxes(axis[2, 1])
fig.delaxes(axis[3, 1])
 
plt.tight_layout()
plt.show()
 
# -----------------------
 
fig, axis = plt.subplots(figsize = (5, 4))
 
sns.countplot(data = import_data, x = "room_type", hue = "neighbourhood_group")
 
plt.show()
 
# ------------------
 
# Factorize the Room Type and Neighborhood Data - turning the text categories into ints so when we feed it in the correlational heatmap, the math works
import_data["room_type"] = pd.factorize(import_data["room_type"])[0]
import_data["neighbourhood_group"] = pd.factorize(import_data["neighbourhood_group"])[0]
import_data["neighbourhood"] = pd.factorize(import_data["neighbourhood"])[0]
 
fig, axes = plt.subplots(figsize=(15, 15))
 
sns.heatmap(import_data[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",
                        "number_of_reviews", "calculated_host_listings_count", "availability_365"]].corr(), annot = True, fmt = ".2f")
 
plt.tight_layout()
plt.show()
 
sns.pairplot(data = import_data)
 
 
# %%
# Step 5: Feature engineering
 
import_data.describe()
 
# ----------------
 
fig, axes = plt.subplots(3, 3, figsize = (15, 15))
 
sns.boxplot(ax = axes[0, 0], data = import_data, y = "neighbourhood_group")
sns.boxplot(ax = axes[0, 1], data = import_data, y = "price")
sns.boxplot(ax = axes[0, 2], data = import_data, y = "minimum_nights")
sns.boxplot(ax = axes[1, 0], data = import_data, y = "number_of_reviews")
sns.boxplot(ax = axes[1, 1], data = import_data, y = "calculated_host_listings_count")
sns.boxplot(ax = axes[1, 2], data = import_data, y = "availability_365")
sns.boxplot(ax = axes[2, 0], data = import_data, y = "room_type")
 
plt.tight_layout()
plt.show()
 
# The purpose of showing all these boxplots is basically to give us an intuition on where the outliers are. As we can see, many of the columns actually have outliers
# However, the three worst offenders are "price", "minimum_nights" and "number_of_reviews"
 
 
# %%
# Outlier handling:
 
# outlier detection for price
price_stats = import_data["price"].describe()
price_iqr = price_stats["75%"] - price_stats["25%"]
upper_limit = price_stats["75%"] + 1.5 * price_iqr
lower_limit = price_stats["25%"] - 1.5 * price_iqr
 
# Clean the outliers
total_data = import_data[import_data["price"] > 0]
 
count_0 = import_data[import_data["price"] == 0].shape[0]
count_1 = import_data[import_data["price"] == 1].shape[0]
 
# outlier detection for number_of_reviews ----------------------------------
 
review_stats = import_data["number_of_reviews"].describe()
review_stats
 
# IQR for number_of_reviews
review_iqr = review_stats["75%"] - review_stats["25%"]
 
upper_limit = review_stats["75%"] + 1.5 * review_iqr
lower_limit = review_stats["25%"] - 1.5 * review_iqr
 
# outlier detection minimum_nights ----------------------------------------
nights_stats = total_data["minimum_nights"].describe()
nights_iqr = nights_stats["75%"] - nights_stats["25%"]
 
upper_limit = nights_stats["75%"] + 1.5 * nights_iqr
lower_limit = nights_stats["25%"] - 1.5 * nights_iqr
 
total_data = total_data[total_data["minimum_nights"] <= 15]
 
count_0 = total_data[total_data["minimum_nights"] == 0].shape[0]
count_1 = total_data[total_data["minimum_nights"] == 1].shape[0]
count_2 = total_data[total_data["minimum_nights"] == 2].shape[0]
count_3 = total_data[total_data["minimum_nights"] == 3].shape[0]
count_4 = total_data[total_data["minimum_nights"] == 4].shape[0]
 
 
# %%
# Feature Scaling:
 
from sklearn.preprocessing import MinMaxScaler
 
num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count",
                 "availability_365", "neighbourhood_group", "room_type"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(import_data[num_variables])
df_scal = pd.DataFrame(scal_features, index = import_data.index, columns = num_variables)
df_scal["price"] = import_data["price"]
df_scal.head()
 
 
# %%
# Step 6: Feature Selection
 
# These two chi2 and SelectKBest are used to test the relationship between the "feature" columns and the "target" column - where we select the k=4 best columns
# that are related to the target column
from sklearn.feature_selection import chi2, SelectKBest
 
# This is going to be used to split the dataframe up into the test_data and the train_data. Thank you, sklearn!
from sklearn.model_selection import train_test_split
 
# this is the entire dataframe object without the target column
X = df_scal.drop("price", axis = 1)
 
# this is dataframe with ONLY the target column
y = df_scal["price"]
 
# splitting the data into features/attributes and target/class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
 
selection_model = SelectKBest(chi2, k = 4)
selection_model.fit(X_train, y_train)
 
# get_support() returns a boolean mask - an array of True/False values, one for each feature column, telling you which ones were selected and which ones were dropped
ix = selection_model.get_support()
 
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])
 
# Print selected features and their chi2 scores
print("\n--- Feature Selection Results ---")
print("Selected features:", list(X_train.columns.values[ix]))
print("\nAll chi2 scores (ranked):")
for feat, score in sorted(zip(X_train.columns, selection_model.scores_), key=lambda x: -x[1]):
    marker = " <-- selected" if feat in X_train.columns.values[ix] else ""
    print(f"  {feat}: {score:.2f}{marker}")
print("---------------------------------\n")
 
# Save the data
X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)
 
processed_dir = os.path.join(base_dir, "../data/processed")
os.makedirs(processed_dir, exist_ok=True)

X_train_sel.to_csv(os.path.join(processed_dir, "clean_train_data.csv"), index = False)
X_test_sel.to_csv(os.path.join(processed_dir, "clean_test_data.csv"), index = False)
 
# gives us an overview of the data
X_train_sel.head()
 