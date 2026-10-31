import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# The 4 Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

# Evaluation Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score




# 1.1 Load Dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")

print("--- Data Structure & Summary ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)


# 1.2 Remove Customer ID
df.drop("customerID", axis=1, inplace=True)


# 1.3 Convert TotalCharges into Numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)


# 1.4 Check and Handle Missing Values
print("\n--- Missing Values Check ---")
print(df.isnull().sum())

cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("Churn")

num_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n--- Missing Values After Imputation ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())


# 1.5 Encode Target Variable
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])

print("\nTarget Mapping:")
for i, class_name in enumerate(le.classes_):
    print(i, "=", class_name)


# 1.6 Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()


# 1.7 Tenure Distribution
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="tenure", kde=True)
plt.title("Distribution of Customer Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Frequency")
plt.show()


# 1.8 Monthly Charges Distribution
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Monthly Charges")
plt.show()

# 1.9 Contract Type vs Customer Churn

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.show()


# 10 Correlation Heatmap
plt.figure(figsize=(9, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# =============================================================================
# DATA SPLITTING AND FEATURE SCALING
# =============================================================================

# Feature matrix X and Target y
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Encode Categorical Features
try:
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse=False
    )

X_train_cat = encoder.fit_transform(X_train[cat_cols])
X_test_cat = encoder.transform(X_test[cat_cols])

X_train_cat = pd.DataFrame(
    X_train_cat,
    columns=encoder.get_feature_names_out(cat_cols),
    index=X_train.index
)

X_test_cat = pd.DataFrame(
    X_test_cat,
    columns=encoder.get_feature_names_out(cat_cols),
    index=X_test.index
)

# Keep Numerical Features
X_train_num = X_train[num_cols]
X_test_num = X_test[num_cols]

# Combine Numerical and Categorical Features
X_train_final = pd.concat(
    [X_train_num, X_train_cat], axis=1
)

X_test_final = pd.concat(
    [X_test_num, X_test_cat], axis=1
)

# StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)


# =============================================================================
# UNIT 2: MODEL 1 - LOGISTIC REGRESSION
# =============================================================================

print("\nUNIT 2: LOGISTIC REGRESSION")

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

plt.figure(figsize=(5, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred_lr),
    annot=True, fmt="d", cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# =============================================================================
# UNIT 3 (PART 1): MODEL 2 - K-NEAREST NEIGHBORS (KNN)
# =============================================================================

print("\nUNIT 3 (PART 1): K-NEAREST NEIGHBORS (KNN)")

knn_model = KNeighborsClassifier(
    n_neighbors=9, metric="euclidean"
)

knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

plt.figure(figsize=(5, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred_knn),
    annot=True, fmt="d", cmap="Greens",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# =============================================================================
# UNIT 3 (PART 2): MODEL 3 - GAUSSIAN NAIVE BAYES
# =============================================================================

print("\nUNIT 3 (PART 2): GAUSSIAN NAIVE BAYES")

nb_model = GaussianNB()

nb_model.fit(X_train_scaled, y_train)
y_pred_nb = nb_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))

plt.figure(figsize=(5, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred_nb),
    annot=True, fmt="d", cmap="Oranges",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title("Gaussian Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# =============================================================================
# UNIT 3 (PART 3): MODEL 4 - RANDOM FOREST CLASSIFIER
# =============================================================================

print("\nUNIT 3 (PART 3): RANDOM FOREST CLASSIFIER")

rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Random Forest Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred_rf),
    annot=True, fmt="d", cmap="Purples",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Visualize One Tree from Random Forest
plt.figure(figsize=(18, 9))
plot_tree(
    rf_model.estimators_[0],
    max_depth=2,
    filled=True,
    feature_names=X_train_final.columns,
    class_names=le.classes_,
    rounded=True
)
plt.title("Random Forest - One Decision Tree")
plt.show()


# =============================================================================
# FINAL SUMMARY: MODEL COMPARISON TABLE
# =============================================================================

print("\nFINAL PERFORMANCE COMPARISON")

comparison_df = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "K-Nearest Neighbors (k=9)",
        "Gaussian Naive Bayes",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_rf)
    ],

    "Precision": [
        precision_score(y_test, y_pred_lr, zero_division=0),
        precision_score(y_test, y_pred_knn, zero_division=0),
        precision_score(y_test, y_pred_nb, zero_division=0),
        precision_score(y_test, y_pred_rf, zero_division=0)
    ],

    "Recall": [
        recall_score(y_test, y_pred_lr, zero_division=0),
        recall_score(y_test, y_pred_knn, zero_division=0),
        recall_score(y_test, y_pred_nb, zero_division=0),
        recall_score(y_test, y_pred_rf, zero_division=0)
    ],

    "F1 Score": [
        f1_score(y_test, y_pred_lr, zero_division=0),
        f1_score(y_test, y_pred_knn, zero_division=0),
        f1_score(y_test, y_pred_nb, zero_division=0),
        f1_score(y_test, y_pred_rf, zero_division=0)
    ]
})

print(comparison_df.to_string(index=False))


# Best Model Based on F1 Score
best_model_name = comparison_df.loc[
    comparison_df["F1 Score"].idxmax(), "Model"
]

print("\nBest Model Based on Churn F1 Score:", best_model_name)


# Model Comparison Graph
comparison_df.plot(
    x="Model",
    y=["Accuracy", "Precision", "Recall", "F1 Score"],
    kind="bar",
    figsize=(10, 5)
)

plt.title("Final Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()
