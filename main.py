import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="Enhanced ML Classifier App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        body { background-color: #ffffff; color: #000000; }
        .stApp { background-color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🤖 Gaussian Process Classifier Comparison Dashboard")

# ---------------------------
# Sidebar Dataset Selection
# ---------------------------
st.sidebar.header("📂 Dataset Options")
dataset_choice = st.sidebar.radio(
    "Choose a dataset source:",
    ("Upload Custom CSV", "Use Sample Dataset")
)

df = None
dataset_name = None

if dataset_choice == "Use Sample Dataset":
    sample_file = st.sidebar.selectbox(
        "Select a sample dataset",
        ("moons.csv", "iris.csv")
    )
    dataset_path = f"{sample_file}"  # local file path
    df = pd.read_csv(dataset_path)
    dataset_name = sample_file
    st.write(f"### Loaded Sample Dataset: {sample_file}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        dataset_name = "custom"
        st.write("### Uploaded Dataset Preview")
    else:
        st.info("📥 Upload a CSV file or select a sample dataset to begin.")
        st.stop()

st.dataframe(df.head())

# ---------------------------
# Preprocessing
# ---------------------------
st.sidebar.header("🧹 Data Preprocessing")

# Auto-detect target column only for known datasets
if dataset_name == "moons.csv":
    target_col = "label"
elif dataset_name == "iris.csv":
    target_col = "Species"
else:
    # For custom datasets, let the user select manually
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)

# Ensure the selected/auto-detected column exists
if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in dataset.")
    st.stop()

X = df.drop(columns=[target_col])
y = df[target_col]

# Optional preprocessing steps
if st.sidebar.checkbox("Handle Missing Values"):
    df = df.fillna(df.mean(numeric_only=True))
    st.sidebar.success("Missing values filled with column mean.")

if st.sidebar.checkbox("Scale Features"):
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    st.sidebar.success("Features scaled using StandardScaler.")

# Adjustable Train-Test Split
split_ratio = st.sidebar.slider("Train-Test Split (test size %)", 10, 50, 30)
test_size = split_ratio / 100.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# ---------------------------
# Model Hyperparameters
# ---------------------------
st.sidebar.header("⚙️ Model Parameters")
rf_n_estimators = st.sidebar.slider("Random Forest Trees", 10, 300, 100)
svm_c = st.sidebar.slider("SVM Regularization (C)", 0.01, 10.0, 1.0)

# ---------------------------
# Classifiers
# ---------------------------
classifiers = {
    "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(length_scale=1.0)),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(C=svm_c, probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=rf_n_estimators),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
}

# ---------------------------
# Model Training & Evaluation
# ---------------------------
results = {}
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

for name, clf in classifiers.items():
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    except Exception as e:
        results[name] = f"Error: {e}"

results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
results_df["Accuracy"] = pd.to_numeric(results_df["Accuracy"], errors="coerce")
results_df = results_df.sort_values(by="Accuracy", ascending=False)

st.write("## 📊 Accuracy Comparison")
st.dataframe(results_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 5))
results_df["Accuracy"].plot(kind="bar", ax=ax, color="skyblue")

ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)

# ---------------------------
# Best Model Summary
# ---------------------------
best_model_name = results_df["Accuracy"].idxmax()
best_acc = results_df["Accuracy"].max()
st.success(f"🏆 Best Model: **{best_model_name}** with Accuracy: **{best_acc:.2f}**")

# ---------------------------
# Detailed Analysis Section
# ---------------------------
st.write("---")
st.subheader("🔍 Model Evaluation Details")

selected_model = st.selectbox(
    "Choose a model for detailed evaluation", list(classifiers.keys())
)
clf = classifiers[selected_model]
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Confusion Matrix
st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Classification Report
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# ---------------------------
# Training and Testing Loss Graphs
# ---------------------------
st.write("### 📈 Training vs Testing Loss")

if hasattr(clf, "predict_proba"):
    y_train_proba = clf.predict_proba(X_train)
    y_test_proba = clf.predict_proba(X_test)

    train_loss = log_loss(y_train, y_train_proba)
    test_loss = log_loss(y_test, y_test_proba)

    fig, ax = plt.subplots()
    ax.bar(["Training Loss", "Testing Loss"], [train_loss, test_loss], color=["#66b3ff", "#ff9999"])
    ax.set_ylabel("Log Loss")
    st.pyplot(fig)
else:
    st.warning("⚠️ This model does not support probability outputs for loss calculation.")

# ---------------------------
# Feature Importance
# ---------------------------
if hasattr(clf, "feature_importances_"):
    st.write("### 🔎 Feature Importance")
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    st.bar_chart(importances.sort_values(ascending=False))

# ---------------------------
# Cross Validation
# ---------------------------
if st.checkbox("Run 5-Fold Cross-Validation on Selected Model"):
    scores = cross_val_score(clf, X, y, cv=5)
    st.write(f"Average CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# ---------------------------
# Save Best Model
# ---------------------------
if st.button("💾 Save Best Model"):
    best_model = classifiers[best_model_name]
    joblib.dump(best_model, f"{best_model_name}.pkl")
    st.success(f"Saved {best_model_name} model as `{best_model_name}.pkl`")
