import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
# st.set_page_config(page_title="Enhanced ML Classifier App", layout="wide")
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
st.title("🤖 Machine Learning Classifier Comparison Dashboard")

# File Upload
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])
st.write("after uploading a suitable dataset, select the target column.")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Preprocessing
    # ---------------------------
    st.sidebar.header("🧹 Data Preprocessing")
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if st.sidebar.checkbox("Handle Missing Values"):
        df = df.fillna(df.mean(numeric_only=True))
        st.sidebar.success("Missing values filled with column mean.")

    if st.sidebar.checkbox("Scale Features"):
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        st.sidebar.success("Features scaled using StandardScaler.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
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
    for name, clf in classifiers.items():
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
        except Exception as e:
            results[name] = f"Error: {e}"

    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
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

else:
    st.info("📥 Upload a CSV file to begin model comparison.")
