import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Quantum XAI | Drug Sensitivity", page_icon="🧬", layout="wide")
st.title("🧬 Quantum XAI — Drug Sensitivity Explorer")
st.caption("Member 7 application prototype | Explainable hybrid quantum-classical research workflow")

st.warning("Research prototype only. Demo data are synthetic; this dashboard is not a clinical decision-support tool. The model below is a classical baseline, not a trained quantum model.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    st.markdown("CSV should contain feature columns and one target column.")
    st.header("Model")
    task = st.selectbox("Prediction task", ["Classification", "Regression"])
    test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
    trees = st.slider("Random Forest trees", 50, 500, 150, 50)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)

if uploaded:
    df = pd.read_csv(uploaded)
else:
    demo_path = "data/demo_drug_sensitivity.csv"
    try:
        df = pd.read_csv(demo_path)
    except Exception:
        st.info("Upload a CSV to begin. The bundled synthetic demo CSV is available in the package's data folder.")
        st.stop()

if df.empty or len(df.columns) < 2:
    st.error("Dataset must contain at least one feature column and one target column.")
    st.stop()

st.subheader("1. Dataset overview")
c1,c2,c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{len(df.columns):,}")
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
st.dataframe(df.head(20), use_container_width=True)

target = st.selectbox("Choose target column", list(df.columns), index=len(df.columns)-1)
X = df.drop(columns=[target]).copy()
y = df[target].copy()
valid = y.notna()
X, y = X.loc[valid], y.loc[valid]
if X.shape[1] == 0 or len(y) < 8:
    st.error("Need at least one feature and eight labelled rows.")
    st.stop()

numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]
transformers = []
if numeric:
    transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")),("scale", StandardScaler())]), numeric))
if categorical:
    transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
pre = ColumnTransformer(transformers)

if task == "Classification":
    if y.nunique() < 2:
        st.error("Classification requires at least two target classes.")
        st.stop()
    stratify = y if y.value_counts().min() >= 2 else None
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=test_size,random_state=int(seed),stratify=stratify)
    model = Pipeline([("prep",pre),("model",RandomForestClassifier(n_estimators=int(trees),random_state=int(seed),class_weight="balanced"))])
else:
    if not pd.api.types.is_numeric_dtype(y):
        st.error("Regression requires a numeric target column.")
        st.stop()
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=test_size,random_state=int(seed))
    model = Pipeline([("prep",pre),("model",RandomForestRegressor(n_estimators=int(trees),random_state=int(seed)))])

if st.button("Train classical baseline", type="primary"):
    try:
        model.fit(Xtr,ytr)
        pred = model.predict(Xte)
        st.session_state["model"] = model
        st.session_state["features"] = list(X.columns)
        st.session_state["target"] = target
        st.session_state["task"] = task
        st.session_state["Xte"] = Xte
        st.session_state["yte"] = yte
        st.session_state["pred"] = pred
    except Exception as e:
        st.error(f"Training failed: {e}")

if "model" in st.session_state:
    st.subheader("2. Baseline evaluation")
    pred = st.session_state["pred"]
    yte = st.session_state["yte"]
    if st.session_state["task"] == "Classification":
        a,b = st.columns(2)
        a.metric("Accuracy", f"{accuracy_score(yte,pred):.3f}")
        b.metric("Weighted F1", f"{f1_score(yte,pred,average='weighted',zero_division=0):.3f}")
        fig,ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(yte,pred,ax=ax,xticks_rotation=45)
        st.pyplot(fig)
    else:
        a,b = st.columns(2)
        a.metric("MAE", f"{mean_absolute_error(yte,pred):.3f}")
        b.metric("R²", f"{r2_score(yte,pred):.3f}")

    st.subheader("3. Global feature importance")
    fitted = st.session_state["model"]
    prep = fitted.named_steps["prep"]
    rf = fitted.named_steps["model"]
    names = []
    if numeric: names.extend(numeric)
    if categorical:
        enc = prep.named_transformers_["cat"].named_steps["onehot"]
        names.extend(enc.get_feature_names_out(categorical).tolist())
    imp = pd.DataFrame({"Feature":names,"Importance":rf.feature_importances_}).sort_values("Importance",ascending=False).head(20)
    fig,ax = plt.subplots(figsize=(8,5))
    ax.barh(imp["Feature"][::-1],imp["Importance"][::-1])
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("Top features (global, model-specific)")
    st.pyplot(fig)
    st.dataframe(imp, use_container_width=True)

    st.subheader("4. Explainability and quantum workflow")
    st.markdown("""
    - **Current explanation:** global Random Forest feature importance. It indicates how the fitted classical model used features; it is not causal evidence.
    - **Next integration:** connect SHAP or LIME to the team's final trained model for local and global explanations.
    - **Quantum stage:** connect the team's verified feature-encoding and variational quantum circuit outputs. The included PennyLane file is a template only.
    - **Robustness:** evaluate repeated seeds, noise levels, feature-selection stability, and calibration before comparing classical and hybrid approaches.
    """)
else:
    st.info("Select the target column and press **Train classical baseline** to see evaluation and global feature importance.")

st.divider()
st.caption("Member 7 contribution: application interface and project documentation. Replace demo inputs with the team's approved dataset and integrated model before final evaluation.")
