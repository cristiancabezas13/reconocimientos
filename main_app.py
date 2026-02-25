import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
)


# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Digits (MNIST-like) Classifier", layout="wide")


# ----------------------------
# Data
# ----------------------------
@st.cache_data
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Digits dataset (8x8 images), similar to MNIST but smaller and included in sklearn.
    X: (n_samples, 64)
    y: (n_samples,)
    images: (n_samples, 8, 8)
    target_names: array([0..9])
    """
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(int)
    images = digits.images
    target_names = digits.target_names
    return X, y, images, target_names


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    supports_proba: bool


def _safe_logreg(params: Dict[str, Any]) -> LogisticRegression:
    """
    Compatibilidad sklearn:
    - En versiones recientes 'multi_class' fue retirado.
    - Probamos con y sin ese argumento para evitar TypeError.
    """
    base_kwargs = dict(
        C=float(params.get("lr_C", 1.0)),
        max_iter=int(params.get("lr_max_iter", 1000)),
        solver="lbfgs",
    )
    try:
        return LogisticRegression(**base_kwargs, multi_class="auto")
    except TypeError:
        return LogisticRegression(**base_kwargs)


def get_model_specs(needs_proba: bool, params: Dict[str, Any]) -> Dict[str, ModelSpec]:
    # Logistic Regression
    lr = _safe_logreg(params)

    # SVM (RBF)
    svm_probability = True if needs_proba else bool(params.get("svm_probability", False))
    svm = SVC(
        C=float(params.get("svm_C", 1.0)),
        kernel=str(params.get("svm_kernel", "rbf")),
        gamma=str(params.get("svm_gamma", "scale")),
        probability=svm_probability,  # needed for predict_proba (ROC)
    )

    # KNN
    knn = KNeighborsClassifier(
        n_neighbors=int(params.get("knn_k", 5)),
        weights=str(params.get("knn_weights", "uniform")),
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=int(params.get("rf_n_estimators", 200)),
        max_depth=params.get("rf_max_depth", None),
        random_state=int(params.get("random_state", 42)),
    )

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        learning_rate=float(params.get("gb_lr", 0.1)),
        n_estimators=int(params.get("gb_n_estimators", 200)),
        random_state=int(params.get("random_state", 42)),
    )

    # Naive Bayes
    nb = GaussianNB()

    # LDA
    lda = LinearDiscriminantAnalysis()

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(int(params.get("mlp_h", 64)),),
        alpha=float(params.get("mlp_alpha", 1e-4)),
        max_iter=int(params.get("mlp_max_iter", 500)),
        random_state=int(params.get("random_state", 42)),
    )

    return {
        "Logistic Regression": ModelSpec("Logistic Regression", lr, supports_proba=True),
        "SVM (RBF)": ModelSpec("SVM (RBF)", svm, supports_proba=svm_probability),
        "KNN": ModelSpec("KNN", knn, supports_proba=True),
        "Random Forest": ModelSpec("Random Forest", rf, supports_proba=True),
        "Gradient Boosting": ModelSpec("Gradient Boosting", gb, supports_proba=True),
        "Naive Bayes (Gaussian)": ModelSpec("Naive Bayes (Gaussian)", nb, supports_proba=True),
        "LDA": ModelSpec("LDA", lda, supports_proba=True),
        "MLP (Neural Net)": ModelSpec("MLP (Neural Net)", mlp, supports_proba=True),
    }


def make_pipeline(estimator, use_scaler: bool) -> Pipeline:
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> Dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def fig_confusion_matrix(y_true, y_pred, labels) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    return fig


def fig_roc_ovr(y_true: np.ndarray, y_score: np.ndarray, labels: np.ndarray) -> plt.Figure:
    """
    ROC multiclass One-vs-Rest
    y_score: (n_samples, n_classes) probs or scores.
    """
    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=list(labels))

    fig, ax = plt.subplots()

    # per-class ROC
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"Clase {labels[i]} (AUC={roc_auc:.3f})")

    # micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, linestyle="--", label=f"Micro-average (AUC={roc_auc_micro:.3f})")

    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Multiclase (OvR)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def fig_pca_scatter(X2: np.ndarray, y: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=y, s=18)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Clase")
    fig.tight_layout()
    return fig


def fig_decision_boundary_2d(
    X2_train: np.ndarray,
    y_train: np.ndarray,
    X2_test: np.ndarray,
    y_test: np.ndarray,
    clf_2d: Pipeline,
    labels: np.ndarray,
    mesh_step: float = 0.05,
) -> plt.Figure:
    x_min, x_max = X2_train[:, 0].min() - 1.0, X2_train[:, 0].max() + 1.0
    y_min, y_max = X2_train[:, 1].min() - 1.0, X2_train[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step), np.arange(y_min, y_max, mesh_step))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = clf_2d.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.25)

    ax.scatter(X2_train[:, 0], X2_train[:, 1], c=y_train, marker="o", s=18, edgecolors="k", label="Train")
    ax.scatter(X2_test[:, 0], X2_test[:, 1], c=y_test, marker="^", s=22, edgecolors="k", label="Test")

    ax.set_title("Frontera de decisión en PCA (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def show_image_grid(images: np.ndarray, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None, max_items: int = 24):
    n = min(len(images), max_items)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i], cmap="gray")
            if y_pred is None:
                ax.set_title(f"y={y_true[i]}")
            else:
                ok = (y_true[i] == y_pred[i])
                ax.set_title(f"T={y_true[i]}  P={y_pred[i]}\n{'OK' if ok else 'ERR'}", fontsize=9)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ----------------------------
# UI
# ----------------------------
X, y, images, target_names = load_data()
labels = np.array(target_names)

st.title("🧠 Digits (MNIST-like) - Clasificación con Streamlit")
st.write(
    "App interactiva basada en el ejemplo de scikit-learn (Digits 8×8). "
    "Permite entrenar diferentes modelos, ver métricas, ROC (opcional), PCA 2D y frontera de decisión."
)

with st.expander("🔎 Dataset preview (imágenes)"):
    idx = np.random.RandomState(0).choice(len(images), size=24, replace=False)
    show_image_grid(images[idx], y[idx], y_pred=None, max_items=24)


# Sidebar
st.sidebar.header("⚙️ Configuración")
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.25, 0.05)
use_scaler = st.sidebar.checkbox("Estandarizar (StandardScaler)", value=True)

metric_average = st.sidebar.selectbox("Promedio (Precision/Recall/F1)", ["macro", "weighted"], index=0)
show_cv = st.sidebar.checkbox("Mostrar validación cruzada (5-fold)", value=False)

st.sidebar.divider()
st.sidebar.subheader("📈 ROC y visualizaciones")
show_roc = st.sidebar.checkbox("Mostrar ROC multiclase (OvR)", value=False)
show_pca = st.sidebar.checkbox("Mostrar PCA 2D", value=True)
show_boundary = st.sidebar.checkbox("Mostrar frontera de decisión (en PCA 2D)", value=True)

st.sidebar.divider()
st.sidebar.subheader("🤖 Modelo")

model_choice = st.sidebar.selectbox(
    "Selecciona un modelo",
    [
        "Logistic Regression",
        "SVM (RBF)",
        "KNN",
        "Random Forest",
        "Gradient Boosting",
        "Naive Bayes (Gaussian)",
        "LDA",
        "MLP (Neural Net)",
    ],
)

params: Dict[str, Any] = {"random_state": int(random_state)}

# Hiperparámetros por modelo
if model_choice == "Logistic Regression":
    params["lr_C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0, 0.01)
    params["lr_max_iter"] = st.sidebar.slider("max_iter", 200, 5000, 1000, 100)

elif model_choice == "SVM (RBF)":
    params["svm_C"] = st.sidebar.slider("C", 0.01, 20.0, 2.0, 0.01)
    params["svm_gamma"] = st.sidebar.selectbox("gamma", ["scale", "auto"], index=0)
    # si el usuario activa ROC, forzamos probability=True
    params["svm_probability"] = st.sidebar.checkbox("probability=True (necesario para ROC)", value=bool(show_roc))

elif model_choice == "KNN":
    params["knn_k"] = st.sidebar.slider("n_neighbors", 1, 30, 5, 1)
    params["knn_weights"] = st.sidebar.selectbox("weights", ["uniform", "distance"], index=0)

elif model_choice == "Random Forest":
    params["rf_n_estimators"] = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
    md = st.sidebar.slider("max_depth (0=None)", 0, 30, 0, 1)
    params["rf_max_depth"] = None if md == 0 else md

elif model_choice == "Gradient Boosting":
    params["gb_n_estimators"] = st.sidebar.slider("n_estimators", 50, 500, 200, 50)
    params["gb_lr"] = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1, 0.01)

elif model_choice == "MLP (Neural Net)":
    params["mlp_h"] = st.sidebar.slider("hidden_units", 16, 256, 64, 16)
    params["mlp_alpha"] = st.sidebar.select_slider("alpha (L2)", options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)
    params["mlp_max_iter"] = st.sidebar.slider("max_iter", 200, 2000, 500, 100)


# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y, images, test_size=float(test_size), random_state=int(random_state), stratify=y
)

needs_proba = bool(show_roc)
specs = get_model_specs(needs_proba=needs_proba, params=params)
spec = specs[model_choice]

pipe = make_pipeline(spec.estimator, use_scaler=use_scaler)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Probabilidades / scores para ROC
y_score = None
if show_roc:
    if hasattr(pipe, "predict_proba"):
        try:
            y_score = pipe.predict_proba(X_test)
        except Exception:
            y_score = None
    if y_score is None and hasattr(pipe, "decision_function"):
        try:
            df = pipe.decision_function(X_test)
            # decision_function puede ser (n, n_classes) para multiclase
            if df.ndim == 1:
                df = np.vstack([1 - df, df]).T
            y_score = df
        except Exception:
            y_score = None


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Resumen", "📊 Métricas", "🧭 Visualizaciones", "❌ Errores"])

with tab1:
    st.subheader("Resumen")
    c1, c2, c3 = st.columns(3)
    c1.metric("Modelo", spec.name)
    c2.metric("Test size", f"{test_size:.2f}")
    c3.metric("Scaler", "Sí" if use_scaler else "No")

    metrics = compute_metrics(y_test, y_pred, average=metric_average)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['Precision']:.3f}")
    m3.metric("Recall", f"{metrics['Recall']:.3f}")
    m4.metric("F1", f"{metrics['F1']:.3f}")

    if show_cv:
        st.subheader("Validación cruzada (5-fold)")
        try:
            scores = cross_val_score(make_pipeline(spec.estimator, use_scaler), X, y, cv=5, scoring="accuracy")
            st.write(f"Accuracy promedio: **{scores.mean():.3f}**  |  Desv: **{scores.std():.3f}**")
            st.write("Scores:", np.round(scores, 3))
        except Exception as e:
            st.warning(f"No se pudo calcular CV: {e}")

with tab2:
    st.subheader("Métricas y reporte")
    st.pyplot(fig_confusion_matrix(y_test, y_pred, labels), use_container_width=True)

    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T
    st.dataframe(rep_df, use_container_width=True)

with tab3:
    st.subheader("Visualizaciones")

    colA, colB = st.columns(2)

    with colA:
        if show_roc:
            if y_score is None:
                st.warning(
                    "No se pudo calcular ROC para este modelo/configuración. "
                    "Tip: en SVM activa probability=True."
                )
            else:
                st.pyplot(fig_roc_ovr(y_test, y_score, labels), use_container_width=True)
        else:
            st.info("Activa ROC en la barra lateral si quieres ver curvas ROC multiclase.")

    with colB:
        if show_pca:
            # PCA 2D (para scatter y frontera)
            pca_steps = []
            if use_scaler:
                pca_steps.append(("scaler", StandardScaler()))
            pca_steps.append(("pca", PCA(n_components=2, random_state=int(random_state))))
            pca_pipe = Pipeline(pca_steps)

            X2 = pca_pipe.fit_transform(X)
            st.pyplot(fig_pca_scatter(X2, y, "PCA 2D (todas las muestras)"), use_container_width=True)

            if show_boundary:
                # Split coherente en PCA-space
                X2_train = X2[np.isin(np.arange(len(X)), np.where(np.isin(X, X_train).all(axis=1))[0])]
                # La línea anterior no es confiable (por igualdad float). Mejor: recalcular PCA en train y transformar test:
                # Rehacemos PCA entrenando en X_train para coherencia:
                X2_train = pca_pipe.fit_transform(X_train)
                X2_test = pca_pipe.transform(X_test)

                clf_2d = make_pipeline(spec.estimator, use_scaler=False)  # ya escalamos antes con pca_pipe si aplica
                clf_2d.fit(X2_train, y_train)

                st.pyplot(
                    fig_decision_boundary_2d(
                        X2_train, y_train, X2_test, y_test, clf_2d, labels, mesh_step=0.08
                    ),
                    use_container_width=True,
                )
        else:
            st.info("Activa PCA 2D si quieres ver el scatter y/o la frontera de decisión.")

with tab4:
    st.subheader("Errores (misclassifications)")
    mask_err = (y_pred != y_test)
    n_err = int(mask_err.sum())
    st.write(f"Total errores en test: **{n_err}** de **{len(y_test)}**")

    max_show = st.slider("Cuántos errores mostrar", min_value=6, max_value=60, value=24, step=6)

    if n_err == 0:
        st.success("¡No hubo errores con esta configuración!")
    else:
        err_imgs = img_test[mask_err]
        err_true = y_test[mask_err]
        err_pred = y_pred[mask_err]

        # ordenar por (true, pred) para que se vea más organizado
        order = np.lexsort((err_pred, err_true))
        err_imgs = err_imgs[order]
        err_true = err_true[order]
        err_pred = err_pred[order]

        show_image_grid(err_imgs, err_true, err_pred, max_items=int(max_show))

st.sidebar.caption("▶ Ejecuta con: streamlit run main_app.py")
