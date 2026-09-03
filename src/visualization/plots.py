from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay

sns.set_theme(style="whitegrid", palette="muted")
FIGURES_DIR = Path("output/figures")


def _save_fig(fig, save_path):
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figura guardada en: {save_path}")


def plot_churn_distribution(df, target="Churn", save_path=FIGURES_DIR / "churn_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    counts = df[target].value_counts()
    labels = ["No Churn", "Churn"] if set(counts.index) == {0, 1} else counts.index.tolist()
    axes[0].pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452"], startangle=90, explode=(0, 0.05),
    )
    axes[0].set_title("Distribución de Churn", fontsize=14, fontweight="bold")

    sns.countplot(
        x=target,
        hue=target,
        data=df,
        palette=["#4C72B0", "#DD8452"],
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Conteo de Clientes por Clase", fontsize=14, fontweight="bold")
    for p in axes[1].patches:
        axes[1].annotate(
            f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", fontsize=12,
        )

    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)


def plot_numeric_by_churn(df, num_cols, target="Churn", save_path=FIGURES_DIR / "numeric_by_churn.png"):
    fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 5))
    if len(num_cols) == 1:
        axes = [axes]

    churn_vals = df[target].unique()
    colors = ["#4C72B0", "#DD8452"]
    no_churn_vals = {0, "0", "No", "no", False}
    for ax, col in zip(axes, num_cols):
        for val, color in zip(sorted(churn_vals), colors):
            subset = df[df[target] == val][col].dropna()
            label = "No Churn" if val in no_churn_vals else "Churn"
            ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.legend()

    plt.suptitle("Distribución de variables numéricas por Churn", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)


def plot_categorical_churn_rate(df, cat_cols, target="Churn", save_path=FIGURES_DIR / "categorical_churn_rate.png"):
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(4 * len(cat_cols), 5))
    if len(cat_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        rate = df.groupby(col)[target].apply(lambda x: (x == 1).mean() * 100 if x.dtype != object
                                             else (x == "Yes").mean() * 100)
        rate.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#DD8452", edgecolor="black")
        ax.set_title(f"Churn %\n{col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("% Churn")
        ax.tick_params(axis="x", rotation=30)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%",
                        (p.get_x() + p.get_width() / 2.0, p.get_height()),
                        ha="center", va="bottom", fontsize=9)

    plt.suptitle("Tasa de Churn por variable categórica", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)


def plot_categorical_distribution(df, cat_cols, target="Churn", save_path=FIGURES_DIR / "categorical_distribution.png"):
    n = len(cat_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    no_churn_vals = {0, "0", "No", "no", False}

    for ax, col in zip(axes, cat_cols):
        cats = sorted(df[col].dropna().unique())
        x = np.arange(len(cats))
        width = 0.35

        for i, (churn_val, label, color) in enumerate([
            (0, "No Churn", "#4C72B0"),
            (1, "Churn", "#DD8452"),
        ]):
            mask = df[target].apply(lambda v: v in no_churn_vals) if churn_val == 0 else df[target].apply(lambda v: v not in no_churn_vals)
            subset = df[mask]
            total = len(subset)
            counts = [subset[subset[col] == c].shape[0] / total * 100 for c in cats]
            ax.bar(x + i * width, counts, width, label=label, color=color, edgecolor="black", alpha=0.85)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([str(c) for c in cats], rotation=30, ha="right", fontsize=9)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("% dentro del grupo")
        ax.legend(fontsize=8)

    # Ocultar ejes sobrantes
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Distribución de variables categóricas por Churn", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, model_name="Modelo", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues", ax=ax,
    )
    ax.set_title(f"Matriz de Confusión — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, save_path or FIGURES_DIR / f"confusion_matrix_{model_name}.png")
    plt.close(fig)


def plot_roc_curves(trained_models, X_test, y_test, save_path=FIGURES_DIR / "roc_curves.png"):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("tab10", n_colors=len(trained_models))

    for (name, model), color in zip(trained_models.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Curvas ROC — Comparativa de Modelos", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)


def plot_feature_importance(model, feature_names, top_n=20, model_name="Modelo", save_path=None):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    top.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="black")
    ax.set_title(f"Top {top_n} variables mas importantes - {model_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importancia relativa")
    plt.tight_layout()
    _save_fig(fig, save_path or FIGURES_DIR / f"feature_importance_{model_name}.png")
    plt.close(fig)


def plot_shap_summary(model, X_test, model_name="Modelo", save_path=None):
    try:
        import shap
    except ImportError:
        print("shap no disponible. Instala con: pip install shap")
        return

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap >= 0.40 devuelve array 3D (n_samples, n_features, n_classes)
    # shap < 0.40 devuelve lista [clase0, clase1]
    if isinstance(shap_values, list):
        vals = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    # shap.summary_plot crea su propia figura; no pre-crear fig/ax
    shap.summary_plot(vals, X_test, show=False, plot_size=(10, 8))
    fig = plt.gcf()
    plt.title(f"SHAP Summary - {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, save_path or FIGURES_DIR / f"shap_summary_{model_name}.png")
    plt.close(fig)


def plot_churn_score_distribution(scoring_df, save_path=FIGURES_DIR / "churn_score_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma del score
    scoring_df["churn_score"].hist(bins=40, ax=axes[0], color="#4C72B0", edgecolor="white")
    axes[0].axvline(0.3, color="orange", linestyle="--", label="Low/Medium")
    axes[0].axvline(0.6, color="red", linestyle="--", label="Medium/High")
    axes[0].set_title("Distribución del Churn Score", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Probabilidad de Churn")
    axes[0].legend()

    # Conteo por risk tier
    tier_order = ["High", "Medium", "Low"]
    tier_colors = {"High": "#DD8452", "Medium": "#FDD783", "Low": "#4C72B0"}
    counts = scoring_df["risk_tier"].value_counts().reindex(tier_order)
    axes[1].bar(counts.index, counts.values,
                color=[tier_colors[t] for t in counts.index], edgecolor="black")
    axes[1].set_title("Clientes por Nivel de Riesgo", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Número de clientes")
    for i, (idx, val) in enumerate(counts.items()):
        pct = val / len(scoring_df) * 100
        axes[1].annotate(f"{val:,} ({pct:.1f}%)", (i, val),
                         ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)
