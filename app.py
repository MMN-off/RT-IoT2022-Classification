from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

st.set_page_config(
    page_title="RT-IoT2022 Attack Classification",
    page_icon="🛡️",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_TEST_PATH = BASE_DIR / "test_data.csv"
TARGET_COLUMN = "Attack_type"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_COLUMNS = [
    "Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC",
]


# -----------------------------------------------------------------------------
# Load the fitted pipelines. The preprocessing steps are included in each file.
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the five trained models...")
def load_models() -> dict:
    loaded = {}
    missing_files = []

    for display_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename
        if not model_path.is_file():
            missing_files.append(str(model_path))
            continue
        loaded[display_name] = joblib.load(model_path)

    if missing_files:
        raise FileNotFoundError(
            "The following saved model files are missing:\n"
            + "\n".join(missing_files)
        )

    return loaded


def get_expected_features(model) -> list[str]:
    """Get the original input column names recorded by the fitted pipeline."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # Fallback for scikit-learn versions where the pipeline does not expose it.
    for step in getattr(model, "named_steps", {}).values():
        if hasattr(step, "feature_names_in_"):
            return list(step.feature_names_in_)

    raise ValueError(
        "Could not read the feature names from the saved model. "
        "Please check that the complete fitted pipeline was saved."
    )


def calculate_macro_ovr_auc(y_true, scores, class_labels) -> float:
    """Calculate macro one-vs-rest AUC from probabilities or decision scores."""
    class_auc_values = []

    for column_index, class_label in enumerate(class_labels):
        binary_target = (y_true == class_label).astype(int)

        # AUC is undefined when the test file contains only positives or only
        # negatives for a class. Such a class is skipped with a note in the UI.
        if binary_target.nunique() < 2:
            continue

        class_auc_values.append(
            roc_auc_score(binary_target, scores[:, column_index])
        )

    if not class_auc_values:
        return float("nan")

    return float(np.mean(class_auc_values))


@st.cache_data(show_spinner=False, max_entries=3)
def evaluate_dataset(csv_bytes: bytes) -> dict:
    """Evaluate all five saved models on the uploaded CSV; no model is trained."""
    data = pd.read_csv(BytesIO(csv_bytes))

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"The uploaded CSV must contain the target column '{TARGET_COLUMN}'."
        )

    if data.empty:
        raise ValueError("The uploaded CSV has no data rows.")

    models = load_models()
    first_model = next(iter(models.values()))
    expected_features = get_expected_features(first_model)

    feature_data = data.drop(columns=[TARGET_COLUMN])
    missing_features = [
        column for column in expected_features if column not in feature_data.columns
    ]
    extra_features = [
        column for column in feature_data.columns if column not in expected_features
    ]

    if missing_features:
        raise ValueError(
            "The uploaded CSV is missing required input features:\n"
            + ", ".join(missing_features)
        )

    X_test = feature_data.loc[:, expected_features].copy()
    y_test = data[TARGET_COLUMN].copy()

    if y_test.isna().any():
        raise ValueError(
            f"The target column '{TARGET_COLUMN}' contains missing values. "
            "Please correct the CSV and upload it again."
        )

    missing_value_columns = X_test.columns[X_test.isna().any()].tolist()
    if missing_value_columns:
        raise ValueError(
            "Some input features contain missing values: "
            + ", ".join(missing_value_columns)
        )

    metrics_rows = []
    details = {}
    auc_class_counts = {}

    for model_name, model in models.items():
        class_labels = model.named_steps["classifier"].classes_
        unknown_labels = sorted(
            set(y_test.unique()) - set(class_labels), key=str
        )
        if unknown_labels:
            raise ValueError(
                f"The test CSV contains target labels not recognised by "
                f"{model_name}: {unknown_labels}"
            )

        predictions = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            model_scores = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            model_scores = model.decision_function(X_test)
            if model_scores.ndim == 1:
                model_scores = model_scores.reshape(-1, 1)
        else:
            model_scores = None

        if model_scores is not None:
            auc_value = calculate_macro_ovr_auc(
                y_test, model_scores, class_labels
            )
            valid_auc_classes = sum(
                (y_test == label).nunique() == 2 for label in class_labels
            )
        else:
            auc_value = float("nan")
            valid_auc_classes = 0

        metrics_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC": auc_value,
                "Precision": precision_score(
                    y_test,
                    predictions,
                    labels=class_labels,
                    average="macro",
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    predictions,
                    labels=class_labels,
                    average="macro",
                    zero_division=0,
                ),
                "F1 Score": f1_score(
                    y_test,
                    predictions,
                    labels=class_labels,
                    average="macro",
                    zero_division=0,
                ),
                "MCC": matthews_corrcoef(y_test, predictions),
            }
        )

        report = classification_report(
            y_test,
            predictions,
            labels=class_labels,
            target_names=[str(label) for label in class_labels],
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, predictions, labels=class_labels)

        details[model_name] = {
            "class_labels": [str(label) for label in class_labels],
            "confusion_matrix": matrix,
            "classification_report": pd.DataFrame(report).T,
        }
        auc_class_counts[model_name] = valid_auc_classes

    comparison = pd.DataFrame(metrics_rows, columns=METRIC_COLUMNS)

    return {
        "row_count": len(data),
        "feature_count": len(expected_features),
        "comparison": comparison,
        "details": details,
        "extra_features": extra_features,
        "auc_class_counts": auc_class_counts,
    }


# -----------------------------------------------------------------------------
# Streamlit page
# -----------------------------------------------------------------------------
st.title("RT-IoT2022 — Network Attack Classification")
st.caption(
    "Multiclass classification of network traffic using five trained machine "
    "learning models."
)

with st.sidebar:
    st.header("About this project")
    st.write(
        "This application evaluates saved classification pipelines on a labelled "
        "test CSV. The uploaded data is used only for evaluation; no model is "
        "trained or updated here."
    )
    st.divider()
    st.write("**Target column:** `Attack_type`")
    st.write("**Models:** 5")

st.subheader("1. Upload the test dataset")
uploaded_file = st.file_uploader(
    "Choose a labelled test CSV file",
    type=["csv"],
    help="The CSV must contain all model input features and the Attack_type target column.",
)

if uploaded_file is not None:
    csv_bytes = uploaded_file.getvalue()
    source_name = uploaded_file.name
elif SAMPLE_TEST_PATH.is_file():
    csv_bytes = SAMPLE_TEST_PATH.read_bytes()
    source_name = "Bundled test_data.csv"
    st.info(
        "Using the sample held-out test_data.csv included with this project. "
        "You can upload another labelled CSV above to evaluate that file instead."
    )
else:
    st.warning(
        "Upload a labelled CSV to begin. The file must include the "
        "Attack_type column."
    )
    st.stop()

try:
    with st.spinner("Evaluating the five saved models on the test data..."):
        evaluation = evaluate_dataset(csv_bytes)
except Exception as error:
    st.error(f"Unable to evaluate this CSV: {error}")
    st.stop()

st.success(f"Evaluation completed using: {source_name}")

if evaluation["extra_features"]:
    st.warning(
        "The uploaded file contains additional columns which are not used by "
        "the saved models. They were ignored: "
        + ", ".join(evaluation["extra_features"])
    )

st.caption(
    f"Rows evaluated: {evaluation['row_count']:,}  |  "
    f"Input features used: {evaluation['feature_count']}"
)

# Show the comparison across all five models.
st.subheader("2. Model performance comparison")
comparison = evaluation["comparison"]
st.dataframe(
    comparison.style.format(
        {
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "MCC": "{:.4f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download model comparison (CSV)",
    data=comparison.to_csv(index=False).encode("utf-8"),
    file_name="uploaded_test_model_comparison.csv",
    mime="text/csv",
)

# Allow detailed inspection of one model at a time.
st.subheader("3. Inspect an individual model")
selected_model = st.selectbox(
    "Select a model",
    options=list(MODEL_FILES.keys()),
)
selected_details = evaluation["details"][selected_model]
selected_metrics = comparison.loc[
    comparison["Model"] == selected_model
].iloc[0]

metric_columns = st.columns(6)
metric_names = [
    ("Accuracy", "Accuracy"),
    ("AUC", "AUC"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("F1 Score", "F1 Score"),
    ("MCC", "MCC"),
]
for column, (label, key) in zip(metric_columns, metric_names):
    value = selected_metrics[key]
    display_value = "N/A" if pd.isna(value) else f"{value:.4f}"
    column.metric(label, display_value)

valid_auc_count = evaluation["auc_class_counts"][selected_model]
if valid_auc_count < len(selected_details["class_labels"]):
    st.warning(
        "AUC could be calculated for only "
        f"{valid_auc_count} of {len(selected_details['class_labels'])} classes "
        "because at least one class did not have both positive and negative "
        "examples in this test file."
    )

left, right = st.columns([1.15, 1])

with left:
    st.markdown("**Confusion matrix**")
    labels = selected_details["class_labels"]
    matrix = selected_details["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(11, 9))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        values_format="d",
        colorbar=False,
    )
    ax.set_title(f"{selected_model} — Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.markdown("**Classification report**")
    st.dataframe(
        selected_details["classification_report"].style.format(
            {
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "support": "{:.0f}",
            }
        ),
        use_container_width=True,
    )

st.divider()
st.caption(
    "Note: The saved preprocessing and model pipelines are loaded from the "
    "model/ directory. This application performs inference and evaluation only; "
    "it does not retrain the classifiers."
)
