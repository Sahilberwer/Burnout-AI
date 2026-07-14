"""
train_model.py
================================================================
Prediction of Burnout Risk Level Using Decision Tree & Random Forest
Honeywell AI Internship Project — Production Training Pipeline

Run with:
    python train_model.py

This script performs the COMPLETE pipeline used in the source
notebook (Burnout_Risk_Prediction.ipynb):
    1. Load dataset
    2. Clean data (drop ID column, handle missing values)
    3. Encode categorical features with LabelEncoder
    4. Train/test split (80/20, stratified)
    5. Train baseline Decision Tree + Random Forest
    6. Hyperparameter tuning with GridSearchCV for both models
    7. Evaluate & compare both tuned models
    8. Save the final Random Forest model + all encoders + metrics
       + charts needed by the Streamlit app.

Outputs (all written to ./artifacts/):
    model.pkl                 -> final tuned Random Forest model
    encoder.pkl               -> dict of fitted LabelEncoders (features + target)
    feature_columns.pkl       -> ordered list of feature column names
    metrics.json              -> all evaluation metrics for both models
    feature_importance.json   -> Random Forest feature importances
    confusion_matrix_rf.png
    confusion_matrix_dt.png
    correlation_heatmap.png
    class_distribution.png
    feature_importance.png
    roc_curve.png
    model_comparison.png
"""

import json
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")  # headless backend, safe for servers / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_and_clean,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

RANDOM_STATE = 42
DATA_PATH = "ai_student_impact_dataset.csv"
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def art(path: str) -> str:
    return os.path.join(ARTIFACTS_DIR, path)


def main():
    print("=" * 70)
    print("BURNOUT RISK PREDICTION — MODEL TRAINING PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------
    # STEP 1: Load & clean data
    # ------------------------------------------------------------
    print("\n[1/8] Loading and cleaning dataset...")
    df = load_and_clean(DATA_PATH)
    print(f"      Dataset shape after cleaning: {df.shape}")

    # ------------------------------------------------------------
    # STEP 2: Encode categorical features + target
    # ------------------------------------------------------------
    print("[2/8] Encoding categorical features...")
    encoders = {}
    for col in CATEGORICAL_FEATURE_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"      Encoded '{col}': {list(le.classes_)}")

    target_encoder = LabelEncoder()
    df[TARGET_COLUMN] = target_encoder.fit_transform(df[TARGET_COLUMN].astype(str))
    encoders[TARGET_COLUMN] = target_encoder
    print(f"      Encoded target '{TARGET_COLUMN}': {list(target_encoder.classes_)}")

    # ------------------------------------------------------------
    # STEP 3: Feature / target split
    # ------------------------------------------------------------
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    class_names = list(target_encoder.classes_)

    # ------------------------------------------------------------
    # Correlation heatmap + class distribution (EDA charts)
    # ------------------------------------------------------------
    print("[3/8] Generating EDA visualizations...")
    plt.figure(figsize=(14, 10))
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True)
    plt.title("Correlation Heatmap of All Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(art("correlation_heatmap.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    counts = df[TARGET_COLUMN].map(dict(enumerate(class_names))).value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
    plt.title("Burnout Risk Level — Class Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Burnout Risk Level")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(art("class_distribution.png"), dpi=120)
    plt.close()

    # ------------------------------------------------------------
    # STEP 4: Train/test split
    # ------------------------------------------------------------
    print("[4/8] Splitting train/test sets (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    # ------------------------------------------------------------
    # STEP 5: Baseline models
    # ------------------------------------------------------------
    print("[5/8] Training baseline Decision Tree & Random Forest...")
    dt_baseline = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_baseline.fit(X_train, y_train)

    rf_baseline = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
    rf_baseline.fit(X_train, y_train)

    # ------------------------------------------------------------
    # STEP 6: Hyperparameter tuning (GridSearchCV)
    # ------------------------------------------------------------
    print("[6/8] Running GridSearchCV hyperparameter tuning (this may take a minute)...")

    dt_param_grid = {
        "max_depth": [6, 10, None],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 4],
        "criterion": ["gini", "entropy"],
    }
    dt_grid = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=dt_param_grid, cv=3, scoring="accuracy", n_jobs=-1,
    )
    dt_grid.fit(X_train, y_train)
    print(f"      Best Decision Tree params: {dt_grid.best_params_}")

    rf_param_grid = {
        "n_estimators": [100],
        "max_depth": [10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    rf_grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=rf_param_grid, cv=3, scoring="accuracy", n_jobs=1,
    )
    rf_grid.fit(X_train, y_train)
    print(f"      Best Random Forest params: {rf_grid.best_params_}")

    best_dt = dt_grid.best_estimator_
    best_rf = rf_grid.best_estimator_

    # ------------------------------------------------------------
    # STEP 7: Evaluate tuned models
    # ------------------------------------------------------------
    print("[7/8] Evaluating tuned models...")

    def evaluate(model, name):
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        y_test_bin = label_binarize(y_test, classes=sorted(set(y)))
        try:
            roc_auc = roc_auc_score(y_test_bin, proba, average="weighted", multi_class="ovr")
        except Exception:
            roc_auc = None

        report = classification_report(y_test, preds, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, preds)

        print(f"\n      {name} PERFORMANCE (Tuned)")
        print(f"      Accuracy : {acc:.4f}")
        print(f"      Precision: {prec:.4f}")
        print(f"      Recall   : {rec:.4f}")
        print(f"      F1 Score : {f1:.4f}")
        if roc_auc:
            print(f"      ROC-AUC  : {roc_auc:.4f}")

        return {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1_score": f1, "roc_auc": roc_auc,
            "classification_report": report, "confusion_matrix": cm.tolist(),
        }, cm, proba

    dt_metrics, dt_cm, _ = evaluate(best_dt, "DECISION TREE")
    rf_metrics, rf_cm, rf_proba = evaluate(best_rf, "RANDOM FOREST")

    cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring="accuracy")
    rf_metrics["cross_val_accuracy_mean"] = float(cv_scores.mean())
    rf_metrics["cross_val_accuracy_std"] = float(cv_scores.std())
    print(f"\n      Random Forest 5-fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Confusion matrices
    for cm, name, cmap, fname in [
        (dt_cm, "Decision Tree (Tuned)", "Blues", "confusion_matrix_dt.png"),
        (rf_cm, "Random Forest (Tuned)", "Greens", "confusion_matrix_rf.png"),
    ]:
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()
        plt.savefig(art(fname), dpi=120)
        plt.close()

    # Feature importance
    importances = pd.Series(best_rf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="mako", legend=False)
    plt.title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(art("feature_importance.png"), dpi=120)
    plt.close()

    # ROC curve (one-vs-rest, macro)
    y_test_bin = label_binarize(y_test, classes=sorted(set(y)))
    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_proba[:, i])
        auc_i = roc_auc_score(y_test_bin[:, i], rf_proba[:, i])
        plt.plot(fpr, tpr, label=f"{cname} (AUC = {auc_i:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve — Random Forest (One-vs-Rest)", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(art("roc_curve.png"), dpi=120)
    plt.close()

    # Model comparison chart
    comparison_df = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest"],
        "Accuracy": [dt_metrics["accuracy"], rf_metrics["accuracy"]],
        "Precision": [dt_metrics["precision"], rf_metrics["precision"]],
        "Recall": [dt_metrics["recall"], rf_metrics["recall"]],
        "F1 Score": [dt_metrics["f1_score"], rf_metrics["f1_score"]],
    })
    melted = comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Metric", y="Score", hue="Model", data=melted, palette="Set1")
    plt.title("Decision Tree vs Random Forest — Performance Comparison", fontsize=14, fontweight="bold")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(art("model_comparison.png"), dpi=120)
    plt.close()

    # ------------------------------------------------------------
    # STEP 8: Save all artifacts
    # ------------------------------------------------------------
    print("\n[8/8] Saving model, encoders and metrics...")

    with open(art("model.pkl"), "wb") as f:
        pickle.dump(best_rf, f)

    with open(art("encoder.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    with open(art("feature_columns.pkl"), "wb") as f:
        pickle.dump(FEATURE_COLUMNS, f)

    metrics_out = {
        "decision_tree": {k: v for k, v in dt_metrics.items()},
        "random_forest": {k: v for k, v in rf_metrics.items()},
        "best_dt_params": dt_grid.best_params_,
        "best_rf_params": rf_grid.best_params_,
        "comparison_table": comparison_df.to_dict(orient="records"),
        "class_names": class_names,
        "dataset_shape": list(df.shape),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "final_model": "Random Forest Classifier",
    }
    with open(art("metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open(art("feature_importance.json"), "w") as f:
        json.dump(importances.to_dict(), f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE.")
    print(f"Final model: Random Forest  |  Test Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"All artifacts saved to ./{ARTIFACTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
