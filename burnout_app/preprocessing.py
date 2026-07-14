"""
preprocessing.py
================================================================
Shared preprocessing utilities for the Burnout Risk Prediction
project. Used by BOTH train_model.py (training time) and
predict.py / app.py (inference time) so that the exact same
encoding logic is applied everywhere.

Dataset : AI Student Impact Dataset (ai_student_impact_dataset.csv)
Target  : Burnout_Risk_Level (Low / Medium / High)
================================================================
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------
# Column definitions (derived directly from the source notebook)
# ----------------------------------------------------------------

# Column that is dropped before modeling (pure identifier, no signal)
ID_COLUMN = "Student_ID"

# Target column
TARGET_COLUMN = "Burnout_Risk_Level"

# Categorical columns that were passed through sklearn's LabelEncoder
# in the notebook (object/bool dtype columns, excluding the target,
# which is encoded separately since it is the label).
CATEGORICAL_FEATURE_COLUMNS = [
    "Major_Category",
    "Year_of_Study",
    "Primary_Use_Case",
    "Prompt_Engineering_Skill",
    "Paid_Subscription",
    "Institutional_Policy",
]

# Purely numeric columns (already numeric in the raw CSV)
NUMERICAL_FEATURE_COLUMNS = [
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Tool_Diversity",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams",
    "Post_Semester_GPA",
    "Skill_Retention_Score",
]

# Final feature order fed into the model (must stay consistent
# between training and inference)
FEATURE_COLUMNS = [
    "Major_Category",
    "Year_of_Study",
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Primary_Use_Case",
    "Prompt_Engineering_Skill",
    "Tool_Diversity",
    "Paid_Subscription",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Institutional_Policy",
    "Anxiety_Level_During_Exams",
    "Post_Semester_GPA",
    "Skill_Retention_Score",
]

# Human-friendly metadata used to auto-build Streamlit widgets.
# min/max/mean values below were computed directly from the
# uploaded dataset (50,000 rows) and are NOT placeholders.
NUMERIC_FIELD_META = {
    "Pre_Semester_GPA": {"min": 1.18, "max": 4.0, "mean": 3.15, "step": 0.01,
                          "label": "Pre-Semester GPA", "help": "Student's GPA before the semester started (scale 0-4)."},
    "Weekly_GenAI_Hours": {"min": 0.0, "max": 40.0, "mean": 8.43, "step": 0.1,
                            "label": "Weekly GenAI Usage (hours)", "help": "Hours per week spent using Generative AI tools for academics."},
    "Tool_Diversity": {"min": 1, "max": 5, "mean": 3, "step": 1,
                        "label": "Tool Diversity", "help": "Number of distinct GenAI tools used regularly (1-5)."},
    "Traditional_Study_Hours": {"min": 0.0, "max": 36.0, "mean": 11.21, "step": 0.1,
                                 "label": "Traditional Study Hours (weekly)", "help": "Hours per week spent studying without AI assistance."},
    "Perceived_AI_Dependency": {"min": 1, "max": 10, "mean": 4, "step": 1,
                                 "label": "Perceived AI Dependency", "help": "Self-rated dependency on AI tools (1 = low, 10 = high)."},
    "Anxiety_Level_During_Exams": {"min": 1, "max": 10, "mean": 4, "step": 1,
                                    "label": "Anxiety Level During Exams", "help": "Self-rated exam anxiety (1 = calm, 10 = extreme anxiety)."},
    "Post_Semester_GPA": {"min": 0.0, "max": 4.0, "mean": 3.35, "step": 0.01,
                           "label": "Post-Semester GPA", "help": "Student's GPA after the semester ended (scale 0-4)."},
    "Skill_Retention_Score": {"min": 10.0, "max": 100.0, "mean": 75.8, "step": 0.1,
                               "label": "Skill Retention Score", "help": "Score (0-100) measuring how well core skills were retained without AI help."},
}

CATEGORICAL_FIELD_META = {
    "Major_Category": {
        "options": ["Arts", "Business", "Humanities", "Medical", "STEM"],
        "label": "Major Category",
    },
    "Year_of_Study": {
        "options": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
        "label": "Year of Study",
    },
    "Primary_Use_Case": {
        "options": [
            "Copywriting/Drafting",
            "Debugging/Troubleshooting",
            "Direct_Answer_Generation",
            "Ideation",
            "Summarizing_Reading",
        ],
        "label": "Primary GenAI Use Case",
    },
    "Prompt_Engineering_Skill": {
        "options": ["Beginner", "Intermediate", "Advanced"],
        "label": "Prompt Engineering Skill",
    },
    "Paid_Subscription": {
        "options": ["False", "True"],
        "label": "Has Paid AI Subscription?",
    },
    "Institutional_Policy": {
        "options": ["Strict_Ban", "Allowed_With_Citation", "Actively_Encouraged"],
        "label": "Institutional AI Policy",
    },
}

TARGET_CLASSES = ["High", "Low", "Medium"]  # alphabetical order (LabelEncoder default)


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and apply the same cleaning steps as the notebook:
    drop the ID column and fill any missing values defensively."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[ID_COLUMN], errors="ignore")

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode_dataframe(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply already-fitted LabelEncoders (from training) to a dataframe
    that still has raw categorical string values."""
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    return df


def encode_single_input(raw_input: dict, encoders: dict) -> pd.DataFrame:
    """
    Convert a single raw user input (dict of feature_name -> raw value,
    exactly as collected from the Streamlit prediction form) into a
    single-row, model-ready DataFrame with the correct column order
    and encodings.
    """
    row = {}
    for col in FEATURE_COLUMNS:
        value = raw_input[col]
        if col in encoders:
            le = encoders[col]
            value = le.transform([str(value)])[0]
        row[col] = value

    ordered = {col: row[col] for col in FEATURE_COLUMNS}
    return pd.DataFrame([ordered])
