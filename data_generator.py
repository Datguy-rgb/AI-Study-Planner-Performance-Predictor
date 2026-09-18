"""Generates synthetic training data mapping to statistical distributions."""

import numpy as np

def generate_student_dataset(n_samples: int = 800, random_seed: int = 42):
    """
    Creates a dataset with linear dependencies, Gaussian noise, and categorical targets.
    """
    np.random.seed(random_seed)

    # Independent variables
    previous_marks = np.random.uniform(40, 95, n_samples)
    attendance_pct = np.random.uniform(55, 100, n_samples)
    weekly_study_hours = np.random.uniform(5, 35, n_samples)
    assignments_completion = np.random.uniform(50, 100, n_samples)
    
    # Dependent/correlated variables
    test_scores_avg = np.clip((0.7 * previous_marks) + np.random.normal(0, 5, n_samples), 0, 100)
    sleep_hours_avg = np.random.uniform(4.5, 9.0, n_samples)

    ds_mastery = np.clip(previous_marks + np.random.normal(0, 10, n_samples), 10, 100)
    prob_mastery = np.clip(previous_marks + np.random.normal(0, 12, n_samples), 10, 100)
    other_mastery = np.clip(previous_marks + np.random.normal(0, 8, n_samples), 10, 100)

    # Feature Matrix (X)
    X = np.column_stack([
        previous_marks, attendance_pct, weekly_study_hours,
        assignments_completion, test_scores_avg, sleep_hours_avg,
        ds_mastery, prob_mastery, other_mastery
    ])

    # Target 1: Continuous Expected Score (Linear Regression)
    expected_score = (
        0.25 * previous_marks +
        0.15 * attendance_pct +
        0.60 * weekly_study_hours +
        0.10 * assignments_completion +
        0.20 * test_scores_avg +
        np.random.normal(0, 2.5, n_samples)
    )
    y_score = np.clip(expected_score, 0, 100)

    # Target 2: Categorical Risk Level (Logistic Regression)
    y_risk = np.where(y_score >= 75, 0, np.where(y_score >= 55, 1, 2))

    return X, y_score, y_risk