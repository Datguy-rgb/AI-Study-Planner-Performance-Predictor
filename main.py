"""Main application entry point orchestrating data generation, training, and inference."""

import numpy as np
from config import RISK_CLASSES, WEAKNESS_THRESHOLD
from data_generator import generate_student_dataset
from models import PerformancePredictorPipeline
from agent import IntelligentStudyAgent

def main():
    print("\n--- AI STUDY PLANNER & PERFORMANCE PREDICTOR ---")
    
    # 1. Initialization and Training
    X, y_score, y_risk = generate_student_dataset(n_samples=800)
    pipeline = PerformancePredictorPipeline()
    metrics = pipeline.train(X, y_score, y_risk)

    print("\n[+] System Trained Successfully")
    print(f"    Regression R2: {metrics['reg_r2']:.3f}")
    print(f"    Classification Accuracy: {metrics['clf_acc'] * 100:.2f}%")

    # 2. Define Target Student Profile
    student_profile = {
        "previous_marks": 72.0,
        "attendance_pct": 82.0,
        "weekly_study_hours": 14.0,
        "assignments_completion": 88.0,
        "test_scores_avg": 70.0,
        "sleep_hours_avg": 6.5,
        "ds_mastery": 52.0,       # High deficit
        "prob_mastery": 61.0,     # Moderate deficit
        "other_mastery": 80.0     # Low deficit
    }
    available_weekly_hours = 12.0

    raw_features = np.array([[
        student_profile["previous_marks"], student_profile["attendance_pct"],
        student_profile["weekly_study_hours"], student_profile["assignments_completion"],
        student_profile["test_scores_avg"], student_profile["sleep_hours_avg"],
        student_profile["ds_mastery"], student_profile["prob_mastery"],
        student_profile["other_mastery"]
    ]])

    # 3. Execute ML Inference
    pred = pipeline.predict(raw_features)
    risk_label = RISK_CLASSES[pred["risk_index"]]

    # 4. Execute Agent Diagnostics
    agent = IntelligentStudyAgent(weakness_threshold=WEAKNESS_THRESHOLD)
    subject_map = {
        "Data Structures": student_profile["ds_mastery"],
        "Probability": student_profile["prob_mastery"],
        "Other Subjects": student_profile["other_mastery"]
    }
    weaknesses = agent.diagnose_weaknesses(subject_map)
    schedule = agent.allocate_study_hours(subject_map, available_weekly_hours)

    # 5. Output Generation
    print("\n--- INFERENCE REPORT ---")
    print(f"Projected Score   : {pred['expected_score_range'][0]}% - {pred['expected_score_range'][1]}%")
    print(f"Academic Risk     : {risk_label}")
    print(f"Student Archetype : Cluster {pred['cluster_id']}")
    print(f"Critical Subjects : {', '.join(weaknesses)}")
    
    print("\n--- OPTIMIZED STUDY SCHEDULE ---")
    for subject, alloc in schedule.items():
        print(f"{subject:<18} : {alloc['hours']:>5.2f} hrs ({alloc['percentage']}%)")
    print("------------------------------------------------\n")

if __name__ == "__main__":
    main()