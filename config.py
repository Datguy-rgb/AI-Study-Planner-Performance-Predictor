"""Configuration variables and system constraints."""

# Feature for student profiles
FEATURE_NAMES = [
    "previous_marks",          
    "attendance_pct",          
    "weekly_study_hours",      
    "assignments_completion",  
    "test_scores_avg",         
    "sleep_hours_avg",         
    "ds_mastery",              
    "prob_mastery",            
    "other_mastery"            
]

# Classification labels for the Logistic Regression model
RISK_CLASSES = ["Low Risk", "Medium Risk", "High Risk"]

# Agent threshold for identifying weak subjects
WEAKNESS_THRESHOLD = 75.0