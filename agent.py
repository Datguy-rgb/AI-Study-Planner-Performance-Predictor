"""Intelligent Study Agent for resource allocation based on deficit evaluation."""

from typing import Dict, List

class IntelligentStudyAgent:
    def __init__(self, weakness_threshold: float = 70.0):
        self.weakness_threshold = weakness_threshold

    def diagnose_weaknesses(self, subject_scores: Dict[str, float]) -> List[str]:
        """Identifies subjects falling below the mastery threshold."""
        return [subj for subj, score in subject_scores.items() if score < self.weakness_threshold]

    def allocate_study_hours(self, subject_scores: Dict[str, float], total_hours: float) -> Dict[str, Dict[str, float]]:
        """
        Calculates optimal study allocations using a quadratic deficit formula.
        Prioritizes subjects with the lowest mastery scores.
        """
        if total_hours <= 0:
            raise ValueError("Available hours must be greater than zero.")

        deficits = {subj: (100.0 - score) ** 2 for subj, score in subject_scores.items()}
        total_deficit = sum(deficits.values())

        if total_deficit == 0:
            allocations = {s: 1.0 / len(subject_scores) for s in subject_scores}
        else:
            allocations = {s: (d / total_deficit) for s, d in deficits.items()}

        schedule = {}
        for subject, fraction in allocations.items():
            schedule[subject] = {
                "percentage": round(fraction * 100, 1),
                "hours": round(fraction * total_hours, 2)
            }
        return schedule