# AI Study Planner & Performance Predictor

## 📖 About This Project
This is my flipped course project for CSA2001 (Fundamentals in AI and ML). I wanted to build something genuinely useful for students. Basically, it's an AI-driven study advisor. Instead of just guessing how much time you should spend on Data Structures versus Probability, this tool takes your past marks, attendance, and study habits, predicts how you'll do, and then acts as an intelligent agent to build a custom study schedule for you[cite: 2].

**Author:** Aman Kumar Singh (25MIM10175)

## ✨ What It Actually Does
I built this using a few core concepts from our syllabus to hit the functional requirements[cite: 1, 2]:
* **Grade & Risk Prediction:** Uses Linear Regression to guess your final score based on your current stats, and Logistic Regression to flag if you're at High, Medium, or Low risk of failing[cite: 2].
* **Student Profiling:** Runs K-Means clustering to see what "type" of student profile you fit into based on habits like sleep and attendance[cite: 2].
* **Smart Scheduling Agent:** This is the cool part. I wrote a utility-based agent that looks at your weakest subjects and your available free time, then uses a math formula (quadratic deficit) to prioritize and divide up your study hours for the week[cite: 2].

## 🛠 Tech Stack
* **Language:** Python 3.10+
* **Libraries:** `scikit-learn` (for all the ML models), `numpy` (for handling the math and making the dummy dataset)[cite: 1]
* **Version Control:** Git & GitHub[cite: 1]

## 🚀 How to Run It
If you want to test it out on your machine, just follow these steps[cite: 1]:

1. Clone the repo:
   ```bash
   git clone [https://github.com/Datguy-rgb/AI-Study-Planner-Performance-Predictor.git](https://github.com/Datguy-rgb/AI-Study-Planner-Performance-Predictor.git)
   cd study-planner-ai