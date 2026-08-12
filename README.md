# Mathine Test — ML.NET Machine Learning Education

A hands-on C# / ML.NET project for learning Machine Learning through controlled experiments.

The project starts with a simple synthetic classification problem and gradually evolves into a complete ML experimentation pipeline:

Dataset Generation
→ Feature Engineering
→ Training
→ Evaluation
→ Cross Validation
→ Hyperparameter Tuning
→ Generalization
→ Distribution Shift
→ Concept Shift
→ Data Drift Detection
→ Noise Robustness
→ Noise Augmentation
→ Computer Vision
→ Audio / Spectrograms
→ Real-Time ML

The goal is not simply to achieve the highest possible Accuracy.

The goal is to understand **why a model works, when it fails, how stable it is, and how it behaves when the real world changes.**

---

# 🚀 Current Progress

The current project uses a synthetic binary classification problem.

The model classifies people as:

- `Normal`
- `Overweight`

The dataset contains:

- `Height`
- `Weight`
- `IsAthlete`
- `BMI`
- `RandomFeatures`

The current best feature set is:

```text
Weight + IsAthlete
