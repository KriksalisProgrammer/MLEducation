# Mathine Test — ML.NET Machine Learning Education

A hands-on C# / ML.NET project for learning machine learning through practical experiments.

The project starts with a simple synthetic classification problem and progressively explores the complete ML workflow:

**Dataset → Features → Training → Evaluation → Cross Validation → Hyperparameter Tuning → Final Validation**

The goal is not just to achieve a high accuracy score, but to understand **why a model performs well, which features matter, how stable the result is, and how training parameters affect the model.**

---

## 🚀 Current Progress

The current experiment uses a synthetic dataset for classifying people into:

* `Normal`
* `Overweight`

The best feature set discovered so far is:

```text
Weight + IsAthlete
```

After hyperparameter grid search, the best configuration achieved:

| Metric                    |     Result |
| ------------------------- | ---------: |
| Cross-Validation Accuracy | **99.43%** |
| Learning Rate             |   **0.05** |
| L2 Regularization         | **0.0001** |
| Iterations                |    **500** |

> ⚠️ The 99.43% score is a cross-validation result. Final validation on completely unseen data is the next step.

---

## 🧠 What We Are Learning

This project is being developed as a practical ML learning path.

### 1. Dataset Generation

Synthetic datasets are generated automatically with controlled features and labels.

Current features include:

```text
Height
Weight
IsAthlete
BMI
RandomFeatures
```

This allows us to intentionally introduce useful, redundant and completely random features and observe how the model reacts.

### 2. Feature Engineering

We compare different feature combinations:

```text
Weight
Weight + IsAthlete
Weight + IsAthlete + BMI
Weight + IsAthlete + Random Features
```

This demonstrated an important ML principle:

> More features do not necessarily mean a better model.

### 3. Model Evaluation

The project evaluates models using:

* Accuracy
* Micro Accuracy
* Macro Accuracy
* Precision
* Recall
* F1 Score
* LogLoss
* Confusion Matrix
* Train/Test Gap
* Standard Deviation

### 4. Cross Validation

10-Fold Cross Validation is used to evaluate model stability across different data splits.

Example:

```text
Average Accuracy: 98.06%
StdDev Accuracy:  0.62%
Average LogLoss:   0.1833
```

### 5. Hyperparameter Tuning

A grid search automatically evaluates different training configurations.

Parameters being explored include:

```text
Learning Rate
L2 Regularization
Maximum Iterations
```

The current best configuration:

```text
Learning Rate:     0.05
L2 Regularization: 0.0001
Iterations:        500

Accuracy:          99.43%
```

---

## 🛠 Technologies

* C#
* .NET 8
* ML.NET
* JetBrains Rider / Visual Studio
* Git / GitHub

---

## 🚀 Quick Start

### Requirements

* .NET 8 SDK or newer
* Rider, Visual Studio or VS Code

### Clone

```bash
git clone https://github.com/KriksalisProgrammer/MLEducation.git
cd MLEducation
```

### Run

```bash
dotnet run
```

---

## 🎮 Available Experiments

The console application provides several experimental modes.

### Train

Generates the dataset, trains the model and evaluates its performance.

### Predict

Loads the trained model and predicts the class for a new person.

Example input:

```text
Height: 180
Weight: 85
Athlete: true
```

### Feature Comparison

Compares different feature combinations and measures their effect on model performance.

### Multiple Experiments

Repeats experiments with different datasets / configurations to check whether the results are stable.

### Cross Validation

Runs 10-Fold Cross Validation and reports:

```text
Average Accuracy
Standard Deviation
Average LogLoss
```

### Hyperparameter Tuning

Runs an automated grid search over different model configurations and identifies the best parameters.

---

## 📂 Project Structure

```text
MLEducation/
│
├── Experiments/
│   ├── CrossValidationExperiment.cs
│   ├── CrossValidationFeatureExperiment.cs
│   └── HyperparameterExperiment.cs
│
├── Models/
│   └── PersonData.cs
│
├── Services/
│   └── MlService.cs
│
├── DataGenerator.cs
├── Program.cs
├── Mathine test.csproj
└── README.md
```

---

## 📊 Experimental Results

### Feature Comparison

Current results show that `Weight + IsAthlete` performs better than using Weight alone.

```text
Weight
≈ 93%

Weight + IsAthlete
≈ 98%

Weight + IsAthlete + BMI
≈ 96%
```

This experiment demonstrated that adding a mathematically meaningful feature does not automatically improve a machine learning model.

---

## 🔬 Learning Roadmap

### Completed

* [x] Synthetic dataset generation
* [x] Binary classification
* [x] Basic ML.NET training
* [x] Train/Test evaluation
* [x] Confusion Matrix
* [x] Precision / Recall / F1
* [x] Feature comparison
* [x] Random feature experiments
* [x] 10-Fold Cross Validation
* [x] Cross-validation feature comparison
* [x] Hyperparameter experiments
* [x] Hyperparameter Grid Search
* [x] Best configuration discovery

### Next

* [ ] Final validation on completely unseen data
* [ ] Model calibration and probability analysis
* [ ] Automated experiment result storage
* [ ] More advanced ML algorithms
* [ ] Real-world datasets
* [ ] Image classification
* [ ] Computer vision
* [ ] Object detection
* [ ] Video analysis
* [ ] Audio classification
* [ ] Spectrogram-based ML
* [ ] Real-time inference

---

## 🎯 Long-Term Goal

The project will gradually move from simple tabular classification toward real-world machine learning problems.

The planned progression is:

```text
Tabular Data
     ↓
Feature Engineering
     ↓
Model Evaluation
     ↓
Hyperparameter Optimization
     ↓
Real Datasets
     ↓
Images
     ↓
Computer Vision
     ↓
Video
     ↓
Audio / Spectrograms
     ↓
Real-Time ML
```

The purpose of the project is to learn each stage rather than immediately jumping to a complex neural network.

---

## 📚 Philosophy

This project focuses on experimentation.

Instead of simply training a model and looking at Accuracy, every major change is tested and compared.

The main questions are:

* Which features actually help?
* Which features add noise?
* Is the model overfitting?
* How stable is the result?
* How does cross-validation change our conclusions?
* Which hyperparameters produce the best model?
* Does the model perform equally well on completely unseen data?

---

## 👨‍💻 Author

**KriksalisProgrammer**

This repository is a personal machine learning education project built through incremental experiments with C# and ML.NET.
