# Mathine Test — ML.NET Console Sandbox

A C# console application designed for machine learning experiments using ML.NET. The project generates a synthetic dataset, trains a classification model, and provides tools for making predictions and analyzing feature performance.

## 🚀 Quick Start

### Prerequisites
* **.NET 8.0 SDK** (or newer)
* **JetBrains Rider** or Visual Studio / VS Code

### Running the Project
1. Clone the repository:
   ```bash
   git clone KriksalisProgrammer/MLEducation
   cd "Mathine test/Mathine test"
   ```
2. Run the application:
   ```bash
   dotnet run
   ```

---

## 🛠 Features & Modes

Upon launching the application, an interactive menu allows you to choose from four operational modes:

| Option | Mode | Description |
| :--- | :--- | :--- |
| **`1`** | **Train** | Generates `Data/people.csv` (10,000 records), trains the ML model, and saves weights to `Data/model.zip`. |
| **`2`** | **Predict** | Loads `Data/model.zip` and predicts outcomes based on user input (Height, Weight, Athlete status). |
| **`3`** | **Compare Features** | Analyzes and compares the contribution of individual features to model performance. |
| **`4`** | **Multiple Experiments** | Runs multiple automated experimental runs to evaluate model stability. |

---

## 📂 Architecture & Key Components

* **`MlService`**: Core machine learning service responsible for training pipelines, saving/loading `.zip` models, running predictions, and feature analysis.
* **`DataGenerator`**: Utility class for generating synthetic sample datasets (`Data/people.csv`).
* **`PersonData` / `Models`**: Data structures defining input features (`Height`, `Weight`, `isAtlete`) and output prediction scores.

---

## 📊 Data Schema

* **`Height`** (`float`) — Individual's height.
* **`Weight`** (`float`) — Individual's weight.
* **`isAtlete`** (`bool`) — Boolean flag indicating athlete status.

> **Note:** Сreated datasets (`.csv`) and trained model files (`.zip`) are saved in the `Data/` directory and excluded via `.gitignore`.
