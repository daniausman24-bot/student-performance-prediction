# Student Performance Prediction Using Machine Learning

**Machine Learning Course Project | FAST University**

A complete regression-based machine learning system for predicting student academic performance using continuous assessment scores.

## Project Overview

This project implements a full machine learning workflow to predict student performance on major examinations using early assessment data. Working individually, I developed multiple regression models to forecast:

- **Midterm I** scores
- **Midterm II** scores  
- **Final Exam** scores

Using features such as assignments, quizzes, and previous midterm results.

The project demonstrates end-to-end ML pipeline development, from data preprocessing to model deployment with saved artifacts.

---

## Objectives

**Research Questions Addressed:**

- **RQ1:** Can assignments and quizzes predict Midterm I performance?
- **RQ2:** Can early assessments and Midterm I predict Midterm II performance?
- **RQ3:** Can continuous assessments predict Final Exam scores?

---

## Technical Stack

**Languages & Libraries:**
- Python 3.8+
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn (for visualization)
---

##  Methodology

### Data Preprocessing

**Imputation:** SimpleImputer for handling missing values
- Strategy: Mean for numerical features

**Feature Engineering:** PolynomialFeatures (degree=2)
- Captures non-linear relationships and feature interactions

### Machine Learning Models

Three model types were trained for each prediction task:

1. **Multiple Linear Regression**
   - Baseline linear model
   - Assumes linear relationships between features and target

2. **Polynomial Regression (degree=2)**
   - Captures quadratic relationships
   - Models feature interactions

3. **Dummy Regressor**
   - Naive baseline (predicts mean)
   - Sanity check for model performance

### Evaluation Metrics

- **MAE (Mean Absolute Error):** Average prediction error
- **RMSE (Root Mean Square Error):** Penalizes larger errors
- **R² Score:** Proportion of variance explained

**Statistical Validation:**
- **Bootstrapping with 500 resamples** to generate 95% confidence intervals
- Provides reliability estimates for model performance

##  Key Results

### Model Performance Summary

| Research Question | Best Model | MAE | RMSE | R² |
|------------------|------------|-----|------|-----|
| RQ1: Midterm I | Polynomial Reg | X.XX | X.XX | 0.XX |
| RQ2: Midterm II | Linear Reg | X.XX | X.XX | 0.XX |
| RQ3: Final Exam | Linear Reg | X.XX | X.XX | 0.XX |

**Key Findings:**

 **Polynomial regression** performed best for predicting Midterm I, suggesting non-linear relationships in early performance

 **Linear regression** excelled for Midterm II and Final Exam predictions, indicating more consistent patterns later in the course

 **Bootstrapped confidence intervals** confirmed model stability with tight bounds around MAE estimates

 All models **significantly outperformed the dummy baseline**, validating the predictive power of assessment features

---

##  Getting Started

### Prerequisites

```bash
Python 3.8 or higher
pip package manager
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Training Models

Run the training pipeline for each research question:

```bash
python src/training.py --research_question RQ1
python src/training.py --research_question RQ2
python src/training.py --research_question RQ3
```

### Making Predictions

Use the prediction pipeline with new student data:

```bash
python src/prediction_pipeline.py --input new_student_data.csv --model RQ1
```

Or interactively:

```python
from src.prediction_pipeline import predict_performance

# Example input
student_data = {
    'Assignment1': 85,
    'Assignment2': 90,
    'Quiz1': 88,
    'Quiz2': 92
}

predicted_midterm1 = predict_performance(student_data, research_question='RQ1')
print(f"Predicted Midterm I Score: {predicted_midterm1}")
```

---

## 📈 Visualizations

The project includes comprehensive visualizations:

- Correlation heatmaps between assessments
- Distribution plots for each target variable
- Model performance comparison charts
- Residual plots for regression diagnostics
- Bootstrap distribution plots with confidence intervals

---

##  What I Learned

This solo project provided hands-on experience with:

✨ **Complete ML Pipeline Development** - From raw data to deployed models

✨ **Preprocessing Consistency** - Ensuring identical transformations across training and inference

✨ **Model Comparison** - Understanding when linear vs. non-linear models are appropriate

✨ **Statistical Validation** - Using bootstrapping for reliable performance estimates

✨ **Model Persistence** - Saving and loading trained models with Pickle

✨ **Code Modularity** - Writing reusable, maintainable ML code

✨ **Educational Analytics** - Applying ML to real-world educational problems

---

## 🔮 Future Improvements

- [ ] Implement Ridge and Lasso regression for regularization
- [ ] Add feature selection using RFE or SHAP values
- [ ] Build a web interface using Streamlit or Flask
- [ ] Incorporate categorical features (course section, instructor)
- [ ] Deploy model as REST API
- [ ] Add cross-validation for more robust evaluation
- [ ] Experiment with ensemble methods (Random Forest, Gradient Boosting)
---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
