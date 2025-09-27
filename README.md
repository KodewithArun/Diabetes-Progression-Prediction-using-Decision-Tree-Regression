# Diabetes Progression Prediction using Decision Tree Regression

This project implements a Decision Tree Regression model to predict diabetes disease progression one year after baseline, using the well-known Diabetes dataset (442 samples, 10 features).

## Features

- **Data Preprocessing:** Clean and prepare the dataset for modeling.
- **Model Training:** Train a Decision Tree Regressor on the processed data.
- **Evaluation:** Assess model performance using metrics such as Mean Squared Error (MSE) and R² score.
- **Reproducibility:** Modular code structure for easy experimentation and reproducibility.

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Diabetes-Progression-Prediction-using-Decision-Tree-Regression
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the main script:**
    ```bash
    python main.py
    ```

## Dataset

- The Diabetes dataset is available in `sklearn.datasets`.
- Features: 10 baseline variables (age, sex, BMI, blood pressure, etc.)
- Target: Disease progression after one year.

## Results

- The model's performance is evaluated and reported in the console after training.

## Project Structure

```
.
├──Diabetes Progression Prediction using Decision Tree Regression.ipynb
├── application.py
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License.

