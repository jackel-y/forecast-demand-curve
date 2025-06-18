# Forecast Demand Curve

## Project Overview

This repository hosts a robust Flight Sales Forecasting System designed to predict daily seat sales for flights across their entire booking window (from 341 days before departure down to the day of departure). The system prioritizes the use of actual observed sales data when available and dynamically generates predictions for unobserved days using a trained XGBoost model.

The project addresses the critical need for accurate demand forecasting in the airline industry, enabling better capacity planning, pricing strategies, and revenue management.

## Key Features

* **Data Preprocessing and Feature Engineering:**
    * Creation of essential time-based features (e.g., day of week, day of year, month, year, quarter, etc.) from `Date`.
    * One-hot encoding to capture route-specific demand patterns.
    * Generation of crucial **lagged `corrected_sales` features** (e.g., 1, 2, and 7 days prior) to incorporate historical sales momentum into predictions.
* **Sequential Forecasting Logic:** Implements a specialized prediction loop that iterates backward from the furthest point in the booking window (341 days out) down to the departure day. This allows the model to leverage previously observed actual sales or prior day's predictions as input for future predictions, simulating a real-world forecasting scenario.
* **XGBoost Model Training & Evaluation:**
    * Utilizes the powerful XGBoost Regressor for its ability to handle complex, non-linear relationships and interactions within the data.
    * Includes hyperparameter tuning and cross-validation using `GridSearchCV` to find optimal model parameters.
    * Evaluates model performance using standard regression metrics like RMSE, MAE, and R-squared on a dedicated hold-out test set.
* **Model Persistence:** Provides functionality to save and load the trained XGBoost model using `joblib` or XGBoost's native format, enabling efficient deployment and future use without retraining.
* **Output Formats:** Generates forecasts in both wide format (each day as a column) and long format (each day as a row), catering to different analytical and reporting needs.

## Repository Structure

```

FORECAST-DEMAND-CURVE/
├── .venv/                      # Python virtual environment
├── data/                       # Stores raw and processed data
│   ├── dataset.csv             # Raw input data for training
│   └── output.csv              # Input data for generating new forecasts (with some actuals)
├── notebooks/                  # Jupyter notebooks for experimentation and analysis
│   ├── discovery.ipynb         # Initial data exploration or model discovery
│   ├── execute.ipynb           # Main execution notebook (likely contains the full prediction pipeline)
├── src/                        # Source code for modular functions
│   ├── pycache/            # Python bytecode cache
│   ├── trained_models/         # Directory to store trained models
│   │   └── xgb_model_joblib.pkl  # Example of a saved XGBoost model
│   └── nodes.py                # Contains core functions like feature_engineering and prediction logic
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── README.md                   # This README file
└── requirements.txt            # Python dependencies for the project

```

## Getting Started

Follow these instructions to set up the project and run the forecasting system.

### Prerequisites

* Python 3.x (Python 3.9.6 or 3.13.5 recommended based on project)
* Git

### 1. Clone the Repository


git clone [https://github.com/jackel-y/FORECAST-DEMAND-CURVE.git](https://github.com/jackel-y/FORECAST-DEMAND-CURVE.git)

### 2. Set up the Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

### 3. Install Dependencies
Install the required Python packages:
pip install -r requirements.txt

### 4. Data Preparation
Place the data for new forecasts (which might include partial actuals) in data/output.csv. 

### 5. Run the Forecasting Pipeline
The main forecasting pipeline is typically orchestrated within the execute.ipynb notebook or a dedicated script.

