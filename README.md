# FIFA Player Value Prediction

This repository contains an end-to-end pipeline for predicting FIFA player market values using advanced data preprocessing and an XGBoost-based predictive model. This project was developed as a university project at Teesside University and serves as a comprehensive demonstration of data cleaning, feature engineering, and machine learning model development applied to sports analytics.

## Overview

The primary goal of this project is to accurately predict player market values by leveraging the rich FIFA dataset. The project involves rigorous data preprocessing, including handling missing values, encoding categorical features, and extracting numeric ratings from complex text fields. An XGBoost model is then trained on the processed data to capture non-linear relationships and deliver precise market value predictions.

## Project Structure

The repository is organized as follows:

- **data-preprocessing.R**  
  Contains the complete script for cleaning and preprocessing the raw FIFA dataset. This script includes tasks such as imputation of missing values, feature engineering (e.g., generating player-specific metrics), one-hot encoding, and preparation of training and testing datasets.

- **xgb_model.R**  
  Implements the model training and evaluation pipeline using XGBoost. It merges training and testing data for consistent encoding, applies a logarithmic transformation to the target variable, splits the data into training and validation sets, and performs model training with early stopping. Detailed metrics (such as MAE, RMSE, and R²) and feature importance analysis are also provided.


## Setup and Installation

Before running the scripts, ensure you have the required R packages installed. The following libraries are used throughout the project:

- **Data Manipulation & Visualization:** tidyverse, lubridate, corrplot, ggplot2  
- **Data Preprocessing & Modeling:** modeest, fastDummies, caret, dplyr  
- **Modeling:** xgboost, Matrix, doParallel

You can install any missing packages using the R command:
```r
install.packages(c("tidyverse", "lubridate", "modeest", "fastDummies", "caret", "corrplot", "dplyr", "xgboost", "Matrix", "doParallel", "ggplot2"))
```

## How to Run

1. **Data Preprocessing:**  
   Run the `data-preprocessing.R` script. This script reads the FIFA dataset, cleans and preprocesses the data, performs feature engineering, and generates the training and testing CSV files required for modeling.

2. **Model Training and Evaluation:**  
   Once the data is prepared, run the `xgb_model.R` script. This script loads the processed data, applies one-hot encoding and scaling, splits the data into training, validation, and testing sets, and then trains an XGBoost model with early stopping. It also evaluates model performance by reporting key metrics and generating visualizations of the predicted versus actual values and feature importance.


## Methodology

This project follows a systematic approach:
- **Data Preprocessing:** Rigorous cleaning of the dataset to handle missing values, correct inconsistencies, and transform complex text attributes (e.g., "82+3" to a numeric rating).
- **Feature Engineering:** Creation of derived metrics such as aggregated positional ratings and customized features tailored to the specifics of football analytics.
- **Predictive Modeling:** Implementation of an XGBoost model, optimized with hyperparameter tuning and early stopping, to predict player market values. The model evaluation includes both log-transformed and original scale metrics.
- **Visualization and Analysis:** Generation of performance metrics and visualizations to assess the model’s predictive accuracy and to understand feature importance.

## Results

The model achieves high performance with:
- **Mean Absolute Error (MAE):** Reflecting the average prediction error in both logarithmic scale and original EUR values.
- **Root Mean Squared Error (RMSE):** Providing insights into the model's prediction variance.
- **R² Score:** Indicating the proportion of variance explained by the model.

The feature importance analysis further validates the model by highlighting key predictors such as overall ratings and release clause values.

## Acknowledgments

This project is a university project completed as part of the curriculum at Teesside University. It has provided invaluable insights into the challenges and solutions in sports analytics, and it showcases the practical application of advanced data preprocessing and machine learning techniques.
