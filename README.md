# Fatigue Prediction Prototype using Synthetic Data

## Overview

This project presents a prototype that uses synthetic data to simulate wearable device data for predicting fatigue levels. It uses a RandomForest model to classify fatigue as high or low.

## Dataset

Synthetic data was generated to mimic features commonly collected by wearable devices:

* Heart Rate (bpm)
* Skin Temperature (°C)
* Electrodermal Activity (EDA)
* Activity Level (arbitrary scale 0–10)
* Sleep Duration (hours)
* Steps (per day)

Target label:

* Fatigue Level: 0 (Low), 1 (High)

## Pipeline

1. **Data Generation**: Synthetic data creation using NumPy.
2. **Train/Test Split**: Split data before preprocessing.
3. **Preprocessing**:
   * Standardization using `StandardScaler`
4. **Model Training**:
   * Train Random Forest model on all features.
5. **Feature Selection**:
   * Select top 4 features based on `feature_importances_`
6. **Final Model Training**:
   * Retrain Random Forest on selected features
7. **Evaluation**:
   * Accuracy, confusion matrix, classification report
   * Feature importance visualization

## Results
* The model shows separation of fatigue levels using selected features.
* Visualizations of confusion matrix and feature importance is added.

## Optional: Integration with Digital Twin

This model can be integrated into a Digital Twin for health monitoring:

* Continuous ingestion of live data from wearables (e.g., Garmin, Fitbit)
* Predict fatigue risk in real-time
* Visual suggestion on user’s digital twin avatar to suggest hydration or rest

## Requirements

* Python 3.x
* Libraries: pandas, numpy, sklearn, seaborn, matplotlib