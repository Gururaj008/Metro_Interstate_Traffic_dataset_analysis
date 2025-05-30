# Data-Driven Traffic Forecasting and Control: Metro Interstate Dataset Analysis

This project implements an end-to-end pipeline for analyzing and forecasting traffic volume using the Metro Interstate Traffic Volume dataset. It further explores a prototype Reinforcement Learning (RL) agent for traffic light control.

## Project Overview

The primary goals of this project are:
1.  To perform comprehensive Exploratory Data Analysis (EDA) to understand temporal patterns and correlations in traffic volume and exogenous weather variables.
2.  To develop and evaluate univariate forecasting models (LSTM, ARIMA) as baselines.
3.  To build and assess multivariate, multi-step forecasting models (LSTM) for a 24-hour horizon, incorporating weather features.
4.  To conduct error diagnostics and feature ablation studies to understand model performance and feature importance.
5.  To design and train a prototype Reinforcement Learning (DQN) agent for a simplified traffic light control scenario.

This project serves as a demonstration of skills in time-series analysis, deep learning for forecasting, and an introduction to reinforcement learning applications in transportation systems.

## Dataset

*   **Source:** Metro Interstate Traffic Volume Dataset (commonly found on the UCI Machine Learning Repository).
*   **Description:** Contains hourly interstate traffic volume for MN DoT ATR station 301, roughly midway between Minneapolis and St Paul, MN. It also includes weather and holiday features.
*   **File Used:** `Metro_Interstate_Traffic_Volume.csv`
*   **Preprocessing:** The notebook details steps for cleaning, resampling to hourly frequency (for the period 2016-01-01 to 2018-12-31), and imputation of missing values.

## Notebook Structure

The Jupyter Notebook (`Metro_Interstate_Traffic_dataset_analysis.ipynb`) is structured as follows:

1.  **Configuration & Imports:** Setup and library imports.
2.  **Data Loading & Cleaning:** Reading the raw data, handling duplicates, resampling, and imputation.
3.  **Exploratory Data Analysis (EDA):**
    *   STL Decomposition of Traffic Volume.
    *   Daily and Weekly Profiles of Traffic Volume.
    *   Correlation with Weather Features.
    *   Distribution Analysis of Traffic Volume.
    *   Long-term Temporal Patterns.
4.  **Univariate Forecasting Baselines:** One-step LSTM and ARIMA models.
5.  **Multi-Step Forecasting:** Direct 24-hour ahead LSTM forecasting using only traffic volume.
6.  **Error Diagnostics:** MAE heatmap by hour-of-day and forecast horizon for the multi-step model.
7.  **Multivariate Modeling & Ablation:**
    *   Multivariate direct 24-hour ahead LSTM forecasting (traffic, temp, rain).
    *   Feature ablation study to assess the impact of weather features.
8.  **RL Traffic-Light Control Prototype:**
    *   Custom Gymnasium environment for a 2-phase traffic light.
    *   DQN agent training using Stable-Baselines3.
    *   Evaluation via cumulative reward plot.

## Key Findings & Results

*   **EDA:** Strong daily and weekly seasonality in traffic volume was observed. Weak linear correlation with temperature and rainfall.
*   **Univariate Forecasting:** LSTM (MAE: 215.0) significantly outperformed ARIMA(2,1,2) (MAE: 2791.9) for one-step ahead prediction.
*   **Multi-Step Forecasting (Univariate):** Direct 24-hour LSTM showed increasing MAE with longer horizons, averaging 282.1.
*   **Multivariate Forecasting:** Incorporating temperature and rain improved the 24-hour direct LSTM forecast (overall MAE: 321.2 - *Note: This MAE is from a separate run in the notebook for the multivariate model before ablation. The ablation study trains shorter models for comparison.*).
*   **Feature Ablation:** Temperature was found to be more influential than rainfall for the multivariate LSTM model's 24-hour forecast accuracy.
*   **RL Prototype:** The DQN agent demonstrated learning by improving its cumulative reward (reducing queue lengths) in the simulated environment.

