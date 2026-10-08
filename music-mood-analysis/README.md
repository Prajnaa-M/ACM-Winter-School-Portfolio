# Predictive Spotify Mood Analytics



An interactive Streamlit application exploring the relationship between Spotify audio features and track valence using multiple linear regression.



## Overview



This project models track valence using three audio features:



- Tempo

- Energy

- Danceability



The application provides an interactive interface for generating valence predictions and exploring how each feature contributes to the model.



## Features



- Interactive tempo, energy, and danceability inputs

- Predicted valence score

- Mood classification

- Model coefficient visualization

- Live feature contribution analysis

- Regression diagnostics

- Interactive Plotly visualizations



## Model



The multiple linear regression model is:



`Valence = -0.09199 + (0.000481 × Tempo) + (0.1896 × Energy) + (0.6804 × Danceability)`



The model was fitted on 114,000 distinct Spotify tracks.



## Model Diagnostics



- **Multiple R-squared:** 0.2694

- **Adjusted R-squared:** 0.2694

- **Residual Standard Error:** 0.2216

- **F-statistic:** 1.401e4

- **p-value:** < 2.2e-16



The model leaves substantial unexplained variance, highlighting that emotional perception cannot be fully captured by these three audio features alone.



## Technology Stack



- Python

- Pandas

- Plotly

- Streamlit

- Multiple Linear Regression



## Run Locally



Install the dependencies:



```bash

pip install -r requirements.txt

