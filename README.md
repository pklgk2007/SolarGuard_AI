# SolarGuard AI ☀️

SolarGuard AI is an AI-powered web application that predicts solar radiation using Machine Learning and real-time weather data.

This project was developed for the MLX Session Zero Hackathon and later extended into a complete interactive web application with live weather integration, radiation forecasting, safety recommendations, and awareness features.

---

## Features

- Solar radiation prediction using Machine Learning
- Real-time weather integration using live location
- Whole-day solar radiation forecasting
- Interactive dashboard and visualizations
- Radiation risk analysis
- Safety recommendations
- Awareness pages about harmful solar exposure

---

## Tech Stack

- Python
- Streamlit
- LightGBM
- Scikit-learn
- Plotly
- Pandas
- NumPy
- Open-Meteo API

---

## Live Demo

https://solarguardai-xynyneafqcrspewpw8crfr.streamlit.app/

---

## Project Files

- `app.py` - main Streamlit web application
- `train_model.py` - model training script
- `requirements.txt` - required libraries
- `solar_model.pkl` - trained ML model

---

## How to Run

### Install Requirements

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
