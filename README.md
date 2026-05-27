# SolarGuard AI

SolarGuard AI is a solar radiation prediction and awareness web app.

## Features

- Manual input prediction
- Live location based weather import
- Whole-day solar radiation graph
- Risk level classification
- Awareness pages about harmful effects and safety tips
- Streamlit web interface

## Project Files

- `app.py` - main Streamlit app
- `train_model.py` - trains and saves `solar_model.pkl`
- `requirements.txt` - required libraries

## How to Run

Install requirements:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python train_model.py
```

Run app:

```bash
streamlit run app.py
```

## Important

Place `train_df_1.csv` in the project folder before running `train_model.py`.

If `solar_model.pkl` is missing, the app will still run using a demo formula, but real ML prediction needs the trained model.
