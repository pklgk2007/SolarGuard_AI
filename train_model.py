import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from lightgbm import LGBMRegressor

def create_features(df):
    df = df.copy()

    dt = pd.to_datetime(df["Data"] + " " + df["Time"], format="%d-%m-%Y %H:%M:%S")
    sunrise = pd.to_datetime(df["Data"] + " " + df["TimeSunRise"], format="%d-%m-%Y %H:%M:%S")
    sunset = pd.to_datetime(df["Data"] + " " + df["TimeSunSet"], format="%d-%m-%Y %H:%M:%S")

    df["hour"] = dt.dt.hour + dt.dt.minute / 60
    df["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear
    df["weekday"] = dt.dt.weekday

    df["sunrise_min"] = sunrise.dt.hour * 60 + sunrise.dt.minute
    df["sunset_min"] = sunset.dt.hour * 60 + sunset.dt.minute
    df["day_length"] = df["sunset_min"] - df["sunrise_min"]
    df["time_from_sunrise"] = df["minute_of_day"] - df["sunrise_min"]
    df["time_to_sunset"] = df["sunset_min"] - df["minute_of_day"]

    df["solar_progress"] = df["time_from_sunrise"] / df["day_length"]
    df["solar_angle"] = np.sin(np.pi * df["solar_progress"]).clip(0, None)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    df["wind_sin"] = np.sin(np.deg2rad(df["WindDirection(Degrees)"]))
    df["wind_cos"] = np.cos(np.deg2rad(df["WindDirection(Degrees)"]))

    df["temp_humidity"] = df["Temperature"] * df["Humidity"]
    df["temp_pressure"] = df["Temperature"] * df["Pressure"]
    df["solar_temp"] = df["solar_angle"] * df["Temperature"]
    df["solar_humidity"] = df["solar_angle"] * df["Humidity"]

    df = df.drop(["ID", "Data", "Time", "TimeSunRise", "TimeSunSet"], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

train = pd.read_csv("Data/train_df_1.csv")
train = train.dropna(subset=["Radiation"])

X = create_features(train.drop("Radiation", axis=1))
y = train["Radiation"]

lgb = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.2,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1
)

et = ExtraTreesRegressor(
    n_estimators=500,
    max_features=0.9,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model = VotingRegressor(
    estimators=[("lgb", lgb), ("et", et)],
    weights=[0.65, 0.35]
)

model.fit(X, y)
joblib.dump(model, "solar_model.pkl")

print("solar_model.pkl saved successfully.")
