import pandas as pd
import numpy as np
import joblib

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


train = pd.read_csv("train_df_1.csv")

train = train.dropna(subset=["Radiation"])


def create_features(df):
    df = df.copy()

    dt = pd.to_datetime(
        df["Data"] + " " + df["Time"],
        format="%d-%m-%Y %H:%M:%S"
    )

    sunrise = pd.to_datetime(
        df["Data"] + " " + df["TimeSunRise"],
        format="%d-%m-%Y %H:%M:%S"
    )

    sunset = pd.to_datetime(
        df["Data"] + " " + df["TimeSunSet"],
        format="%d-%m-%Y %H:%M:%S"
    )

    df["hour"] = dt.dt.hour + dt.dt.minute / 60
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear

    df["sunrise_min"] = sunrise.dt.hour * 60 + sunrise.dt.minute
    df["sunset_min"] = sunset.dt.hour * 60 + sunset.dt.minute

    df["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute

    df["day_length"] = df["sunset_min"] - df["sunrise_min"]

    df["solar_progress"] = (
        (df["minute_of_day"] - df["sunrise_min"])
        / df["day_length"]
    )

    df["solar_angle"] = np.sin(np.pi * df["solar_progress"])
    df["solar_angle"] = df["solar_angle"].clip(0, None)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["wind_sin"] = np.sin(
        np.deg2rad(df["WindDirection(Degrees)"])
    )

    df["wind_cos"] = np.cos(
        np.deg2rad(df["WindDirection(Degrees)"])
    )

    df["temp_humidity"] = (
        df["Temperature"] * df["Humidity"]
    )

    df["solar_temp"] = (
        df["solar_angle"] * df["Temperature"]
    )

    drop_cols = [
        "ID",
        "Data",
        "Time",
        "TimeSunRise",
        "TimeSunSet"
    ]

    df = df.drop(columns=drop_cols)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


X = create_features(
    train.drop("Radiation", axis=1)
)

y = train["Radiation"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


model.fit(X_train, y_train)


preds = model.predict(X_valid)

rmse = np.sqrt(
    mean_squared_error(y_valid, preds)
)

print(f"Validation RMSE: {rmse:.4f}")


joblib.dump(
    model,
    "solar_model.pkl",
    compress=3
)

print("Model saved successfully.")
