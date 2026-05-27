import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from streamlit_js_eval import get_geolocation


st.set_page_config(
    page_title="SolarGuard AI",
    page_icon="☀️",
    layout="wide"
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 46px;
        font-weight: 800;
        color: #ffb703;
    }
    .subtitle {
        font-size: 20px;
        color: #d6d6d6;
    }
    </style>
    """,
    unsafe_allow_html=True
)


MODEL_PATH = "solar_model.pkl"


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


model = load_model()


def create_input_features(date, time, temp, humidity, pressure, wind_speed, wind_dir, sunrise, sunset):
    df = pd.DataFrame([{
        "Data": date.strftime("%d-%m-%Y"),
        "Time": time.strftime("%H:%M:%S"),
        "Temperature": temp,
        "Humidity": humidity,
        "Pressure": pressure,
        "Speed": wind_speed,
        "WindDirection(Degrees)": wind_dir,
        "TimeSunRise": sunrise.strftime("%H:%M:%S"),
        "TimeSunSet": sunset.strftime("%H:%M:%S")
    }])

    dt = pd.to_datetime(df["Data"] + " " + df["Time"], format="%d-%m-%Y %H:%M:%S")
    sr = pd.to_datetime(df["Data"] + " " + df["TimeSunRise"], format="%d-%m-%Y %H:%M:%S")
    ss = pd.to_datetime(df["Data"] + " " + df["TimeSunSet"], format="%d-%m-%Y %H:%M:%S")

    df["hour"] = dt.dt.hour + dt.dt.minute / 60
    df["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear
    df["weekday"] = dt.dt.weekday

    df["sunrise_min"] = sr.dt.hour * 60 + sr.dt.minute
    df["sunset_min"] = ss.dt.hour * 60 + ss.dt.minute

    df["day_length"] = df["sunset_min"] - df["sunrise_min"]
    df["time_from_sunrise"] = df["minute_of_day"] - df["sunrise_min"]
    df["time_to_sunset"] = df["sunset_min"] - df["minute_of_day"]

    df["solar_progress"] = df["time_from_sunrise"] / df["day_length"]
    df["solar_angle"] = np.sin(np.pi * df["solar_progress"])
    df["solar_angle"] = df["solar_angle"].clip(0, None)

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

    df = df.drop(["Data", "Time", "TimeSunRise", "TimeSunSet"], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def predict_radiation(features):
    if model is not None:
        try:
            prediction = float(model.predict(features)[0])
            return max(0, prediction)
        except Exception:
            pass

    solar_angle = features["solar_angle"].iloc[0]
    temp = features["Temperature"].iloc[0]
    humidity = features["Humidity"].iloc[0]

    return max(0, 900 * solar_angle + 2 * temp - 0.8 * humidity)


def risk_level(value):
    if value < 200:
        return "Low", "🟢 Safe radiation level", "#22c55e"
    elif value < 500:
        return "Moderate", "🟡 Use sunscreen if outside", "#eab308"
    elif value < 800:
        return "High", "🟠 Avoid long direct exposure", "#f97316"
    else:
        return "Extreme", "🔴 Stay indoors if possible", "#ef4444"


def get_live_weather():
    location = get_geolocation()

    if not location:
        return None

    lat = location["coords"]["latitude"]
    lon = location["coords"]["longitude"]

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,"
        "surface_pressure,wind_speed_10m,wind_direction_10m"
        "&daily=sunrise,sunset"
        "&timezone=auto"
    )

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    return response.json()


def create_day_forecast(date, temp, humidity, pressure, wind_speed, wind_dir, sunrise, sunset):
    sunrise_dt = datetime.combine(date, sunrise)
    sunset_dt = datetime.combine(date, sunset)

    times = pd.date_range(start=sunrise_dt, end=sunset_dt, freq="1h")

    radiation_values = []
    risk_values = []

    for t in times:
        features = create_input_features(
            date=date,
            time=t.time(),
            temp=temp,
            humidity=humidity,
            pressure=pressure,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            sunrise=sunrise,
            sunset=sunset
        )

        pred = predict_radiation(features)
        level, _, _ = risk_level(pred)

        radiation_values.append(pred)
        risk_values.append(level)

    return pd.DataFrame({
        "Time": times,
        "Predicted Radiation": radiation_values,
        "Risk Level": risk_values
    })


st.sidebar.title("☀️ SolarGuard AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Predict Solar Radiation",
        "Why Solar Radiation Matters",
        "Harmful Effects",
        "Safety Tips",
        "About Model"
    ]
)


if page == "Home":
    st.markdown('<div class="main-title">☀️ SolarGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Solar Radiation Prediction & Awareness Platform</div>', unsafe_allow_html=True)

    st.write("")
    st.write(
        """
        SolarGuard AI predicts solar radiation using weather and time-based data.
        It also helps users understand the risks of excessive sunlight exposure.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("ML Powered", "Prediction")
    col2.metric("Live Weather", "API Based")
    col3.metric("Health Focused", "Awareness")

    st.info("Go to the Prediction page to predict current and full-day solar radiation.")


elif page == "Predict Solar Radiation":
    st.title("📈 Predict Solar Radiation")

    if model is None:
        st.warning("Model file `solar_model.pkl` not found. The app is currently using a demo prediction formula.")

    mode = st.radio(
        "Choose Input Method",
        ["Manual Input", "Use My Live Location"],
        horizontal=True
    )

    if mode == "Manual Input":
        st.subheader("Manual Weather Input")

        col1, col2 = st.columns(2)

        with col1:
            date = st.date_input("Date")
            time = st.time_input("Time")
            temp = st.number_input("Temperature (°C)", value=25.0)
            humidity = st.number_input("Humidity (%)", value=50.0)
            pressure = st.number_input("Pressure (hPa)", value=1010.0)

        with col2:
            wind_speed = st.number_input("Wind Speed", value=5.0)
            wind_dir = st.number_input("Wind Direction (degrees)", value=180.0)
            sunrise = st.time_input("Sunrise Time", value=datetime.strptime("06:00:00", "%H:%M:%S").time())
            sunset = st.time_input("Sunset Time", value=datetime.strptime("18:00:00", "%H:%M:%S").time())

    else:
        st.info("Allow location access when the browser asks permission.")
        data = get_live_weather()

        if data is None:
            st.warning("Location not available yet. Please allow location permission.")
            st.stop()

        current = data["current"]
        daily = data["daily"]

        date = datetime.now().date()
        time = datetime.now().time()

        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        pressure = current["surface_pressure"]
        wind_speed = current["wind_speed_10m"]
        wind_dir = current["wind_direction_10m"]

        sunrise = pd.to_datetime(daily["sunrise"][0]).time()
        sunset = pd.to_datetime(daily["sunset"][0]).time()

        st.success("Live weather data imported successfully!")

        st.subheader("Live Weather Data")

        c1, c2, c3 = st.columns(3)
        c1.metric("Temperature", f"{temp} °C")
        c2.metric("Humidity", f"{humidity}%")
        c3.metric("Pressure", f"{pressure} hPa")

        c4, c5, c6 = st.columns(3)
        c4.metric("Wind Speed", f"{wind_speed}")
        c5.metric("Wind Direction", f"{wind_dir}°")
        c6.metric("Sunrise / Sunset", f"{sunrise} / {sunset}")

    st.divider()

    if st.button("Predict Radiation", type="primary"):
        features = create_input_features(
            date=date,
            time=time,
            temp=temp,
            humidity=humidity,
            pressure=pressure,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            sunrise=sunrise,
            sunset=sunset
        )

        prediction = predict_radiation(features)
        level, message, color = risk_level(prediction)

        graph_df = create_day_forecast(
            date=date,
            temp=temp,
            humidity=humidity,
            pressure=pressure,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            sunrise=sunrise,
            sunset=sunset
        )

        peak_idx = graph_df["Predicted Radiation"].idxmax()
        peak_time = graph_df.loc[peak_idx, "Time"]
        peak_value = graph_df.loc[peak_idx, "Predicted Radiation"]
        avg_radiation = graph_df["Predicted Radiation"].mean()

        safe_hours = graph_df[graph_df["Predicted Radiation"] < 500]

        st.subheader("Current Prediction")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Radiation", f"{prediction:.2f} W/m²")
        m2.metric("Risk Level", level)
        m3.metric("Peak Time", peak_time.strftime("%I:%M %p"))
        m4.metric("Peak Radiation", f"{peak_value:.2f} W/m²")

        st.write(message)

        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Radiation Trend",
                "🌡️ Gauge & Risk",
                "📊 Hourly Analysis",
                "🛡️ Safety Insights"
            ]
        )

        with tab1:
            st.subheader("Whole-Day Solar Radiation Forecast")

            fig_line = px.line(
                graph_df,
                x="Time",
                y="Predicted Radiation",
                markers=True,
                title="Predicted Solar Radiation Throughout the Day"
            )

            fig_line.add_vline(
                x=peak_time.to_pydatetime(),
                line_dash="dash",
                line_color="red"
            )

            fig_line.add_annotation(
                x=peak_time.to_pydatetime(),
                y=peak_value,
                text="Peak",
                showarrow=True,
                arrowhead=2
            )

            st.plotly_chart(fig_line, use_container_width=True)

            st.info(f"Average predicted radiation for the day: {avg_radiation:.2f} W/m²")

        with tab2:
            col_gauge, col_pie = st.columns(2)

            with col_gauge:
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prediction,
                        title={"text": "Current Radiation Level"},
                        gauge={
                            "axis": {"range": [0, 1000]},
                            "bar": {"color": color},
                            "steps": [
                                {"range": [0, 200], "color": "#14532d"},
                                {"range": [200, 500], "color": "#713f12"},
                                {"range": [500, 800], "color": "#7c2d12"},
                                {"range": [800, 1000], "color": "#7f1d1d"}
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 4},
                                "thickness": 0.75,
                                "value": prediction
                            }
                        }
                    )
                )

                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_pie:
                risk_counts = graph_df["Risk Level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Hours"]

                fig_pie = px.pie(
                    risk_counts,
                    names="Risk Level",
                    values="Hours",
                    title="Risk Level Distribution During the Day"
                )

                st.plotly_chart(fig_pie, use_container_width=True)

        with tab3:
            st.subheader("Hourly Radiation Bar Chart")

            fig_bar = px.bar(
                graph_df,
                x="Time",
                y="Predicted Radiation",
                color="Risk Level",
                title="Hourly Radiation Risk Levels"
            )

            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("Radiation Data Table")
            st.dataframe(graph_df, use_container_width=True)

        with tab4:
            st.subheader("Safety Recommendation")

            st.success("🌅 Recommended outdoor time: 06:00 AM to 09:00 AM")
            st.success("🌇 Recommended outdoor time: After 4:30 PM")
            st.warning("Avoid direct sunlight between 11:00 AM and 3:00 PM.")

            if prediction >= 800:
                st.error("Extreme radiation. Avoid outdoor exposure if possible.")
            elif prediction >= 500:
                st.warning("High radiation. Use sunscreen, sunglasses, and avoid long exposure.")
            elif prediction >= 200:
                st.info("Moderate radiation. Basic sun protection is recommended.")
            else:
                st.success("Low radiation. Outdoor conditions are relatively safer.")

            st.markdown(
                """
                ### Recommended Protection
                - Use sunscreen before going outside
                - Wear UV-protection sunglasses
                - Stay hydrated
                - Avoid long exposure near noon
                - Wear light but covered clothing
                """
            )


elif page == "Why Solar Radiation Matters":
    st.title("🌍 Why Solar Radiation Matters")

    st.write(
        """
        Solar radiation affects solar power generation, agriculture,
        weather patterns, climate monitoring, and human health.
        """
    )

    st.markdown(
        """
        ### Key Uses
        - Solar power planning
        - Weather analysis
        - Crop and irrigation decisions
        - Outdoor safety alerts
        - Climate awareness
        """
    )


elif page == "Harmful Effects":
    st.title("⚠️ Harmful Effects of Excess Solar Radiation")

    st.markdown(
        """
        Excessive exposure can cause:

        - Sunburn
        - Skin aging
        - Eye damage
        - Cataracts
        - Heat exhaustion
        - Increased skin cancer risk
        - Immune system stress
        """
    )


elif page == "Safety Tips":
    st.title("🛡️ Safety Tips")

    st.markdown(
        """
        - Avoid direct sunlight from 11 AM to 3 PM
        - Wear sunscreen
        - Use UV-protection sunglasses
        - Stay hydrated
        - Wear protective clothing
        - Check radiation or UV levels before going outside
        """
    )


elif page == "About Model":
    st.title("🧠 About the ML Model")

    st.write(
        """
        This project predicts solar radiation using weather and time-based features.
        It supports manual input and live location-based weather import.
        """
    )

    st.markdown(
        """
        ### Features Used
        - Temperature
        - Humidity
        - Pressure
        - Wind speed
        - Wind direction
        - Date and time
        - Sunrise and sunset
        - Solar angle features

        
    )
