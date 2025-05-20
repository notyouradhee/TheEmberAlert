import streamlit as st
import pandas as pd
import datetime
import numpy as np
import folium
from streamlit_folium import st_folium


# ----------------------------
# 🔹 Load Data Based on Country
# ----------------------------
@st.cache_data
def load_data(country):
    # """
    # Load wildfire data CSV file for the selected country.
    # """
    if country == "Nepal":
        # ✅ Update this path if your file is elsewhere
        df = pd.read_csv("./data/processed/nepal_combined.csv")
        df["datetime"] = pd.to_datetime(
            df["date"] + " " + df["time"], format="%d-%b-%Y %I:%M:%S %p"
        )
    elif country == "South Korea":
        df = pd.read_csv("./data/processed/korea_combined.csv")
        df["datetime"] = pd.to_datetime(
            df["date"] + " " + df["time"], format="%d-%b-%Y %I:%M:%S %p"
        )
    else:
        st.error("Country not supported. Please select either Nepal or South Korea.")
        return pd.DataFrame()

    # # ✅ Use 'datetime' column from your CSV and convert to datetime type
    # df['acq_date'] = pd.to_datetime(df['datetime'])
    return df


def predict_fire_risk(latitude, longitude):
    # load the model
    import joblib
    model = joblib.load("./models/wildfire_model.pkl")
    scaler = joblib.load("./models/scaler.pkl")
    # Prepare the input data
    input_data = np.array([[latitude, longitude]])
    
    confidence = model.predict(input_data)
    return confidence[0]


def main():
    # Set the page title and layout
    st.set_page_config(
        page_title="🔥 Wildfire Detection System",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Initialize session state for the map

    # Set the title of the app
    st.title("🔥 Wildfire Detection System")

    # Load data for the selected country
    country = st.sidebar.selectbox("🌍 Select Country", ["Nepal", "South Korea"])
    df = load_data(country)

    # Get all unique dates from 'acq_date'
    unique_dates = sorted(df["datetime"].dt.date.unique(), reverse=True)

    # Dropdown to pick a date
    selected_date = st.sidebar.selectbox("📅 Select Date", unique_dates)

    # Filter data based on selected date
    filtered_data = df[df["datetime"].dt.date == selected_date]

    show_map = st.sidebar.toggle("Show Map")

    if show_map:
        st.markdown(f"### Wildfire Map for **{country}** on **{selected_date}**")
        st.markdown(
            "This map shows the locations of wildfires detected on the selected date."
        )

        if not filtered_data.empty:
            # Create map centered on the average wildfire location
            map = folium.Map(
                location=[
                    filtered_data["latitude"].mean(),
                    filtered_data["longitude"].mean(),
                ],
                zoom_start=6,
            )

            # Add each fire point to the map
            for _, row in filtered_data.iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=5,
                    popup=f"Confidence: {row['confidence']}",
                    color="red",
                    fill=True,
                    fill_opacity=0.7,
                ).add_to(map)

            # Show the map
            st_folium(map, width=700, height=500)
            # ----------------------------

        else:
            st.warning("No wildfire data available for this date.")




if __name__ == "__main__":
    main()
