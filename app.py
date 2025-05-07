import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

# ----------------------------
# 🔹 Load Data Based on Country
# ----------------------------
@st.cache_data
def load_data(country):
    """
    Load wildfire data CSV file for the selected country.
    """
    if country == 'Nepal':
        # ✅ Update this path if your file is elsewhere
        df = pd.read_csv(r"D:\EmberAlert_wildfire_Prediction\nepal_combined.csv")
    elif country == 'South Korea':
        df = pd.read_csv(r"D:\EmberAlert_wildfire_Prediction\korea_combined.csv")
    else:
        df = pd.DataFrame()

    # ✅ Use 'datetime' column from your CSV and convert to datetime type
    df['acq_date'] = pd.to_datetime(df['datetime'])

    return df

# ----------------------------
# 🔹 Sidebar UI
# ----------------------------
st.sidebar.title("🔥 Wildfire Dashboard Settings")

# Dropdown to select the country
available_countries = ['Nepal', 'South Korea']
country = st.sidebar.selectbox("🌍 Select Country", available_countries)

# Load corresponding data
df = load_data(country)

# ----------------------------
# 🔹 Date Selector
# ----------------------------
# Get all unique dates from 'acq_date'
unique_dates = sorted(df['acq_date'].dt.date.unique(), reverse=True)

# Dropdown to pick a date
selected_date = st.sidebar.selectbox("📅 Select Date", unique_dates)

# Filter data based on selected date
filtered_data = df[df['acq_date'].dt.date == selected_date]

# ----------------------------
# 🔹 Main Display
# ----------------------------
st.title(f"🔥 Wildfire Detection System: {country}")
st.markdown(f"### Showing wildfire data for **{country}** on **{selected_date}**")
st.markdown(f"Total fire points detected: **{len(filtered_data)}**")

# ----------------------------
# 🔹 Map Visualization
# ----------------------------
if not filtered_data.empty:
    # Create map centered on the average wildfire location
    m = folium.Map(
        location=[filtered_data['latitude'].mean(), filtered_data['longitude'].mean()],
        zoom_start=6
    )

    # Add each fire point to the map
    for _, row in filtered_data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=f"Confidence: {row['confidence']}",
            color='red',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    # Show the map
    st_folium(m, width=700, height=500)
else:
    st.warning("No wildfire data available for this date.")
