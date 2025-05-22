import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import requests
from branca.element import Template, MacroElement



# load model artifacts
@st.cache_resource
def load_model_artifacts():
    import joblib

    model = joblib.load("./models/wildfire_predictor_model.pkl")
    scaler = joblib.load("./models/wildfire_predictor_scaler.pkl")
    return model, scaler


# load data based on the selected country
@st.cache_data
def load_data(country):
    if country == "Nepal":
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
    return df


# ----------------------------
# 🔹 Add a Legend to the Map
# ----------------------------
def add_legend_to_map(map_object):

    macro = MacroElement()
    # macro._template = Template(legend_html)
    map_object.get_root().add_child(macro)


# Geocode Latitude and Longitude to Location Name
@st.cache_data(show_spinner=False)
def geocode_lat_long_to_name(lat: float, lon: float, confidence: float) -> tuple:
    language = "en"
    confidence_dict = {
        "red": "🟥",
        "orange": "🟧",
        "yellow": "🟨",
    }
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&accept-language={language}"
    headers = {"User-Agent": "YourAppName/1.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        name = data.get("display_name")
        color = (
            confidence_dict.get("red")
            if confidence >= 70
            else (
                confidence_dict.get("orange")
                if 40 <= confidence < 70
                else confidence_dict.get("yellow")
            )
        )
        return (name, confidence, color)
    return (None, confidence, "❓")


@st.cache_data(show_spinner="Loading location data...")
def build_locations_df(df):
    lat_lon_list = [
        (lat, lon, conf)
        for lat, lon, conf in zip(df["latitude"], df["longitude"], df["confidence"])
    ]
    locations = [
        geocode_lat_long_to_name(lat, lon, conf) for lat, lon, conf in lat_lon_list
    ]
    return pd.DataFrame(locations, columns=["Location", "Confidence", "Severity"])


def show_overview(df, country):
    st.subheader(f"📏 Dataset Information of: *{country}*")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **Total records:** {df.shape[0]}")
        st.markdown(f"- **Missing values:** {df.isnull().sum().sum()}")
        st.markdown(f"- **Total features:** {df.shape[1]}")
        st.markdown(f"- **Features:** {df.columns.tolist()}")
        st.markdown(
            f"- **Categorical features:** {df.select_dtypes(include='object').columns.tolist()}"
        )
    with col2:
        st.markdown("#### Source of Data")
        st.markdown("- **Source:** [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)")
        st.markdown("- **Data Type:** CSV")
        st.markdown("- **Last Updated:** 2023-10-01")


def plot_the_map(df, date):
    st.subheader(f"🌍 Wildfire Map on {date}")

    # Legend
    st.markdown("#### 🚨 Locations Affected")
    st.markdown("🟥: High (≥ 70), 🟧: Medium (40–69), 🟨: Low (< 40)")

    # Build cached DataFrame
    locations_df = build_locations_df(df)
    st.markdown(f"Total locations affected: {locations_df.shape[0]}")
    # Display as styled HTML table
    severity_map = {"🟥": "Red", "🟧": "Orange", "🟨": "Yellow", "❓": "Unknown"}
    locations_df["Severity Label"] = locations_df["Severity"].map(severity_map)
    locations_df["Severity"] = (
        locations_df["Severity"] + " " + locations_df["Severity Label"]
    )
    st.dataframe(
        locations_df[["Location", "Confidence", "Severity"]], use_container_width=True
    )

    # 🔸 Create a map with folium
    if not df.empty:
        fol_map = folium.Map(
            location=[
                df["latitude"].mean(),
                df["longitude"].mean(),
            ],
            zoom_start=8,
        )

        for _, row in df.iterrows():
            confidence = row["confidence"]
            if confidence >= 70:
                color = "red"
            elif confidence >= 40:
                color = "orange"
            else:
                color = "yellow"

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                popup=f"Confidence: {confidence}",
                color=color,
                fill=True,
                fill_opacity=0.7,
            ).add_to(fol_map)

        # 🔸 Add floating legend to map
        add_legend_to_map(fol_map)
        st_folium(
            fol_map,
            height=400,  # Fixed height matches CSS
            use_container_width=True,
            key="fixed_map",
            returned_objects=[]  # Prevents unused return values
        )
    else:
        st.warning("No wildfire data available for this date.")
    st.markdown("---")


# day vs night proportion analysis
def day_night_analysis(df):
    st.subheader("🌞 Day vs Night Proportion Analysis")
    df["hour"] = df["datetime"].dt.hour
    day_count = df[df["hour"].between(6, 18)].shape[0]
    night_count = (
        df[df["hour"].between(19, 23)].shape[0] + df[df["hour"].between(0, 5)].shape[0]
    )
    total_count = day_count + night_count
    # plot the pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Day", "Night"],
                values=[day_count, night_count],
                hole=0.3,
                textinfo="label+percent",
                textfont_size=20,
                marker=dict(colors=["#FF9999", "#66B3FF"]),
            )
        ]
    )
    fig.update_layout(
        title_text="Day vs Night Proportion of Wildfire Incidents",
        title_x=0.5,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")


def plot_wildfire_by_lat_long(df):
    st.subheader("📈 Wildfire Incidents by Location")
    location_names = [
        geocode_lat_long_to_name(lat, lon, conf)[0]
        for lat, lon, conf in zip(df["latitude"], df["longitude"], df["confidence"])
    ]
    names = [
        f"{address.split(',')[0].strip()}, {address.split(',')[1].strip()}"
        for address in location_names
    ]
    df.loc[:, "location_name"] = names

    # bar chart
    fig = px.bar(
        df,
        x="location_name",
        y="confidence",
        color="confidence",
        title="Wildfire Incidents by Location",
        color_continuous_scale=px.colors.sequential.Plasma,
        labels={"location_name": "Location Name", "confidence": "Confidence"},
    )
    fig.update_layout(
        xaxis_title="Location Name",
        yaxis_title="Confidence",
        title_x=0.5,
        xaxis_tickangle=-45,
        xaxis=dict(tickmode="linear"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")


def show_yearly_trend(df):
    st.subheader("📈 Yearly Trend of Wildfire Incidents")
    df["year"] = df["datetime"].dt.year
    yearly_trend = df.groupby("year").size().reset_index(name="count")
    fig = px.line(
        yearly_trend,
        x="year",
        y="count",
        title="Yearly Trend of Wildfire Incidents",
        labels={"year": "Year", "count": "Count"},
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Count",
        title_x=0.5,
    )
    st.plotly_chart(fig, use_container_width=True)


def display_model_info(model):
    st.markdown("### 📊 Model Information")
    feature_names = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'bright_t31', 'frp', 'daynight']

    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Optional: Add model score (if test set score is known)
    model_info_df = pd.DataFrame({
        'Metric': ['Model Type', 'n_estimators', 'Max Depth'], #, 'Model Score (train)'],
        'Value': ['RandomForestRegressor', model.n_estimators, model.max_depth] #, model.score(X_train, y_train)]
    })
    model_info_df['Value'] = model_info_df['Value'].astype(str)
    # Display in Streamlit
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Feature Importances")
        st.dataframe(importance_df)
    with col2:
        st.markdown("### Model Summary")
        st.dataframe(model_info_df)

# model prediction
def predict_fire_risk(model, scaler, input_data_df):
    scaled_data = scaler.transform(input_data_df)
    prediction = model.predict(scaled_data)
    if isinstance(prediction, (np.ndarray, list)) and len(prediction) == 1:
        prediction_value = prediction[0]
    else:
        prediction_value = prediction  

    st.success(f"Predicted Confidence: **{prediction_value:.2f}**")
    # Optional: Add severity emoji
    if prediction >= 70:
        severity = "🟥 High"
    elif prediction >= 40:
        severity = "🟧 Medium"
    else:
        severity = "🟨 Low"
    location_name = geocode_lat_long_to_name(
        input_data_df["latitude"][0],
        input_data_df["longitude"][0],
        prediction_value,
    )[0]
    location_name = f"{location_name.split(',')[0].strip()}, {location_name.split(',')[1].strip()}"
    st.markdown(f"**Predicted 🔥 Severity:** {severity} for **Location:** {location_name}")


def get_user_input(df):
    st.subheader("Input Data for Prediction")
    daynight_dict = {"D": 1, "N": 0}
    with st.sidebar.form("filter_form"):
        with st.container():
            # Latitude range slider
            lat = st.slider(
                "Latitude",
                min_value=float(df["latitude"].min()),
                max_value=float(df["latitude"].max()),
                value=float(df["latitude"].mean()),
            )
            lon = st.slider(
                "Longitude",
                float(df["longitude"].min()),
                float(df["longitude"].max()),
                float(df["longitude"].mean()),
            )
            brightness = st.slider(
                "Brightness",
                float(df["brightness"].min()),
                float(df["brightness"].max()),
                float(df["brightness"].mean()),
            )
            scan = st.slider(
                "Scan",
                float(df["scan"].min()),
                float(df["scan"].max()),
                float(df["scan"].mean()),
            )
            track = st.slider(
                "Track",
                float(df["track"].min()),
                float(df["track"].max()),
                float(df["track"].mean()),
            )
            bright_t31 = st.slider(
                "Brightness T31",
                float(df["bright_t31"].min()),
                float(df["bright_t31"].max()),
                float(df["bright_t31"].mean()),
            )
            frp = st.slider(
                "FRP",
                float(df["frp"].min()),
                float(df["frp"].max()),
                float(df["frp"].mean()),
            )
            daynight = st.selectbox("Day/Night", options=["D", "N"], index=0)

            user_input_df = pd.DataFrame(
                {
                    "latitude": [lat],
                    "longitude": [lon],
                    "brightness": [brightness],
                    "scan": [scan],
                    "track": [track],
                    "bright_t31": [bright_t31],
                    "frp": [frp],
                    "daynight": [daynight_dict[daynight]],
                }
            )
            # Submit button
            submitted = st.form_submit_button("Submit")

            if submitted:
                return user_input_df
    return None


# main function
def main():
    st.set_page_config(
        page_title="Wildfire Visualization",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # to control the unusual behavior of the map
    st.markdown("""
        <style>
            /* Nuclear option for removing all padding */
            .stApp > div > div > div > div > section {
                padding-top: 0rem !important;
                padding-bottom: 0rem !important;
            }
            
            /* Target specific map container */
            .folium-map {
                margin: 0 !important;
                padding: 0 !important;
                border: none !important;
                height: 400px !important;
            }
            
            /* Fix Streamlit's default column gaps */
            [data-testid="column"] {
                gap: 0rem !important;
            }
            
            /* Remove empty space above headers */
            .st-emotion-cache-10trblm {
                padding-top: 0rem !important;
            }
        </style>
        """, unsafe_allow_html=True)

    st.title("🔥 Wildfire Visualization and Analysis")
    st.markdown("---")

    country = st.sidebar.selectbox("🌍 Select Country", ["Nepal", "South Korea"])
    df = load_data(country)
    if df.empty:
        st.warning("No data loaded.")
        return
    show_overview(df, country)
    st.markdown("---")
    show_yearly_trend(df)
    st.markdown("---")

    available_dates = set(df["datetime"].dt.date.unique())

    selected_date = st.sidebar.selectbox(
        "📅 Select a Date",
        options=available_dates,
    )
    # filtering date based on the selected date
    if selected_date not in available_dates:
        st.sidebar.warning(
            f"No wildfire data available for {selected_date}. Please select a different date."
        )
        filtered_data = pd.DataFrame()
    else:
        filtered_data = df[df["datetime"].dt.date == selected_date]

    # showing map for the filtered data
    show_map = st.sidebar.toggle("Show Map")
    if show_map:
        plot_the_map(filtered_data, selected_date)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Statistical Analysis")
    st.sidebar.checkbox("Show Statistical Analysis", value=False, key="stat_analysis")
    if st.session_state.stat_analysis:
        st.subheader(f"📊 Statistical Analysis for {selected_date}")
        col1, col2 = st.columns(2)
        with col1:
            day_night_analysis(filtered_data)
        with col2:
            plot_wildfire_by_lat_long(filtered_data)

    st.sidebar.markdown("---")
    model_predict = st.sidebar.checkbox("Predict Fire Risk", value=False, key="model_predict")
    if not model_predict:
        st.sidebar.warning("Please check the box to enable prediction.")
        return
    
    st.sidebar.subheader("🔥 Predict Fire Risk")
    user_input_df = get_user_input(df)
    st.markdown("Fire risk prediction based on latitude and longitude.")
    st.markdown(
        "**Note:** The model is trained on historical data and may not reflect real-time conditions."
    )
    model, scaler = load_model_artifacts()
    display_model_info(model)
    st.dataframe(user_input_df, use_container_width=True)
    if user_input_df is not None:
        predict_fire_risk(model, scaler, user_input_df)


if __name__ == "__main__":
    main()
