import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import datetime
from branca.element import Template, MacroElement

# ----------------------------
# 🔹 Load Data Based on Country
# ----------------------------
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
    legend_html = """
     {% macro html(this, kwargs) %}
     <div style='
         position: fixed;
         bottom: 50px;
         left: 50px;
         width: 150px;
         height: 110px;
         background-color: black;
         border:2px solid grey;
         z-index:9999;
         font-size:14px;
         padding: 10px;
         '>
         <b>🔥 Fire Severity</b><br>
         <i style='background:red;width:10px;height:10px;display:inline-block'></i> High (≥ 70)<br>
         <i style='background:orange;width:10px;height:10px;display:inline-block'></i> Medium (40–69)<br>
         <i style='background:yellow;width:10px;height:10px;display:inline-block'></i> Low (< 40)
     </div>
     {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(legend_html)
    map_object.get_root().add_child(macro)


# ----------------------------
# 🔹 Predict Wildfire Risk (Mock)
# ----------------------------
def predict_fire_risk(latitude, longitude):
    import joblib
    model = joblib.load("./models/wildfire_model.pkl")
    scaler = joblib.load("./models/scaler.pkl")
    input_data = np.array([[latitude, longitude]])
    confidence = model.predict(input_data)
    return confidence[0]

# ----------------------------
# 🔹 Streamlit App
# ----------------------------
def main():
    st.set_page_config(
        page_title="🔥 Wildfire Detection System",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔥 Wildfire Detection System")

    country = st.sidebar.selectbox("🌍 Select Country", ["Nepal", "South Korea"])
    df = load_data(country)

    if df.empty:
        st.warning("No data loaded.")
        return

    min_date = df["datetime"].dt.date.min()
    max_date = df["datetime"].dt.date.max()

    selected_date = st.sidebar.date_input(
        "📅 Select Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    available_dates = set(df["datetime"].dt.date.unique())

    if selected_date not in available_dates:
        st.sidebar.warning(
            f"No wildfire data available for {selected_date}. Please select a different date."
        )
        filtered_data = pd.DataFrame()
    else:
        filtered_data = df[df["datetime"].dt.date == selected_date]

    show_map = st.sidebar.toggle("Show Map")

    if show_map:
        st.markdown(f"### Wildfire Map for **{country}** on **{selected_date}**")
        st.markdown(
            "This map shows the locations of wildfires detected on the selected date."
        )

        if not filtered_data.empty:
            m = folium.Map(
                location=[
                    filtered_data["latitude"].mean(),
                    filtered_data["longitude"].mean(),
                ],
                zoom_start=6,
            )

            for _, row in filtered_data.iterrows():
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
                ).add_to(m)

            # 🔸 Add floating legend to map
            add_legend_to_map(m)

            # 🔸 Add legend description above map
            st.markdown("""
            #### 🔥 Fire Severity Legend
            - 🟥 **Red**: High severity (Confidence ≥ 70)
            - 🟧 **Orange**: Medium severity (40 ≤ Confidence < 70)
            - 🟨 **Yellow**: Low severity (Confidence < 40)
            """)

            st_folium(m, width=700, height=500)
        else:
            st.warning("No wildfire data available for this date.")

if __name__ == "__main__":
    main()
