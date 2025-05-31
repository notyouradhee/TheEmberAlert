import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit_folium import st_folium
from st_aggrid import AgGrid, GridOptionsBuilder
import logging
import smtplib
from email.message import EmailMessage
import io
from fpdf import FPDF
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import tempfile
import os

# =========== SUPABASE (REPLACE MYSQL) ===========
from supabase import create_client, Client

SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========== LOGGING ===========
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('wildfire_dashboard.log')
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# =========== CONSTANTS ===========
allowed_domains = ["gmail.com", "naver.com", "outlook.com", "yahoo.com", "hotmail.com", "protonmail.com"]
severity_options = ["Low", "Medium", "High"]

# =========== EMAIL ===========
def send_fire_alert_email(receiver_email, location, confidence, severity):
    sender_email = "ldawa9808@gmail.com"
    sender_password = "zsdrgookrvtsrrga"
    subject = f"🔥 Wildfire Alert! [{severity}]"
    body = f"""
    ALERT! 🔥🔥

    A possible wildfire has been detected.

    Location: {location}
    Confidence Level: {confidence}%
    Severity: {severity}

    Stay safe and take necessary precautions.

    -- Wildfire Detection System
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
    except Exception as e:
        logger.warning(f"Failed to send email to {receiver_email}: {e}")

def send_subscription_confirmation_email(receiver_email, lat, lon, severity_levels):
    sender_email = "ldawa9808@gmail.com"
    sender_password = "zsdrgookrvtsrrga"
    subject = "✅ Wildfire Alert Subscription Successful"
    levels = ", ".join(severity_levels)
    body = f"""
    Hello,

    You have successfully subscribed to Wildfire Alerts!

    - Location: ({lat}, {lon})
    - Severity Level(s): {levels}

    You'll receive wildfire alerts for your chosen area and severity level.

    Thank you for using EmberAlert!

    -- EmberAlert Team
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
    except Exception as e:
        logger.warning(f"Failed to send confirmation email to {receiver_email}: {e}")

def is_valid_email(email):
    import re
    pattern = r"^[a-zA-Z0-9._%+-]+@(" + "|".join(re.escape(domain) for domain in allowed_domains) + ")$"
    return re.match(pattern, email)

# =========== GEO DISTANCE ===========
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c

# =========== SEVERITY FROM CONFIDENCE ===========
def get_severity(conf):
    if conf >= 70:
        return "High"
    elif conf >= 40:
        return "Medium"
    else:
        return "Low"

# =========== ALERT LOGIC (SUPABASE) ===========
def notify_matching_subscribers(fire_lat, fire_lon, fire_confidence, fire_severity, location_str):
    try:
        query = supabase.table("subscription").select("*").eq("severity_level", fire_severity)
        all_subs = query.execute().data
        notified = 0
        for sub in all_subs:
            sub_lat, sub_lon = float(sub['latitude']), float(sub['longitude'])
            distance = haversine(fire_lat, fire_lon, sub_lat, sub_lon)
            if distance <= 50:
                send_fire_alert_email(sub['email'], location_str, fire_confidence, fire_severity)
                notified += 1
        if notified:
            st.success(f"🔥 Alert sent to {notified} user(s) subscribed to {fire_severity} within 50 km!")
        else:
            st.info("No subscribers found for this event (within 50 km and correct severity).")
    except Exception as e:
        logger.error(f"Error in notify_matching_subscribers: {e}")
        st.error("Error sending alerts.")

# =========== DATA HELPERS ===========
@st.cache_data
def load_data(country):
    if country == 'Nepal':
        base_dir = os.path.dirname(__file__)
        csv_path = os.path.join(base_dir, "data", "processed", "nepal_combined.csv")
        df = pd.read_csv(csv_path)
    elif country == 'South Korea':
        base_dir = os.path.dirname(__file__)
        csv_path = os.path.join(base_dir, "data", "processed", "korea_combined.csv")
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()
    if not df.empty:
        df['acq_date'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    return df

@st.cache_resource
def load_model_artifacts():
    import joblib
    model = joblib.load("./models/wildfire_predictor_model.pkl")
    scaler = joblib.load("./models/wildfire_predictor_scaler.pkl")
    return model, scaler

@st.cache_data(show_spinner=False)
def geocode_lat_long_to_name(lat: float, lon: float, confidence: float) -> str:
    language = "en"
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&accept-language={language}"
    headers = {"User-Agent": "WildfireApp/1.0"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            name = data.get("display_name")
            return name
    except Exception:
        pass
    return f"Lat: {lat}, Lon: {lon}"

# ========== PDF & SUMMARY ==========
def summarize_fire_data(df):
    if df.empty:
        return "No wildfire data for the selected filters."
    summary = []
    summary.append(f"Total wildfire incidents: {len(df)}")
    avg_conf = df['confidence'].mean()
    summary.append(f"Average fire confidence: {avg_conf:.2f}")
    high = len(df[df['confidence'] >= 70])
    medium = len(df[(df['confidence'] >= 40) & (df['confidence'] < 70)])
    low = len(df[df['confidence'] < 40])
    summary.append(f"High severity (confidence >= 70): {high}")
    summary.append(f"Medium severity (40 <= confidence < 70): {medium}")
    summary.append(f"Low severity (confidence < 40): {low}")
    return "\n".join(summary)

def generate_pdf_report(filtered_data, country, selected_date):
    summary = summarize_fire_data(filtered_data)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Wildfire Risk Report - {country} - {selected_date}", ln=1, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for line in summary.split('\n'):
        pdf.cell(0, 10, line, ln=1)
    pdf.ln(5)
    if not filtered_data.empty:
        pdf.set_font("Arial", 'B', 11)
        columns = ["latitude", "longitude", "confidence"]
        for col in columns:
            pdf.cell(40, 8, col, border=1)
        pdf.ln()
        pdf.set_font("Arial", size=10)
        for idx, row in filtered_data.iterrows():
            for col in columns:
                pdf.cell(40, 8, str(row[col]), border=1)
            pdf.ln()
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# ======= PDF VISUALIZATION REPORT: Well-Labeled Bar Chart =======
def generate_viz_pdf_report(filtered_viz, region, date_range):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Fire Risk Visualization Report", ln=1, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    region_lat, region_lon = region
    pdf.cell(0, 8, f"Region: Lat {region_lat[0]:.2f}-{region_lat[1]:.2f}, Lon {region_lon[0]:.2f}-{region_lon[1]:.2f}", ln=1)
    pdf.cell(0, 8, f"Period: {date_range[0]} to {date_range[1]}", ln=1)
    pdf.ln(5)
    summary = summarize_fire_data(filtered_viz)
    for line in summary.split('\n'):
        pdf.cell(0, 8, line, ln=1)
    pdf.ln(4)
    if not filtered_viz.empty:
        filtered_viz['severity'] = filtered_viz['confidence'].apply(get_severity)
        daily_counts = filtered_viz.groupby([filtered_viz['acq_date'].dt.date, 'severity']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 3))
        daily_counts.plot(kind='bar', stacked=True, ax=ax, color={'High': 'red', 'Medium': 'orange', 'Low': 'yellow'})
        ax.set_title('Wildfire Incidents per Day by Severity')
        ax.set_xlabel('Date')
        ax.set_ylabel('Incidents')
        ax.legend(title="Severity")
        ax.set_xticks(range(len(daily_counts.index)))
        ax.set_xticklabels([d.strftime('%b %d, %Y') for d in daily_counts.index], rotation=45, ha='right', fontsize=7)
        plt.tight_layout()
        img_temp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(img_temp.name, format='png', bbox_inches='tight')
        plt.close(fig)
        pdf.image(img_temp.name, x=10, y=pdf.get_y(), w=190)
        img_temp.close()
        os.unlink(img_temp.name)
        pdf.ln(65)
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# ========== VISUALIZATION FUNCTIONS ==========
def show_overview(df, country):
    st.subheader(f"Dataset Information: {country}")
    st.markdown(f"- **Total records:** {df.shape[0]}")
    st.markdown(f"- **Missing values:** {df.isnull().sum().sum()}")
    st.markdown(f"- **Total features:** {df.shape[1]}")
    st.markdown(f"- **Features:** {df.columns.tolist()}")
    st.markdown("- **Source of Data:** NASA FIRMS")

def show_yearly_trend(df):
    st.subheader("Yearly Trend of Wildfire Incidents")
    df["year"] = df["acq_date"].dt.year
    yearly_trend = df.groupby("year").size().reset_index(name="count")
    fig = px.line(
        yearly_trend,
        x="year",
        y="count",
        title="Yearly Trend of Wildfire Incidents",
        labels={"year": "Year", "count": "Count"},
    )
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=yearly_trend["year"].tolist(),
            ticktext=[str(y) for y in yearly_trend["year"].tolist()],
            title="Year"
        ),
        yaxis_title="Count",
        title_x=0.5,
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_the_map(df, date):
    st.subheader(f"Wildfire Map on {date}")
    if not df.empty:
        fol_map = folium.Map(
            location=[df["latitude"].mean(), df["longitude"].mean()],
            zoom_start=8,
        )
        for _, row in df.iterrows():
            confidence = row["confidence"]
            color = (
                "red" if confidence >= 70 else
                "orange" if confidence >= 40 else
                "yellow"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                popup=f"Confidence: {confidence}",
                color=color,
                fill=True,
                fill_opacity=0.7,
            ).add_to(fol_map)

        st_folium(
            fol_map,
            height=400,
            use_container_width=True,
            key="fixed_map",
            returned_objects=[]
        )
    else:
        st.warning("No wildfire data available for this date.")
    st.markdown("---")

def day_night_analysis(df):
    st.subheader("Day vs Night Proportion Analysis")
    df["hour"] = df["acq_date"].dt.hour
    day_count = df[df["hour"].between(6, 18)].shape[0]
    night_count = df[df["hour"].between(19, 23)].shape[0] + df[df["hour"].between(0, 5)].shape[0]
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
    st.subheader("Wildfire Incidents by Location")
    fig = px.scatter(
        df,
        x="longitude",
        y="latitude",
        color="confidence",
        size="confidence",
        labels={"latitude": "Latitude", "longitude": "Longitude", "confidence": "Confidence"},
        title="Wildfire Locations and Confidence",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

def display_model_info(model):
    st.markdown("### Model Information")
    if hasattr(model, "feature_importances_"):
        feature_names = ["latitude", "longitude", "brightness", "scan", "track", "bright_t31", "frp", "daynight"]
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            st.write({name: f"{imp:.4f}" for name, imp in zip(feature_names, importances)})
        else:
            st.write("Feature importances not available.")
    else:
        st.write("No feature importance info available.")

def get_user_input(df):
    st.subheader("Input Data for Prediction")
    daynight_dict = {"D": 1, "N": 0}
    with st.form("filter_form"):
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
        submitted = st.form_submit_button("Submit")
        if submitted:
            return user_input_df
    return None

# ====================== MAIN APP =========================
def main():
    st.set_page_config(
        page_title="Wildfire Visualization",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🔥 EmberAlert - The Wildfire Detection System")
    st.markdown("---")

    country = st.sidebar.selectbox("🌍 Select Country", ['Nepal', 'South Korea'])
    with st.spinner("Loading data... Please wait!"):
        df = load_data(country)
    if df.empty:
        st.warning("No data loaded.")
        return

    tabs = st.tabs([
        "Overview",
        "Map & Filter",
        "Reports",
        "Statistics",
        "ML Prediction",
        "Subscribe/Manage",
        "Fire Risk Visualization",
        "Help/FAQ"
    ])

    # 0 - Overview & Trends
    with tabs[0]:
        show_overview(df, country)
        st.markdown("---")
        show_yearly_trend(df)

    # 1 - Map & Location Filter (Date Picker moved here)
    with tabs[1]:
        unique_dates = sorted(df['acq_date'].dt.date.unique())
        if unique_dates:
            selected_date = st.selectbox("📅 Select Date (Fire Data Available)", unique_dates, index=len(unique_dates) - 1)
        else:
            selected_date = pd.Timestamp.today().date()
        st.session_state['selected_date'] = selected_date
        st.info(f"Selected Date: **{selected_date}**")

        # ------- Severity Legend in the body --------
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 10px;">
            <b>Severity Legend:</b>
            <span style='color: red; font-size: 22px;'>&#11044;</span> <span style="margin-right:15px;">High (Confidence ≥ 70)</span>
            <span style='color: orange; font-size: 22px;'>&#11044;</span> <span style="margin-right:15px;">Medium (40 ≤ Confidence &lt; 70)</span>
            <span style='color: gold; font-size: 22px;'>&#11044;</span> <span>Low (Confidence &lt; 40)</span>
        </div>
        """, unsafe_allow_html=True)
        # --------------------------------------------

        filtered_data = df[df['acq_date'].dt.date == selected_date]
        st.write(f"Showing data for: **{selected_date}**")
        if not filtered_data.empty:
            st.sidebar.markdown("### 🔎 Filter by Location")
            lat_min_val = float(filtered_data['latitude'].min())
            lat_max_val = float(filtered_data['latitude'].max())
            lon_min_val = float(filtered_data['longitude'].min())
            lon_max_val = float(filtered_data['longitude'].max())

            if lat_min_val == lat_max_val:
                st.sidebar.write(f"Only one latitude: {lat_min_val}")
                lat_min = lat_max = lat_min_val
            else:
                lat_min = st.sidebar.slider('Min Latitude', lat_min_val, lat_max_val, lat_min_val)
                lat_max = st.sidebar.slider('Max Latitude', lat_min_val, lat_max_val, lat_max_val)

            if lon_min_val == lon_max_val:
                st.sidebar.write(f"Only one longitude: {lon_min_val}")
                lon_min = lon_max = lon_min_val
            else:
                lon_min = st.sidebar.slider('Min Longitude', lon_min_val, lon_max_val, lon_min_val)
                lon_max = st.sidebar.slider('Max Longitude', lon_min_val, lon_max_val, lon_max_val)

            filtered_data = filtered_data[
                (filtered_data['latitude'] >= lat_min) & (filtered_data['latitude'] <= lat_max) &
                (filtered_data['longitude'] >= lon_min) & (filtered_data['longitude'] <= lon_max)
            ]
            plot_the_map(filtered_data, selected_date)
        else:
            st.warning("No data for this date.")

    # 2 - Reports/Downloads
    with tabs[2]:
        unique_dates = sorted(df['acq_date'].dt.date.unique())
        selected_date = st.session_state.get('selected_date', unique_dates[-1] if unique_dates else pd.Timestamp.today().date())
        filtered_data = df[df['acq_date'].dt.date == selected_date]
        st.markdown("### 📄 Fire Risk Report Summary")
        st.info(summarize_fire_data(filtered_data))
        if not filtered_data.empty:
            st.download_button(
                "Download Fire Data (CSV)",
                data=filtered_data.to_csv(index=False),
                file_name=f"wildfire_{country}_{selected_date}.csv",
                mime="text/csv"
            )
            pdf_bytes = generate_pdf_report(filtered_data, country, selected_date)
            st.download_button(
                "Download Fire Risk Report (PDF)",
                data=pdf_bytes,
                file_name=f"wildfire_report_{country}_{selected_date}.pdf",
                mime="application/pdf"
            )

    # 3 - Statistics
    with tabs[3]:
        unique_dates = sorted(df['acq_date'].dt.date.unique())
        selected_date = st.session_state.get('selected_date', unique_dates[-1] if unique_dates else pd.Timestamp.today().date())
        filtered_data = df[df['acq_date'].dt.date == selected_date]
        st.subheader(f"Statistical Analysis for {selected_date}")
        col1, col2 = st.columns(2)
        with col1:
            day_night_analysis(filtered_data)
        with col2:
            plot_wildfire_by_lat_long(filtered_data)

    # 4 - ML Prediction & Alerting
    with tabs[4]:
        st.subheader("🔥 Predict Fire Risk for Any Location")
        user_input_df = get_user_input(df)
        st.markdown("Fire risk prediction based on latitude and longitude.\n\n**Note:** The model is trained on historical data and may not reflect real-time conditions.")
        model, scaler = load_model_artifacts()
        display_model_info(model)
        if user_input_df is not None:
            st.dataframe(user_input_df, use_container_width=True)
            scaled_data = scaler.transform(user_input_df)
            prediction = model.predict(scaled_data)
            prediction_value = prediction[0] if isinstance(prediction, (np.ndarray, list)) else prediction
            st.success(f"Predicted Confidence: **{prediction_value:.2f}**")
            severity = get_severity(prediction_value)
            location_name = geocode_lat_long_to_name(
                user_input_df["latitude"][0],
                user_input_df["longitude"][0],
                prediction_value,
            )
            location_str = location_name if location_name else "Unknown"
            st.markdown(f"**Predicted 🔥 Severity:** {severity} for **Location:** {location_str}")

            notify_matching_subscribers(
                float(user_input_df["latitude"][0]),
                float(user_input_df["longitude"][0]),
                float(prediction_value),
                severity,
                location_str
            )

    # 5 - Subscribe / Manage (SUPABASE)
    with tabs[5]:
        tab = st.radio("🚦 Alert System", ["Subscribe to Alerts", "Manage Your Alerts"])
        if tab == "Subscribe to Alerts":
            st.markdown("### Subscribe to Wildfire Alerts")
            with st.form("subscription_form"):
                email = st.text_input("📧 Email", key="sub_email")
                col1, col2 = st.columns(2)
                with col1:
                    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.0, step=0.01)
                with col2:
                    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=84.0, step=0.01)
                severity_level = st.multiselect("Alert Severity Level(s)", severity_options)
                submitted = st.form_submit_button("Subscribe")
            if submitted:
                if not is_valid_email(email):
                    st.error("❌ Invalid email! Use domains: " + ", ".join(allowed_domains))
                elif not severity_level:
                    st.error("❌ Select at least one severity level.")
                else:
                    with st.spinner("Subscribing..."):
                        inserted = 0
                        already_subscribed = 0
                        for sev in severity_level:
                            # Check if subscription exists
                            query = supabase.table("subscription").select("*").eq("email", email).eq("latitude", lat).eq("longitude", lon).eq("severity_level", sev)
                            count = len(query.execute().data)
                            if count == 0:
                                # Insert new
                                supabase.table("subscription").insert({
                                    "email": email,
                                    "latitude": lat,
                                    "longitude": lon,
                                    "severity_level": sev,
                                    # 'subscribed_at': auto by supabase
                                }).execute()
                                inserted += 1
                            else:
                                already_subscribed += 1
                        st.session_state['user_email'] = email

                        if inserted > 0:
                            st.success(f"✅ You successfully subscribed to {inserted} alert(s)!")
                            send_subscription_confirmation_email(email, lat, lon, severity_level)
                            if already_subscribed > 0:
                                st.info(f"⚠️ Already subscribed to {already_subscribed} selected alert(s).")
                        else:
                            st.info("⚠️ Already subscribed to all selected alert(s).")

        elif tab == "Manage Your Alerts":
            st.markdown("### Manage Your Alert Subscriptions")
            manage_email = st.text_input("📧 Enter your email", value=st.session_state.get('user_email', ''), key="manage_email")
            if manage_email:
                if not is_valid_email(manage_email):
                    st.error("❌ Invalid email address!")
                else:
                    subs = supabase.table("subscription").select("*").eq("email", manage_email).order("subscribed_at", desc=True).execute().data
                    if subs:
                        st.markdown(f"**You have {len(subs)} active subscription(s).**")
                        for sub in subs:
                            with st.expander(
                                f"📍 ({float(sub['latitude']):.2f}, {float(sub['longitude']):.2f}) - {sub['severity_level']} (Since {sub['subscribed_at'][:10]})"
                            ):
                                # Unsubscribe button
                                if st.button("Unsubscribe", key=f"del_{sub['subscription_id']}"):
                                    supabase.table("subscription").delete().eq("id", sub['subscription_id']).execute()
                                    st.success("✅ Unsubscribed successfully!")
                                    st.experimental_rerun()

                                # Toggle edit form
                                if f'editing_{sub["subscription_id"]}' not in st.session_state:
                                    st.session_state[f'editing_{sub["subscription_id"]}'] = False

                                if not st.session_state[f'editing_{sub["subscription_id"]}']:
                                    if st.button("Edit Subscription", key=f"edit_btn_{sub['subscription_id']}"):
                                        st.session_state[f'editing_{sub["subscription_id"]}'] = True
                                else:
                                    with st.form(f"edit_form_{sub['subscription_id']}"):
                                        new_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=float(sub["latitude"]), key=f"lat_{sub['id']}")
                                        new_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=float(sub["longitude"]), key=f"lon_{sub['id']}")
                                        new_sev = st.selectbox("Severity Level", severity_options, index=severity_options.index(sub["severity_level"]), key=f"sev_{sub['id']}")
                                        update_submit = st.form_submit_button("Update Subscription")
                                        if update_submit:
                                            supabase.table("subscription").update({
                                                "latitude": new_lat,
                                                "longitude": new_lon,
                                                "severity_level": new_sev
                                            }).eq("subscription_id", sub['subscription_id']).execute()
                                            st.success("✅ Subscription updated!")
                                            st.session_state[f'editing_{sub["subsription_id"]}'] = False
                                            st.experimental_rerun()
                                    if st.button("Cancel Edit", key=f"cancel_{sub['id']}"):
                                        st.session_state[f'editing_{sub["id"]}'] = False
                    else:
                        st.info("ℹ️ No active subscriptions found.")

    # 6 - Fire Risk Visualization
    with tabs[6]:
        st.markdown("## Fire Risk Visualization")
        min_lat, max_lat = float(df["latitude"].min()), float(df["latitude"].max())
        min_lon, max_lon = float(df["longitude"].min()), float(df["longitude"].max())
        unique_dates = sorted(df['acq_date'].dt.date.unique())
        start_date = st.date_input("Start Date", value=unique_dates[0], min_value=unique_dates[0], max_value=unique_dates[-1])
        end_date = st.date_input("End Date", value=unique_dates[-1], min_value=start_date, max_value=unique_dates[-1])
        lat_range = st.slider("Latitude Range", min_lat, max_lat, (min_lat, max_lat))
        lon_range = st.slider("Longitude Range", min_lon, max_lon, (min_lon, max_lon))
        mask = (
            (df['acq_date'].dt.date >= start_date) &
            (df['acq_date'].dt.date <= end_date) &
            (df['latitude'] >= lat_range[0]) & (df['latitude'] <= lat_range[1]) &
            (df['longitude'] >= lon_range[0]) & (df['longitude'] <= lon_range[1])
        )
        filtered_viz = df[mask]
        st.info(f"Total records found: {len(filtered_viz)}")
        if not filtered_viz.empty:
            filtered_viz['severity'] = filtered_viz['confidence'].apply(get_severity)
            daily_counts = filtered_viz.groupby([filtered_viz['acq_date'].dt.date, 'severity']).size().unstack(fill_value=0)
            st.bar_chart(daily_counts)
            st.map(filtered_viz[['latitude', 'longitude']])
            st.markdown(summarize_fire_data(filtered_viz))
            vis_pdf_bytes = generate_viz_pdf_report(
                filtered_viz,
                region=(lat_range, lon_range),
                date_range=(start_date, end_date)
            )
            st.download_button(
                "Download Fire Risk Visualization PDF",
                data=vis_pdf_bytes,
                file_name=f"fire_risk_viz_{start_date}_to_{end_date}.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No fire records in selected region and date range.")

    # 7 - Help / FAQ
    with tabs[7]:
        st.title("❓ Help & FAQ")
        st.markdown("Welcome to the EmberAlert support page. Here you’ll find answers to common questions and guidance on using each feature.")

        with st.expander("🧭 What does each tab do?"):
            st.markdown("""
            - **Overview**: Shows total records, missing data, and yearly trends.
            - **Map & Filter**: Visualize fire incidents by selecting date and filtering region.
            - **Reports**: Download wildfire data reports as CSV or PDF.
            - **Statistics**: See day vs night fire comparison and location-wise scatter plot.
            - **ML Prediction**: Predict fire risk using custom input data.
            - **Subscribe / Manage**: Get alerts via email based on fire severity and location.
            - **Fire Risk Visualization**: View daily severity chart and download visual report.
            """)

        with st.expander("📈 What is 'Confidence', 'Severity' and 'FRP'?"):
            st.markdown("""
            - **Confidence**: Probability (%) that the satellite detected a wildfire.
            - **Severity**:
                - 🔴 **High**: Confidence ≥ 70%
                - 🟠 **Medium**: 40%–69%
                - 🟡 **Low**: < 40%
            - **FRP (Fire Radiative Power)**: Intensity of fire measured in megawatts.
            - **Day/Night**: Whether fire was detected during daytime or nighttime.
            """)

        with st.expander("🔔 How do wildfire alerts work?"):
            st.markdown("""
            - Users can **subscribe** with their email, coordinates, and severity preference.
            - If a fire occurs within **50 km** of your location and matches selected severity, you’ll get an **email alert**.
            - Supported email domains: `gmail.com`, `naver.com`, `yahoo.com`, etc.
            - You can also **edit or unsubscribe** anytime in the 'Manage Alerts' section.
            """)

        with st.expander("🧠 How does the ML Prediction work?"):
            st.markdown("""
            - Input features like latitude, brightness, scan, FRP, etc.
            - A machine learning model trained on historical wildfire data predicts **confidence score (0–100%)**.
            - Based on confidence, it also shows predicted **severity level**.
            - Alerts can be triggered if prediction matches a subscription.
            """)

        with st.expander("💾 How to download reports?"):
            st.markdown("""
            - Go to **Reports** tab to download filtered fire data (CSV or PDF).
            - Go to **Fire Risk Visualization** tab for PDF with bar charts.
            - All files are auto-named and downloadable directly in browser.
            """)

        with st.expander("⚠️ Common Issues & Fixes"):
            st.markdown("""
            - **Invalid Email**: Only allowed domains can be used.
            - **DB Connection Failed**: Ensure MySQL is running with correct credentials.
            - **Slider min=max error**: Happens when only one value is present. Try a different date.
            - **No Data Warning**: It means there were no fire records for that filter.
            """)

        with st.expander("📧 Need more help?"):
            st.markdown("""
            - For technical issues or feedback, email us at **ldawa9808@gmail.com**
            - We're here to help you stay safe from wildfires! 🔥
            """)

if __name__ == "__main__":
    main()
