from pathlib import Path
import geopandas as gpd
import joblib
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
from datetime import datetime
import time
import numpy as np
import json

try:
    from introduction_page import render_introduction_page
    from chatbot_page import render_chatbot_page
    from auth_page import ensure_authenticated, render_sidebar_user_footer
    _HAS_AUTH = True
except ImportError:
    _HAS_AUTH = False
    def ensure_authenticated(): pass
    def render_sidebar_user_footer(): pass
    def render_introduction_page(): st.info("Introduction page not found.")
    def render_chatbot_page(): st.info("Chatbot page not found.")

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="RainGuardAI", layout="wide", page_icon="🌊")

# ----------------------------------
# PATHS
# ----------------------------------
BASE_DIR          = Path(__file__).resolve().parents[1]
RL_MODEL_PATH     = BASE_DIR / "Phase 3" / "rl" / "alert_policy.pkl"
LOW_ZONES_PATH    = BASE_DIR / "data" / "processed" / "low_lying_zones.geojson"
IMPACT_ZONES_PATH = BASE_DIR / "data" / "processed" / "flood_impact_zones.geojson"
POPULATION_PATH   = BASE_DIR / "data" / "raw" / "Demographics" / "mumbai_wardwise_population_data.csv"

CUSTOM_LOGO_PATH = r"D:\Riverthon\dashboard\assets\RainGuard.png"
LOGO_CANDIDATES = [
    BASE_DIR / "dashboard" / "assets" / "rainguard_logo.png",
    BASE_DIR / "dashboard" / "assets" / "rainguard_logo.jpg",
    BASE_DIR / "dashboard" / "assets" / "rainguard_logo.jpeg",
    BASE_DIR / "dashboard" / "assets" / "logo.png",
    BASE_DIR / "dashboard" / "assets" / "logo.jpg",
    BASE_DIR / "assets" / "rainguard_logo.png",
    BASE_DIR / "assets" / "rainguard_logo.jpg",
]

# Public basemap style (no Mapbox token needed)
MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

# CV paths
CV_AFTER_PATH     = BASE_DIR / "output" / "after_flood_r10m.jpg"
CV_BEFORE_PATH    = BASE_DIR / "cv" / "sample_images" / "before_flood_r10m.jpg"
CV_HL_PATH        = BASE_DIR / "cv" / "output" / "highlighted_after.jpg"
CV_PANEL_PATH     = BASE_DIR / "cv" / "output" / "before_after_panel.jpg"
CV_MASK_PATH      = BASE_DIR / "cv" / "output" / "flood_mask.jpg"
CV_ENH_PATH       = BASE_DIR / "cv" / "output" / "enhanced_before.jpg"

# ----------------------------------
# SESSION STATE
# ----------------------------------
for key, val in [
    ('alert_sent', False), ('alert_history', []),
    ('show_confirmation', False), ('gemini_summary', None),
    ('ai_full_summary', None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

ensure_authenticated()

# ============================================================
# BACKGROUND MUSIC
# ============================================================
MUSIC_LOCAL_PATH = r"D:\Riverthon\dashboard\assets\Intestellar.mp3"
MUSIC_FALLBACK   = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"

def _get_music_url():
    """Return local file as base64 data-URI, else fallback URL."""
    import base64
    local = Path(MUSIC_LOCAL_PATH)
    if not local.exists():
        local = BASE_DIR / "dashboard" / "assets" / "Intestellar.mp3"
    if local.exists():
        with open(local, "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode()
        return f"data:audio/mpeg;base64,{_b64}"
    return MUSIC_FALLBACK

def inject_bg_music(muted: bool):
    """Inject background music with reliable mute/unmute."""
    music_src  = _get_music_url()
    muted_js   = "true" if muted else "false"
    volume_val = "0.0"  if muted else "0.55"

    html_code = f"""
<script>
(function() {{
  var MUTED  = {muted_js};
  var VOLUME = {volume_val};
  var SRC    = "{music_src if music_src.startswith("http") else ""}";

  var audio = window._rgBgAudio;

  if (!audio) {{
    audio = document.createElement('audio');
    audio.id   = 'rg-bgm';
    audio.loop = true;
    audio.style.display = 'none';
    document.body.appendChild(audio);
    window._rgBgAudio = audio;
  }}

  audio.volume = VOLUME;
  audio.muted  = MUTED;

  if (!MUTED && audio.paused) {{
    var p = audio.play();
    if (p !== undefined) {{
      p.catch(function(e) {{
        var tries = 0;
        var iv = setInterval(function() {{
          tries++;
          if (!window._rgBgAudio.paused) {{ clearInterval(iv); return; }}
          window._rgBgAudio.play().catch(function(){{}});
          if (tries > 30) clearInterval(iv);
        }}, 500);
      }});
    }}
  }}

  if (MUTED) {{
    audio.muted  = true;
    audio.volume = 0.0;
  }}

}})();
</script>
<audio id="rg-bgm-src" src="{music_src}"
       autoplay loop style="display:none"
       {"muted" if muted else ""}></audio>
<script>
(function() {{
  var el = document.getElementById('rg-bgm-src');
  if (el) {{
    el.muted  = {muted_js};
    el.volume = {volume_val};
    if (!{muted_js} && el.paused) el.play().catch(function(){{}});
  }}
}})();
</script>
"""
    st.markdown(html_code, unsafe_allow_html=True)

# Inject music after authentication
inject_bg_music(st.session_state.get("audio_muted", False))


def _pick_logo_path():
    custom = CUSTOM_LOGO_PATH.strip()
    if custom:
        custom_path = Path(custom)
        if custom_path.exists():
            return custom_path
    for path in LOGO_CANDIDATES:
        if path.exists():
            return path
    return None


def render_brand_banner():
    st.markdown("""
        <style>
        .brand-sep {
            height: 3px; margin: 8px 0 14px 0; border-radius: 999px;
            background: linear-gradient(90deg,#38BDF8,#2563EB,#22D3EE,#2563EB,#38BDF8);
            background-size: 240% 100%;
            box-shadow: 0 0 10px rgba(56,189,248,0.75),0 0 16px rgba(37,99,235,0.55);
            animation: brandFlow 2.2s linear infinite;
        }
        @keyframes brandFlow { 0%{background-position:0% 0} 100%{background-position:220% 0} }
        </style>""", unsafe_allow_html=True)
    logo_path = _pick_logo_path()
    if logo_path is not None:
        left, center, right = st.columns([1, 4, 1])
        with center:
            st.image(str(logo_path), width="stretch")
    else:
        st.markdown("<h1 style='text-align:center;margin:0;color:#0B3A75;letter-spacing:0.4px;'>RainGuard</h1>",
                    unsafe_allow_html=True)
    st.markdown("<div class='brand-sep'></div>", unsafe_allow_html=True)


def render_sidebar_brand():
    st.sidebar.markdown("""
        <style>
        .sidebar-sep { height:2px;margin:8px 0 12px 0;border-radius:999px;
          background:linear-gradient(90deg,#22D3EE,#2563EB,#38BDF8);
          box-shadow:0 0 8px rgba(56,189,248,0.65); }
        </style>""", unsafe_allow_html=True)
    logo_path = _pick_logo_path()
    if logo_path is not None:
        st.sidebar.image(str(logo_path), width=120)
    else:
        st.sidebar.markdown("<p style='text-align:center;font-weight:700;color:#0B3A75;margin:0'>RainGuard</p>",
                            unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)


# ============================================================
# HELPER — risk zone from flood probability
# ============================================================
def classify_zone(risk_prob):
    if risk_prob >= 0.75:   return "RED"
    elif risk_prob >= 0.50: return "ORANGE"
    else:                   return "GREEN"

def zone_color_rgb(zone):
    return {"RED": [220, 38, 38, 190], "ORANGE": [249, 115, 22, 180],
            "GREEN": [34, 197, 94, 170]}.get(zone, [100, 100, 100, 150])

# ============================================================
# POPULATION ZONE ANALYSIS
# ============================================================
@st.cache_data
def load_population():
    return pd.read_csv(POPULATION_PATH)

def compute_population_zones(risk_prob, pop_df):
    df = pop_df.copy()
    df['pop_density'] = df['Population_2025'] / df['Area_in_Sq_km']
    max_density = df['pop_density'].max()
    df['density_factor'] = df['pop_density'] / max_density
    df['ward_risk'] = (0.80 * risk_prob) + (0.20 * df['density_factor'] * risk_prob)
    df['ward_risk'] = df['ward_risk'].clip(0, 1)
    df['Zone'] = df['ward_risk'].apply(classify_zone)
    return df

WARD_COORDS = {
    'A':   (18.9400, 72.8370), 'B':   (18.9550, 72.8390),
    'C':   (18.9640, 72.8280), 'D':   (18.9700, 72.8180),
    'E':   (18.9780, 72.8140), 'F/S': (19.0050, 72.8250),
    'F/N': (19.0280, 72.8380), 'G/S': (19.0100, 72.8450),
    'G/N': (19.0350, 72.8500), 'H/E': (19.0550, 72.8680),
    'H/W': (19.0700, 72.8380), 'K/E': (19.0900, 72.8950),
    'K/W': (19.1000, 72.8500), 'P/S': (19.1150, 72.8600),
    'P/N': (19.1500, 72.8700), 'R/S': (19.1300, 72.8800),
    'R/C': (19.1700, 72.9100), 'R/N': (19.1900, 72.8650),
    'L':   (19.0950, 72.9250), 'M/E': (19.1200, 72.9400),
    'M/W': (19.0750, 72.9150), 'N':   (19.1450, 72.9300),
    'S':   (19.1050, 72.9550), 'T':   (19.1750, 72.9500),
}

def build_ward_map_data(zone_df):
    rows = []
    for _, row in zone_df.iterrows():
        ward = row['Ward']
        if ward in WARD_COORDS:
            lat, lon = WARD_COORDS[ward]
            rows.append({
                'Ward': ward, 'lat': lat, 'lon': lon,
                'Population_2025': int(row['Population_2025']),
                'Zone': row['Zone'],
                'color': zone_color_rgb(row['Zone']),
                'Ward_Category': row['Ward_Category'],
                'risk_score': round(row['ward_risk'], 3),
            })
    return pd.DataFrame(rows)

# ============================================================
# INFRASTRUCTURE ZONE ANALYSIS
# ============================================================
@st.cache_data
def load_infrastructure():
    gdf = gpd.read_file(IMPACT_ZONES_PATH).to_crs(epsg=4326)
    gdf['lat'] = gdf.geometry.centroid.y
    gdf['lon'] = gdf.geometry.centroid.x
    return gdf

def assign_infra_zones(infra_gdf, risk_prob):
    df = infra_gdf.copy()
    weights = {'Healthcare': 1.10, 'Shelter': 1.05,
               'Police': 1.00, 'Fire_Station': 1.00, 'Waterways': 0.70}
    df['infra_risk'] = df['infrastructure_type'].map(
        lambda t: min(risk_prob * weights.get(t, 1.0), 1.0))
    df['Zone'] = df['infra_risk'].apply(classify_zone)
    df['color'] = df['Zone'].apply(lambda z: zone_color_rgb(z))
    return df

# ============================================================
# GENAI SUMMARY
# ============================================================
def generate_gemini_full_summary(risk_level, risk_prob, rainfall, soil_moisture,
                                  elevation_risk, alert_decision, zone_summary,
                                  infra_summary, gemini_api_key):
    try:
        import google.generativeai as genai
        if not gemini_api_key:
            return None, "Please enter Gemini API key in sidebar."
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
You are RainGuardAI — an expert flood emergency analyst for Mumbai, Maharashtra, India.
Generate a comprehensive, professional flood situation report based on the data below.

=== FLOOD PREDICTION ===
Risk Level: {risk_level}
Risk Probability: {risk_prob:.2%}
Rainfall: {rainfall:.1f} mm/hr
Soil Saturation: {soil_moisture:.2%}
Elevation Risk: {elevation_risk:.2f}
Alert Decision: {"SEND ALERT - Immediate Action Required" if alert_decision else "Continue Monitoring"}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

=== POPULATION ZONE SUMMARY ===
RED (HIGH RISK) Zone:
  - Wards: {zone_summary['red']['wards']}
  - Population at risk: {zone_summary['red']['population']:,}
  - Ward names: {zone_summary['red']['ward_names']}

ORANGE (MEDIUM RISK) Zone:
  - Wards: {zone_summary['orange']['wards']}
  - Population: {zone_summary['orange']['population']:,}
  - Ward names: {zone_summary['orange']['ward_names']}

GREEN (LOW RISK) Zone:
  - Wards: {zone_summary['green']['wards']}
  - Population: {zone_summary['green']['population']:,}
  - Ward names: {zone_summary['green']['ward_names']}

=== CRITICAL INFRASTRUCTURE IMPACT ===
RED Zone: {json.dumps(infra_summary['red'])}
ORANGE Zone: {json.dumps(infra_summary['orange'])}
GREEN Zone: {json.dumps(infra_summary['green'])}

Write a professional emergency situation report with these sections:

## SITUATION OVERVIEW
## POPULATION AT RISK
## CRITICAL INFRASTRUCTURE STATUS
## IMMEDIATE ACTIONS REQUIRED
## 6-HOUR OUTLOOK
## NDMA COMPLIANCE NOTES

Be specific with numbers. Use ward names. Be urgent but professional.
"""
        response = model.generate_content(prompt)
        return response.text, None
    except ImportError:
        return None, "Install: pip install google-generativeai"
    except Exception as e:
        return None, f"Gemini Error: {str(e)}"

# ============================================================
# NOTIFICATIONS
# ============================================================
def send_telegram_alert(bot_token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
        resp.raise_for_status()
        return True, "Telegram alert sent!"
    except Exception as e:
        return False, f"Telegram failed: {e}"

def send_email_alert(to_email, subject, message, smtp_server, smtp_port, from_email, password):
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        msg = MIMEMultipart()
        msg['From'] = from_email; msg['To'] = to_email; msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls(); server.login(from_email, password)
        server.send_message(msg); server.quit()
        return True, "Email sent!"
    except Exception as e:
        return False, f"Email failed: {e}"

# ============================================================
# WEATHER DATA
# ============================================================
def _default_weather_payload():
    return {'rainfall': 0.0, 'humidity': 0.60, 'temp': 28.0,
            'description': 'Estimated',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@st.cache_data(ttl=300)
def fetch_open_meteo_data(lat, lon):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&current=temperature_2m,relative_humidity_2m,precipitation,rain&timezone=auto")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        d = resp.json(); cur = d.get('current', {})
        return {
            'rainfall':    cur.get('rain', 0) + cur.get('precipitation', 0),
            'humidity':    cur.get('relative_humidity_2m', 60) / 100,
            'temp':        cur.get('temperature_2m', 28),
            'description': 'Live data',
            'timestamp':   cur.get('time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        }
    except Exception:
        return _default_weather_payload()

@st.cache_data(ttl=300)
def fetch_weatherapi_data(lat, lon, api_key):
    if not api_key: return None
    try:
        url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}&aqi=no"
        resp = requests.get(url, timeout=20); resp.raise_for_status()
        payload = resp.json(); current = payload.get("current", {}); location = payload.get("location", {})
        return {
            'rainfall': float(current.get("precip_mm", 0.0) or 0.0),
            'humidity': float(current.get("humidity", 60.0) or 60.0) / 100,
            'temp':     float(current.get("temp_c", 28.0) or 28.0),
            'description': current.get("condition", {}).get("text", "Live data"),
            'timestamp': location.get("localtime", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        }
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_quoamean_data(lat, lon, api_key):
    if not api_key: return None
    try:
        url = f"https://devapi.qweather.com/v7/weather/now?location={lon},{lat}&key={api_key}"
        resp = requests.get(url, timeout=20); resp.raise_for_status()
        payload = resp.json(); now_data = payload.get("now", {})
        if payload.get("code") != "200": return None
        return {
            'rainfall':    float(now_data.get("precip", 0.0) or 0.0),
            'humidity':    float(now_data.get("humidity", 60.0) or 60.0) / 100,
            'temp':        float(now_data.get("temp", 28.0) or 28.0),
            'description': now_data.get("text", "Live data"),
            'timestamp':   now_data.get("obsTime", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        }
    except Exception:
        return None

@st.cache_data(ttl=300)
def reverse_geocode_openstreetmap(lat, lon):
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"lat": lat, "lon": lon, "format": "jsonv2"}
        headers = {"User-Agent": "RainGuardAI/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json().get("display_name")
    except Exception:
        return None

def fetch_weather_by_provider(provider, lat, lon, api_key):
    if provider == "OpenStreetMap":
        weather_data = fetch_open_meteo_data(lat, lon)
        place_name = reverse_geocode_openstreetmap(lat, lon)
        if place_name:
            weather_data["description"] = f"OpenStreetMap: {place_name}"
        return weather_data, None
    if provider == "WeatherAPI":
        if not api_key:
            return fetch_open_meteo_data(lat, lon), "WeatherAPI key is required. Using Open-Meteo fallback."
        weather_data = fetch_weatherapi_data(lat, lon, api_key)
        if weather_data: return weather_data, None
        return fetch_open_meteo_data(lat, lon), "WeatherAPI request failed. Using Open-Meteo fallback."
    if provider == "QuoaMean API":
        if not api_key:
            return fetch_open_meteo_data(lat, lon), "QuoaMean API key is required. Using Open-Meteo fallback."
        weather_data = fetch_quoamean_data(lat, lon, api_key)
        if weather_data: return weather_data, None
        return fetch_open_meteo_data(lat, lon), "QuoaMean API request failed. Using Open-Meteo fallback."
    return fetch_open_meteo_data(lat, lon), None

def estimate_soil_moisture(humidity, rainfall, temp):
    return min(humidity * 0.5 + min(rainfall / 50, 0.4) + max(0, (30 - temp) / 30) * 0.1, 1.0)

def get_elevation_risk(lat, lon):
    try:
        resp = requests.get(f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}", timeout=10)
        elev = resp.json()['results'][0]['elevation']
        risk = 0.9 if elev < 5 else 0.7 if elev < 10 else 0.5 if elev < 20 else 0.3 if elev < 50 else 0.1
        return risk, elev
    except Exception:
        return 0.7, None

# ------------------------------------------------------------------
# REAL ANN MODEL
# ------------------------------------------------------------------
ANN_MODEL_PATH   = BASE_DIR / "model" / "ann_flood_model.keras"
ANN_SCALER_PATH  = BASE_DIR / "model" / "scaler.pkl"
ANN_FEATURE_COLS = ["Intensity", "soil_moisture_index", "elevation_risk", "humidity"]

@st.cache_resource
def load_ann_model():
    try:
        from tensorflow.keras.models import load_model as keras_load
        if ANN_MODEL_PATH.exists() and ANN_SCALER_PATH.exists():
            return keras_load(str(ANN_MODEL_PATH)), joblib.load(str(ANN_SCALER_PATH))
        return None, None
    except Exception:
        return None, None

def ann_risk_proxy(rainfall, soil_moisture, elevation_risk, humidity=70.0):
    ann_model, ann_scaler = load_ann_model()
    if ann_model is not None and ann_scaler is not None:
        try:
            X = np.array([[rainfall, soil_moisture, elevation_risk, humidity]])
            X_scaled = ann_scaler.transform(X)
            prob = float(ann_model.predict(X_scaled, verbose=0)[0][0])
            return min(max(prob, 0.0), 1.0)
        except Exception:
            pass
    return min(max(0.5*(rainfall/300) + 0.25*soil_moisture +
                   0.15*elevation_risk + 0.10*(humidity/100), 0), 1)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_rl_model():
    return joblib.load(RL_MODEL_PATH)

# ============================================================
# SIDEBAR
# ============================================================
render_brand_banner()
render_sidebar_brand()

app_section = st.sidebar.radio(
    "Open Section",
    ["Introduction", "Dashboard", "Chatbot Facility"],
    index=0, label_visibility="collapsed",
)
if app_section == "Introduction":
    render_introduction_page()
    render_sidebar_user_footer()
    st.stop()
if app_section == "Chatbot Facility":
    render_chatbot_page()
    render_sidebar_user_footer()
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Configuration")
data_mode = st.sidebar.radio("Data Source", ["Real-Time API", "Manual Input"])

if data_mode == "Real-Time API":
    st.sidebar.subheader("Location")
    latitude  = st.sidebar.number_input("Latitude",  value=19.0760, format="%.4f")
    longitude = st.sidebar.number_input("Longitude", value=72.8777, format="%.4f")
    st.sidebar.subheader("API Selection")
    selected_api = st.sidebar.selectbox("Select API",
        ["OpenStreetMap", "WeatherAPI", "QuoaMean API"], index=0)
    selected_api_key = ""
    if selected_api in ("WeatherAPI", "QuoaMean API"):
        selected_api_key = st.sidebar.text_input(f"{selected_api} Key", type="password")
    refresh_clicked = st.sidebar.button("Refresh Now")
    if refresh_clicked:
        fetch_open_meteo_data.clear()
        fetch_weatherapi_data.clear()
        fetch_quoamean_data.clear()
        reverse_geocode_openstreetmap.clear()
    weather_data, weather_warning = fetch_weather_by_provider(
        selected_api, latitude, longitude, selected_api_key)
    if weather_warning:
        st.sidebar.warning(weather_warning)
    if weather_data:
        st.sidebar.success(f"Updated: {weather_data['timestamp']}")
        st.sidebar.caption(f"Source: {selected_api}")
        st.sidebar.metric("Rainfall", f"{weather_data['rainfall']:.1f} mm/hr")
        st.sidebar.metric("Humidity",  f"{weather_data['humidity']*100:.0f}%")
        soil_moisture  = estimate_soil_moisture(weather_data['humidity'], weather_data['rainfall'], weather_data['temp'])
        elevation_risk, elev = get_elevation_risk(latitude, longitude)
        if elev:
            st.sidebar.metric("Elevation", f"{elev:.1f}m")
        rainfall = weather_data['rainfall']
        humidity = weather_data['humidity'] * 100
    else:
        rainfall = 60; soil_moisture = 0.40; elevation_risk = 0.50; humidity = 65.0
else:
    st.sidebar.subheader("Manual Inputs")
    rainfall       = st.sidebar.slider("Rainfall (mm/hr)", 0, 300, 60)
    soil_moisture  = st.sidebar.slider("Soil Saturation",  0.0, 1.0, 0.40)
    elevation_risk = st.sidebar.slider("Elevation Risk",   0.0, 1.0, 0.50)
    humidity       = st.sidebar.slider("Humidity (%)",     0,   100, 65)

st.sidebar.markdown("---")
st.sidebar.subheader("Gemini AI")
enable_gemini  = st.sidebar.checkbox("Enable Gemini Summary")
gemini_api_key = ""
if enable_gemini:
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    st.session_state['gemini_api_key'] = gemini_api_key

st.sidebar.markdown("---")
st.sidebar.subheader("Notifications")
notify_enabled = st.sidebar.checkbox("Enable Alerts")
notify_method = None
tg_token = tg_chat_id = ""
smtp_srv = "smtp.gmail.com"; smtp_port = 587
from_em = em_pass = to_em = ""
if notify_enabled:
    notify_method = st.sidebar.selectbox("Method", ["Telegram", "Email (SMTP)"])
    if notify_method == "Telegram":
        tg_token   = st.sidebar.text_input("Bot Token",  type="password")
        tg_chat_id = st.sidebar.text_input("Chat ID")
    elif notify_method == "Email (SMTP)":
        smtp_srv  = st.sidebar.text_input("SMTP Server", value="smtp.gmail.com")
        smtp_port = st.sidebar.number_input("Port", value=587)
        from_em   = st.sidebar.text_input("From Email")
        em_pass   = st.sidebar.text_input("Password", type="password")
        to_em     = st.sidebar.text_input("To Email")

render_sidebar_user_footer()

# Re-inject music on every rerender to keep it playing
inject_bg_music(st.session_state.get("audio_muted", False))

# ============================================================
# CORE RISK COMPUTATION
# ============================================================
rl_model   = load_rl_model()
risk_prob  = ann_risk_proxy(rainfall, soil_moisture, elevation_risk, humidity)
risk_level = "HIGH" if risk_prob >= 0.75 else "MEDIUM" if risk_prob >= 0.50 else "LOW"

X_rl = pd.DataFrame({"risk_score": [risk_prob]})
rl_pred = rl_model.predict(X_rl)[0]
if risk_level == "HIGH":
    alert_decision  = 1
    alert_reasoning = "HIGH risk — alert REQUIRED by safety protocol"
elif risk_level == "MEDIUM":
    alert_decision  = rl_pred
    alert_reasoning = f"MEDIUM risk — RL suggests: {'ALERT' if rl_pred else 'MONITOR'}"
else:
    alert_decision  = 0
    alert_reasoning = "LOW risk — continue monitoring"

# ============================================================
# TOP METRICS BAR
# ============================================================
st.markdown("---")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Rainfall",          f"{rainfall:.1f} mm/hr")
m2.metric("Humidity",          f"{humidity:.0f}%")
m3.metric("Soil Saturation",   f"{soil_moisture:.2f}")
m4.metric("Elevation Risk",    f"{elevation_risk:.2f}")
m5.metric("Flood Probability", f"{risk_prob:.2%}")
color_emoji = "🔴" if risk_level=="HIGH" else "🟠" if risk_level=="MEDIUM" else "🟢"
m6.metric("Risk Level",        f"{color_emoji} {risk_level}")

if risk_level == "HIGH":
    st.error(f"CRITICAL FLOOD ALERT — Risk probability {risk_prob:.1%}. Immediate action required.")
elif risk_level == "MEDIUM":
    st.warning(f"ELEVATED FLOOD RISK — Risk probability {risk_prob:.1%}. Prepare response teams.")
else:
    st.success(f"LOW FLOOD RISK — Risk probability {risk_prob:.1%}. Routine monitoring.")

st.info(f"RL Alert Decision: {alert_reasoning}")
st.markdown("---")

# ============================================================
# LOAD AND COMPUTE ALL DATA (needed for alert)
# ============================================================
pop_df    = load_population()
zone_df   = compute_population_zones(risk_prob, pop_df)
infra_gdf = load_infrastructure()
infra_df  = assign_infra_zones(infra_gdf, risk_prob)
ward_map  = build_ward_map_data(zone_df)

red_wards    = zone_df[zone_df['Zone'] == 'RED']
orange_wards = zone_df[zone_df['Zone'] == 'ORANGE']
green_wards  = zone_df[zone_df['Zone'] == 'GREEN']
red_infra    = infra_df[infra_df['Zone'] == 'RED']
orange_infra = infra_df[infra_df['Zone'] == 'ORANGE']
green_infra  = infra_df[infra_df['Zone'] == 'GREEN']

def infra_counts(df):
    if df.empty: return {}
    return df['infrastructure_type'].value_counts().to_dict()

zone_summary = {
    'red':    {'wards': len(red_wards),    'population': int(red_wards['Population_2025'].sum()),
               'ward_names': ', '.join(red_wards['Ward'].tolist()) or 'None'},
    'orange': {'wards': len(orange_wards), 'population': int(orange_wards['Population_2025'].sum()),
               'ward_names': ', '.join(orange_wards['Ward'].tolist()) or 'None'},
    'green':  {'wards': len(green_wards),  'population': int(green_wards['Population_2025'].sum()),
               'ward_names': ', '.join(green_wards['Ward'].tolist()) or 'None'},
}
infra_summary = {
    'red':    infra_counts(red_infra),
    'orange': infra_counts(orange_infra),
    'green':  infra_counts(green_infra),
}
total_pop     = int(zone_df['Population_2025'].sum())
total_at_risk = zone_summary['red']['population'] + zone_summary['orange']['population']
infra_icons   = {'Healthcare': 'Hospital', 'Shelter': 'Shelter', 'Police': 'Police Station',
                 'Fire_Station': 'Fire Station', 'Waterways': 'Waterway'}

# ============================================================
# ALERT CONFIRMATION SECTION (Prominent - disappears after confirm)
# ============================================================
# Initialize alert_sent flag
if 'alert_sent_flag' not in st.session_state:
    st.session_state['alert_sent_flag'] = False

# Reset flag when risk level drops below HIGH
if alert_decision != 1:
    st.session_state['alert_sent_flag'] = False

# Show alert section ONLY if HIGH risk AND alert NOT yet sent
if alert_decision == 1 and not st.session_state['alert_sent_flag']:
    st.markdown(
        "<div style='background:#FEE2E2;border:3px solid #DC2626;border-radius:10px;"
        "padding:20px;margin:10px 0;box-shadow:0 4px 12px rgba(220,38,38,0.3)'>"
        "<h3 style='color:#DC2626;margin:0 0 10px 0'>🚨 FLOOD ALERT READY TO SEND</h3>"
        "<p style='color:#7F1D1D;margin:0;font-size:1.05em'>"
        "High risk detected. Confirm to send comprehensive emergency report to authorities."
        "</p></div>",
        unsafe_allow_html=True
    )
    
    alert_col1, alert_col2, alert_col3 = st.columns([2, 1, 1])
    
    with alert_col1:
        st.markdown(f"""
        **Alert Summary Preview:**
        - Risk Level: **{risk_level}** ({risk_prob:.1%} probability)
        - Red Zone Population: **{zone_summary['red']['population']:,}** people
        - Wards Affected: **{zone_summary['red']['ward_names']}**
        - Critical Assets at Risk: **{len(red_infra)}** facilities
        - Recommended Action: **Immediate evacuation & response**
        """)
    
    with alert_col2:
        if st.button("✅ CONFIRM & SEND", type="primary", use_container_width=True, key="confirm_alert_btn"):
            # Build comprehensive alert message
            infra_details = []
            for infra_type, count in infra_summary['red'].items():
                infra_details.append(f"  • {infra_icons.get(infra_type, infra_type)}: {count}")
            
            msg_text = f"""
🚨 RAINGUARD FLOOD ALERT - {risk_level} RISK
═══════════════════════════════════════════

📊 SITUATION OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Risk Level: {risk_level}
• Flood Probability: {risk_prob:.1%}
• Alert Status: IMMEDIATE ACTION REQUIRED

📍 AFFECTED AREAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RED ZONE (High Risk):
  • Wards: {zone_summary['red']['ward_names']}
  • Population at Risk: {zone_summary['red']['population']:,} people
  • Number of Wards: {zone_summary['red']['wards']}

ORANGE ZONE (Medium Risk):
  • Wards: {zone_summary['orange']['ward_names']}
  • Population: {zone_summary['orange']['population']:,} people
  • Number of Wards: {zone_summary['orange']['wards']}

🏥 CRITICAL INFRASTRUCTURE STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Assets in RED ZONE ({len(red_infra)} facilities):
{chr(10).join(infra_details) if infra_details else '  • None'}

Assets in ORANGE ZONE ({len(orange_infra)} facilities):
  • Prepare for potential impact

📈 WEATHER CONDITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Rainfall Intensity: {rainfall:.1f} mm/hr
• Soil Saturation: {soil_moisture:.1%}
• Humidity: {humidity:.0f}%
• Elevation Risk Factor: {elevation_risk:.2f}

⚠️ RECOMMENDED ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. IMMEDIATE EVACUATION of RED zone wards
2. Alert emergency services in affected areas
3. Activate shelter facilities
4. Deploy rescue teams to vulnerable locations
5. Issue public warnings through all channels
6. Monitor situation continuously

📱 POPULATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Mumbai Population: {total_pop:,}
• Population at Risk (RED + ORANGE): {total_at_risk:,}
• Percentage at Risk: {(total_at_risk/total_pop*100):.1f}%

═══════════════════════════════════════════
RainGuard AI - Flood Early Warning System
Generated by ANN Model + RL Alert Decision System
"""
            
            # Send alert silently (no UI messages)
            alert_sent_success = False
            
            if notify_enabled and notify_method == "Telegram" and tg_token and tg_chat_id:
                ok, msg = send_telegram_alert(tg_token, tg_chat_id, msg_text)
                if ok:
                    alert_sent_success = True
                    st.session_state.alert_history.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'risk_level': risk_level,
                        'risk_prob': f"{risk_prob:.1%}",
                        'method': 'Telegram',
                        'status': 'Sent',
                        'population_at_risk': total_at_risk
                    })
                    
            elif notify_enabled and notify_method == "Email (SMTP)" and from_em and em_pass and to_em:
                ok, msg = send_email_alert(to_em, f"🚨 RAINGUARD FLOOD ALERT - {risk_level} RISK", 
                                          msg_text, smtp_srv, smtp_port, from_em, em_pass)
                if ok:
                    alert_sent_success = True
                    st.session_state.alert_history.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'risk_level': risk_level,
                        'risk_prob': f"{risk_prob:.1%}",
                        'method': 'Email',
                        'status': 'Sent',
                        'population_at_risk': total_at_risk
                    })
            
            # Mark alert as sent (this will hide the section on rerun)
            if alert_sent_success or not notify_enabled:
                st.session_state['alert_sent_flag'] = True
                st.rerun()  # Rerun to hide the alert section immediately
    
    with alert_col3:
        if st.button("❌ CANCEL", use_container_width=True, key="cancel_alert_btn"):
            st.session_state['alert_sent_flag'] = True  # Hide alert section
            st.rerun()  # Rerun to hide immediately

st.markdown("---")

# ============================================================
# MAIN TABS  (7 tabs total)
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📍 Population Zones",
    "🏥 Infrastructure",
    "🗺️ Combined Map",
    "🌐 3D Zone Map",
    "📊 Visual Analytics",
    "🛰️ Satellite CV",
    "🤖 AI Situation Report",
])

# ============================================================
# TAB 1 — POPULATION ZONE ANALYSIS  (unchanged from document)
# ============================================================
with tab1:
    st.subheader("Population Zone Analysis — Ward-wise Risk Classification")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Mumbai Population", f"{total_pop:,}")
    c2.metric("RED Zone Population",    f"{zone_summary['red']['population']:,}",    f"{zone_summary['red']['wards']} wards")
    c3.metric("ORANGE Zone Population", f"{zone_summary['orange']['population']:,}", f"{zone_summary['orange']['wards']} wards")
    c4.metric("GREEN Zone Population",  f"{zone_summary['green']['population']:,}",  f"{zone_summary['green']['wards']} wards")
    st.markdown("---")
    col_map, col_summary = st.columns([3, 2])
    with col_map:
        st.markdown(f"#### Ward Map — {risk_level} Risk Scenario")
        st.caption("Each circle = one ward. Size = population. Color = flood risk zone.")
        ward_layer = pdk.Layer("ScatterplotLayer", data=ward_map,
            get_position="[lon, lat]", get_color="color",
            get_radius="Population_2025 / 20", radius_scale=1,
            radius_min_pixels=12, radius_max_pixels=60, pickable=True)
        st.pydeck_chart(pdk.Deck(layers=[ward_layer],
            initial_view_state=pdk.ViewState(latitude=19.08, longitude=72.88, zoom=10, pitch=15),
            tooltip={"html": "<b>Ward {Ward}</b><br/>Zone: {Zone}<br/>Population: {Population_2025}<br/>Category: {Ward_Category}<br/>Risk Score: {risk_score}",
                     "style": {"backgroundColor": "rgba(0,0,0,0.8)", "color": "white"}},
            map_style=MAP_STYLE))
        l1, l2, l3 = st.columns(3)
        l1.markdown("🔴 **RED** — High Risk (>=75%)")
        l2.markdown("🟠 **ORANGE** — Medium Risk (50-75%)")
        l3.markdown("🟢 **GREEN** — Low Risk (<50%)")
    with col_summary:
        st.markdown("#### Ward-wise Summary Table")
        def style_zone(val):
            return {"RED": "background-color:#FFE5E5;color:#C00000;font-weight:bold",
                    "ORANGE": "background-color:#FFF3E0;color:#E65100;font-weight:bold",
                    "GREEN":  "background-color:#E8F5E9;color:#1B5E20;font-weight:bold"}.get(val, "")
        display_df = zone_df[['Ward','Ward_Category','Population_2025','Zone']].copy()
        display_df.columns = ['Ward','Category','Population (2025)','Risk Zone']
        display_df['Population (2025)'] = display_df['Population (2025)'].apply(lambda x: f"{x:,}")
        st.dataframe(display_df.style.applymap(style_zone, subset=['Risk Zone']), height=450, use_container_width=True)
    st.markdown("---")
    st.markdown("#### Zone Breakdown by Ward")
    zc1, zc2, zc3 = st.columns(3)
    with zc1:
        st.markdown("<div style='background:#FFE5E5;border-left:6px solid #C00000;padding:12px;border-radius:8px'><b style='color:#C00000'>🔴 HIGH RISK ZONE</b></div>", unsafe_allow_html=True)
        st.write("")
        if not red_wards.empty:
            for _, r in red_wards.iterrows():
                st.markdown(f"**Ward {r['Ward']}** — {r['Ward_Category']}")
                st.write(f"Population: {r['Population_2025']:,}")
                st.write(f"Area: {r['Area_in_Sq_km']} km²")
                st.markdown("---")
        else:
            st.success("No wards in HIGH RISK zone.")
    with zc2:
        st.markdown("<div style='background:#FFF3E0;border-left:6px solid #E65100;padding:12px;border-radius:8px'><b style='color:#E65100'>🟠 MEDIUM RISK ZONE</b></div>", unsafe_allow_html=True)
        st.write("")
        if not orange_wards.empty:
            for _, r in orange_wards.iterrows():
                st.markdown(f"**Ward {r['Ward']}** — {r['Ward_Category']}")
                st.write(f"Population: {r['Population_2025']:,}")
                st.write(f"Area: {r['Area_in_Sq_km']} km²")
                st.markdown("---")
        else:
            st.success("No wards in MEDIUM RISK zone.")
    with zc3:
        st.markdown("<div style='background:#E8F5E9;border-left:6px solid #1B5E20;padding:12px;border-radius:8px'><b style='color:#1B5E20'>🟢 LOW RISK ZONE</b></div>", unsafe_allow_html=True)
        st.write("")
        if not green_wards.empty:
            for _, r in green_wards.iterrows():
                st.markdown(f"**Ward {r['Ward']}** — {r['Ward_Category']}")
                st.write(f"Population: {r['Population_2025']:,}")
                st.write(f"Area: {r['Area_in_Sq_km']} km²")
                st.markdown("---")
        else:
            st.info("All wards in higher risk zones.")

# ============================================================
# TAB 2 — CRITICAL INFRASTRUCTURE  (unchanged from document)
# ============================================================
with tab2:
    st.subheader("Critical Infrastructure Risk — Zone-wise Classification")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Total Infrastructure Assets", len(infra_df))
    i2.metric("RED Zone Assets",    len(red_infra))
    i3.metric("ORANGE Zone Assets", len(orange_infra))
    i4.metric("GREEN Zone Assets",  len(green_infra))
    st.markdown("---")
    col_imap, col_isummary = st.columns([3, 2])
    with col_imap:
        st.markdown("#### Infrastructure Risk Map")
        st.caption("Each point = one critical asset. Color = flood risk zone.")
        infra_map_df = infra_df[['lat','lon','infrastructure_type','Zone','color','area_m2']].dropna(subset=['lat','lon']).copy()
        infra_map_df['color'] = infra_map_df['color'].tolist()
        infra_layer = pdk.Layer("ScatterplotLayer", data=infra_map_df,
            get_position="[lon, lat]", get_color="color", get_radius=200,
            radius_min_pixels=8, radius_max_pixels=25, pickable=True)
        st.pydeck_chart(pdk.Deck(layers=[infra_layer],
            initial_view_state=pdk.ViewState(latitude=18.97, longitude=72.82, zoom=11, pitch=20),
            tooltip={"html": "<b>{infrastructure_type}</b><br/>Zone: {Zone}<br/>Area: {area_m2:.0f} m²",
                     "style": {"backgroundColor": "rgba(0,0,0,0.8)", "color": "white"}},
            map_style=MAP_STYLE))
        l1, l2, l3 = st.columns(3)
        l1.markdown("🔴 **RED** — Immediate support needed")
        l2.markdown("🟠 **ORANGE** — Prepare standby")
        l3.markdown("🟢 **GREEN** — Monitor")
    with col_isummary:
        st.markdown("#### Asset Risk Summary Table")
        def style_infra_zone(val):
            return {"RED": "background-color:#FFE5E5;color:#C00000;font-weight:bold",
                    "ORANGE": "background-color:#FFF3E0;color:#E65100;font-weight:bold",
                    "GREEN": "background-color:#E8F5E9;color:#1B5E20;font-weight:bold"}.get(val, "")
        infra_display = infra_df[['infrastructure_type','Zone','area_m2']].copy()
        infra_display.columns = ['Infrastructure Type','Risk Zone','Area (m²)']
        infra_display['Area (m²)'] = infra_display['Area (m²)'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(infra_display.style.applymap(style_infra_zone, subset=['Risk Zone']), height=400, use_container_width=True)
    st.markdown("---")
    st.markdown("#### Infrastructure Breakdown by Zone")
    ic1, ic2, ic3 = st.columns(3)
    def render_infra_card(df_zone, title_html):
        st.markdown(title_html, unsafe_allow_html=True)
        st.write("")
        if not df_zone.empty:
            for infra_type, cnt in df_zone['infrastructure_type'].value_counts().items():
                st.write(f"**{infra_icons.get(infra_type, infra_type)}**: {cnt} assets")
            st.write(f"**Total**: {len(df_zone)} assets")
        else:
            st.success("No assets in this zone.")
    with ic1:
        render_infra_card(red_infra,    "<div style='background:#FFE5E5;border-left:6px solid #C00000;padding:12px;border-radius:8px'><b style='color:#C00000'>🔴 HIGH RISK</b></div>")
    with ic2:
        render_infra_card(orange_infra, "<div style='background:#FFF3E0;border-left:6px solid #E65100;padding:12px;border-radius:8px'><b style='color:#E65100'>🟠 MEDIUM RISK</b></div>")
    with ic3:
        render_infra_card(green_infra,  "<div style='background:#E8F5E9;border-left:6px solid #1B5E20;padding:12px;border-radius:8px'><b style='color:#1B5E20'>🟢 LOW RISK</b></div>")
    st.markdown("---")
    if alert_decision == 1:
        st.error("SEND ALERT — Immediate Response Required")
        if st.button("CONFIRM AND SEND ALERT", type="primary"):
            msg_text = (f"FLOOD ALERT - {risk_level} RISK\nRisk: {risk_prob:.1%}\n"
                        f"Red zone: {zone_summary['red']['population']:,} people\n"
                        f"Critical assets at risk: {len(red_infra)}\nAction: Immediate evacuation")
            if notify_enabled and notify_method == "Telegram" and tg_token and tg_chat_id:
                ok, msg = send_telegram_alert(tg_token, tg_chat_id, msg_text)
                st.success(msg) if ok else st.error(msg)
            elif notify_enabled and notify_method == "Email (SMTP)" and from_em and em_pass and to_em:
                ok, msg = send_email_alert(to_em, f"FLOOD ALERT - {risk_level}", msg_text, smtp_srv, smtp_port, from_em, em_pass)
                st.success(msg) if ok else st.error(msg)
            else:
                st.warning("Enable notifications in sidebar to send alerts.")
    else:
        st.success("NO ALERT — Continue Monitoring")

# ============================================================
# TAB 3 — COMBINED ZONE MAP  (unchanged from document)
# ============================================================
with tab3:
    st.subheader("Combined Zone Map — Population + Infrastructure + Flood Zones")
    st.caption("Large circles = wards (population). Small circles = infrastructure. Shaded area = GIS low-lying flood zones.")
    combined_col, legend_col = st.columns([4, 1])
    with combined_col:
        ward_layer_combined = pdk.Layer("ScatterplotLayer", data=ward_map,
            get_position="[lon, lat]", get_color="color",
            get_radius="Population_2025 / 15", radius_scale=1,
            radius_min_pixels=15, radius_max_pixels=70, pickable=True, opacity=0.6)
        infra_map_df2 = infra_df[['lat','lon','infrastructure_type','Zone','color','area_m2']].dropna(subset=['lat','lon']).copy()
        infra_map_df2['color'] = infra_map_df2['color'].tolist()
        infra_layer_combined = pdk.Layer("ScatterplotLayer", data=infra_map_df2,
            get_position="[lon, lat]", get_color="color", get_radius=300,
            radius_min_pixels=10, radius_max_pixels=20, pickable=True, opacity=0.95)
        flood_color = ([220, 38, 38, 80] if risk_level=="HIGH" else
                       [249, 115, 22, 70] if risk_level=="MEDIUM" else [34, 197, 94, 60])
        layers = [ward_layer_combined, infra_layer_combined]
        try:
            low_zones = gpd.read_file(LOW_ZONES_PATH).to_crs(epsg=4326)
            low_zones_json = json.loads(low_zones.to_json())
            geojson_layer = pdk.Layer("GeoJsonLayer", data=low_zones_json,
                get_fill_color=flood_color, get_line_color=[255, 255, 255, 100],
                line_width_min_pixels=1, opacity=0.5, pickable=False)
            layers = [geojson_layer, ward_layer_combined, infra_layer_combined]
        except Exception:
            pass
        st.pydeck_chart(pdk.Deck(layers=layers,
            initial_view_state=pdk.ViewState(latitude=19.05, longitude=72.88, zoom=10, pitch=25),
            tooltip={"html": "<b>Ward: {Ward}</b><br/>{infrastructure_type}<br/>Zone: {Zone}<br/>Pop: {Population_2025}",
                     "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"}},
            map_style=MAP_STYLE))
    with legend_col:
        st.markdown("#### Legend")
        st.markdown("**🔴 RED ZONE**"); st.caption("High flood risk. Immediate evacuation.")
        st.markdown("**🟠 ORANGE ZONE**"); st.caption("Medium risk. Standby alert.")
        st.markdown("**🟢 GREEN ZONE**"); st.caption("Low risk. Routine monitoring.")
        st.markdown("---")
        st.caption("Large circles = wards"); st.caption("Small circles = infrastructure")
        st.caption("Shaded area = GIS flood zone")
        st.markdown("---")
        st.markdown("**Population at Risk**")
        pop_risk_pct = (total_at_risk / total_pop * 100) if total_pop else 0
        st.metric("", f"{total_at_risk:,}", f"{pop_risk_pct:.1f}% of total")
    st.markdown("---")
    st.markdown("#### Situation Snapshot")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Total Wards", 24)
    s2.metric("High Risk Wards",   zone_summary['red']['wards'])
    s3.metric("Medium Risk Wards", zone_summary['orange']['wards'])
    s4.metric("Low Risk Wards",    zone_summary['green']['wards'])
    s5.metric("Infra at HIGH Risk",len(red_infra))
    s6.metric("Population at Risk",f"{total_at_risk:,}")

# ============================================================
# TAB 4 — 3D COMBINED MAP  (NEW)
# ============================================================
with tab4:
    st.subheader("3D Combined Zone Map")
    st.caption("Extruded ward columns by population · Infrastructure dots · GIS flood overlay · Right-drag to rotate & tilt")

    t4_map_col, t4_leg_col = st.columns([4, 1])

    with t4_map_col:
        # Ward columns — height = population
        ward_3d = ward_map.copy()
        ward_3d['elevation'] = (
            (ward_3d['Population_2025'] / ward_map['Population_2025'].max()) * 8000
        ).astype(int)

        col_layer = pdk.Layer(
            "ColumnLayer",
            data=ward_3d,
            get_position="[lon, lat]",
            get_elevation="elevation",
            elevation_scale=1,
            radius=700,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
            opacity=0.88,
        )

        # Infrastructure scatter (small spheres above terrain)
        infra_3d = infra_df[['lat','lon','infrastructure_type','Zone','color','area_m2']].dropna(subset=['lat','lon']).copy()
        infra_3d['color'] = infra_3d['color'].tolist()

        infra_3d_layer = pdk.Layer(
            "ScatterplotLayer",
            data=infra_3d,
            get_position="[lon, lat]",
            get_color="color",
            get_radius=320,
            radius_min_pixels=5,
            radius_max_pixels=16,
            pickable=True,
            opacity=0.95,
        )

        # GIS flood polygon base
        flood_fill_3d = ([220, 38, 38, 70] if risk_level=="HIGH" else
                         [249, 115, 22, 60] if risk_level=="MEDIUM" else [34, 197, 94, 50])
        layers_3d = [col_layer, infra_3d_layer]
        try:
            low_z       = gpd.read_file(LOW_ZONES_PATH).to_crs(epsg=4326)
            low_z_json  = json.loads(low_z.to_json())
            geojson_3d  = pdk.Layer("GeoJsonLayer", data=low_z_json,
                get_fill_color=flood_fill_3d, get_line_color=[255,255,255,60],
                line_width_min_pixels=1, opacity=0.55, pickable=False, extruded=False)
            layers_3d = [geojson_3d, col_layer, infra_3d_layer]
        except Exception:
            pass

        st.pydeck_chart(pdk.Deck(
            layers=layers_3d,
            initial_view_state=pdk.ViewState(
                latitude=19.05, longitude=72.88,
                zoom=10, pitch=52, bearing=12),
            tooltip={
                "html": ("<b>Ward: {Ward}</b><br/>Zone: {Zone}<br/>"
                         "Population: {Population_2025}<br/>"
                         "{infrastructure_type}<br/>Risk Score: {risk_score}"),
                "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"},
            },
            map_style=MAP_STYLE,
        ))

    with t4_leg_col:
        st.markdown("#### 3D Legend")
        st.markdown("**Column height**")
        st.caption("Proportional to ward population")
        st.markdown("---")
        st.markdown("**🔴 RED** — High Risk")
        st.caption("Immediate evacuation required")
        st.markdown("**🟠 ORANGE** — Medium Risk")
        st.caption("Prepare standby response")
        st.markdown("**🟢 GREEN** — Low Risk")
        st.caption("Routine monitoring only")
        st.markdown("---")
        st.markdown("**Map Layers**")
        st.caption("▮ Tall columns = wards")
        st.caption("● Small dots = infrastructure")
        st.caption("▓ Shaded base = GIS flood zone")
        st.markdown("---")
        st.markdown("**🖱️ Controls**")
        st.caption("Left-drag → pan")
        st.caption("Right-drag → rotate & tilt")
        st.caption("Scroll → zoom")
        st.caption("Click → inspect")
        st.markdown("---")
        st.markdown("**Population at Risk**")
        pop_risk_pct4 = (total_at_risk / total_pop * 100) if total_pop else 0
        st.metric("", f"{total_at_risk:,}", f"{pop_risk_pct4:.1f}% of Mumbai")

    # Situation Snapshot (same as tab3)
    st.markdown("---")
    st.markdown("#### Situation Snapshot")
    ss1, ss2, ss3, ss4, ss5, ss6 = st.columns(6)
    ss1.metric("Total Wards",        24)
    ss2.metric("High Risk Wards",    zone_summary['red']['wards'])
    ss3.metric("Medium Risk Wards",  zone_summary['orange']['wards'])
    ss4.metric("Low Risk Wards",     zone_summary['green']['wards'])
    ss5.metric("Infra at HIGH Risk", len(red_infra))
    ss6.metric("Population at Risk", f"{total_at_risk:,}")

# ============================================================
# TAB 5 — VISUAL ANALYTICS  (NEW)
# ============================================================
with tab5:
    st.subheader("Visual Analytics — Flood Risk Intelligence")
    st.caption("Interactive charts from ward data, infrastructure risk, and historical weather — each with an AI-generated summary.")

    # ── dark card summary helper ──────────────────────────────────
    def _summary_card(icon, title, body_html):
        st.markdown(
            f"<div style='background:#0F172A;border-left:5px solid #38BDF8;"
            f"padding:14px 16px;border-radius:8px;margin-top:10px'>"
            f"<span style='font-size:1.15em'>{icon}</span>"
            f"<strong style='color:#38BDF8;margin-left:8px'>{title}</strong>"
            f"<div style='color:#CBD5E1;margin-top:6px;font-size:0.88em;line-height:1.55'>"
            f"{body_html}</div></div>",
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────
    # 1. WARD POPULATION BAR CHART
    # ─────────────────────────────────────────────────────────────
    st.markdown("### 1 · Ward Population by Risk Zone")
    fig_bar = px.bar(
        zone_df.sort_values("ward_risk", ascending=False),
        x="Ward", y="Population_2025", color="Zone",
        color_discrete_map={"RED":"#EF4444","ORANGE":"#F97316","GREEN":"#22C55E"},
        labels={"Population_2025":"Population (2025)","Ward":"Ward"},
        template="plotly_dark", text_auto=".2s",
    )
    fig_bar.update_layout(
        height=380, plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#CBD5E1", bargap=0.18, legend_title_text="Risk Zone",
    )
    fig_bar.update_traces(textfont_size=10, textangle=0,
                          textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    top_red_wards = red_wards.sort_values('Population_2025', ascending=False)['Ward'].tolist()[:3] if not red_wards.empty else []
    _summary_card("👥", "Population Exposure",
        f"<b>{zone_summary['red']['population']:,}</b> people in <b>{zone_summary['red']['wards']}</b> HIGH-risk wards "
        f"face immediate flood danger. Most exposed: <b>{', '.join(top_red_wards) if top_red_wards else 'None'}</b>. "
        f"Total at risk (RED + ORANGE): <b>{total_at_risk:,}</b> "
        f"({total_at_risk/total_pop*100:.1f}% of Mumbai).")
    st.markdown("---")

    # ─────────────────────────────────────────────────────────────
    # 2. POPULATION DENSITY vs RISK SCORE BUBBLE CHART
    # ─────────────────────────────────────────────────────────────
    st.markdown("### 2 · Population Density vs Risk Score")
    z_df = zone_df.copy()
    z_df['pop_density'] = z_df['Population_2025'] / z_df['Area_in_Sq_km']
    fig_bubble = px.scatter(
        z_df, x="pop_density", y="ward_risk",
        size="Population_2025", color="Zone", hover_name="Ward",
        color_discrete_map={"RED":"#EF4444","ORANGE":"#F97316","GREEN":"#22C55E"},
        labels={"pop_density":"Population Density (per km²)","ward_risk":"Ward Risk Score"},
        template="plotly_dark", size_max=55,
    )
    fig_bubble.update_layout(
        height=420, plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#CBD5E1",
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    dense_ward = z_df.sort_values('pop_density', ascending=False).iloc[0]
    _summary_card("🔬", "Density vs Risk Insight",
        f"Ward <b>{dense_ward['Ward']}</b> has the highest density "
        f"(<b>{dense_ward['pop_density']:,.0f}</b>/km²) — Zone: <b>{dense_ward['Zone']}</b>. "
        f"High-density wards amplify casualties because evacuation corridors congest quickly. "
        f"Bubble size = total population; large bubbles in RED zones demand the most emergency resources.")
    st.markdown("---")

    # ─────────────────────────────────────────────────────────────
    # 3. INFRASTRUCTURE RISK HEATMAP
    # ─────────────────────────────────────────────────────────────
    st.markdown("### 3 · Infrastructure Risk Heatmap")
    infra_types_list = ['Healthcare','Shelter','Police','Fire_Station','Waterways']
    zones_order = ['RED','ORANGE','GREEN']
    friendly_names = {'Healthcare':'Hospital','Shelter':'Shelter','Police':'Police Stn',
                      'Fire_Station':'Fire Stn','Waterways':'Waterway'}
    hm_data = [[len(infra_df[(infra_df['infrastructure_type']==it)&(infra_df['Zone']==z)])
                for z in zones_order] for it in infra_types_list]

    fig_heat = go.Figure(data=go.Heatmap(
        z=hm_data,
        x=zones_order,
        y=[friendly_names.get(t,t) for t in infra_types_list],
        colorscale=[[0.0,"#1E293B"],[0.01,"#22C55E"],[0.35,"#F97316"],[1.0,"#EF4444"]],
        text=hm_data, texttemplate="%{text}", textfont_size=14, showscale=True,
    ))
    fig_heat.update_layout(
        height=320, template="plotly_dark",
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#CBD5E1", xaxis_title="Risk Zone", yaxis_title="Asset Type",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    most_exp, max_cnt = "N/A", 0
    for it in infra_types_list:
        c = len(infra_df[(infra_df['infrastructure_type']==it)&(infra_df['Zone']=='RED')])
        if c > max_cnt:
            max_cnt = c; most_exp = friendly_names.get(it, it)
    _summary_card("🏥", "Infrastructure Vulnerability",
        f"<b>{len(red_infra)}</b> critical assets are in HIGH-risk zones. "
        f"Most exposed type: <b>{most_exp}</b> ({max_cnt} units). "
        f"Healthcare and shelter facilities in RED zones need immediate resource prepositioning. "
        f"Waterways in RED zones signal active inundation pathways.")
    st.markdown("---")

    # ─────────────────────────────────────────────────────────────
    # 4. HISTORICAL RAINFALL & HUMIDITY TIME SERIES
    # ─────────────────────────────────────────────────────────────
    st.markdown("### 4 · Historical Rainfall & Humidity (2016–2019)")
    try:
        core_df = pd.read_csv(BASE_DIR / "data" / "processed" / "Core_features.csv")
        core_df['date'] = pd.to_datetime(core_df['datetime'], dayfirst=True, errors='coerce')
        core_df = core_df.dropna(subset=['date']).sort_values('date')

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=core_df['date'], y=core_df['Intensity'],
            name="Rainfall (mm/hr)", mode='lines',
            line=dict(color="#38BDF8", width=1.5),
            fill='tozeroy', fillcolor='rgba(56,189,248,0.10)',
        ))
        fig_ts.add_trace(go.Scatter(
            x=core_df['date'], y=core_df['humidity'],
            name="Humidity (%)", mode='lines',
            line=dict(color="#A78BFA", width=1.2, dash='dot'),
            yaxis='y2',
        ))
        floods_hist = core_df[core_df['Flood Risk'] == 1]
        fig_ts.add_trace(go.Scatter(
            x=floods_hist['date'], y=floods_hist['Intensity'],
            name="Flood Event", mode='markers',
            marker=dict(color="#EF4444", size=12, symbol='star',
                        line=dict(color='white', width=1)),
        ))
        fig_ts.update_layout(
            height=420, template="plotly_dark",
            plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
            font_color="#CBD5E1", hovermode='x unified',
            xaxis_title="Date",
            yaxis=dict(title="Rainfall (mm/hr)", color="#38BDF8"),
            yaxis2=dict(title="Humidity (%)", overlaying='y', side='right', color="#A78BFA"),
            legend=dict(orientation='h', y=-0.18),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        peak_date = core_df.loc[core_df['Intensity'].idxmax(),'date'].strftime('%d %b %Y')
        peak_val  = core_df['Intensity'].max()
        flood_dates_str = ', '.join(floods_hist['date'].dt.strftime('%d %b %Y').tolist()) or 'None'
        _summary_card("🌧️", "Historical Rainfall Analysis",
            f"Peak rainfall: <b>{peak_val:.0f} mm/hr</b> on <b>{peak_date}</b>. "
            f"Confirmed flood events: <b>{len(floods_hist)}</b> ({flood_dates_str}). "
            f"Humidity stays above 85% during all flood events — a leading indicator. "
            f"Current reading: <b>{humidity:.0f}%</b> humidity, "
            f"<b>{rainfall:.1f} mm/hr</b> → <b>{risk_level}</b> risk.")
    except Exception as _e:
        st.warning(f"Historical data unavailable: {_e}")
    st.markdown("---")

    # ─────────────────────────────────────────────────────────────
    # 5. WARD AREA vs POPULATION COLOUR BAR
    # ─────────────────────────────────────────────────────────────
    st.markdown("### 5 · Ward Area vs Population")
    fig_area = px.bar(
        zone_df.sort_values('Population_2025', ascending=False),
        x='Ward', y='Population_2025', color='Area_in_Sq_km',
        color_continuous_scale=['#22C55E','#F97316','#EF4444'],
        labels={'Population_2025':'Population','Area_in_Sq_km':'Area (km²)'},
        template='plotly_dark', text_auto='.2s',
    )
    fig_area.update_layout(
        height=380, plot_bgcolor="#0F172A", paper_bgcolor="#0F172A", font_color="#CBD5E1",
    )
    st.plotly_chart(fig_area, use_container_width=True)

    largest  = zone_df.sort_values('Area_in_Sq_km', ascending=False).iloc[0]
    smallest = zone_df.sort_values('Area_in_Sq_km').iloc[0]
    _summary_card("📊", "Area vs Population Analysis",
        f"Largest ward: <b>{largest['Ward']}</b> ({largest['Area_in_Sq_km']} km²). "
        f"Smallest: <b>{smallest['Ward']}</b> ({smallest['Area_in_Sq_km']} km²). "
        f"Red-tinted bars = large area wards where flood water spreads farther, "
        f"demanding multi-route evacuation and more rescue resources.")
    st.markdown("---")

    # ─────────────────────────────────────────────────────────────
    # 6. RISK ZONE DONUT + SUNBURST
    # ─────────────────────────────────────────────────────────────
    st.markdown("### 6 · Risk Zone Distribution")
    donut_col, sunburst_col = st.columns(2)
    zone_counts = zone_df['Zone'].value_counts().reset_index()
    zone_counts.columns = ['Zone','Count']

    with donut_col:
        st.markdown("**Ward Count by Zone**")
        fig_donut = px.pie(
            zone_counts, values='Count', names='Zone', hole=0.52,
            color='Zone',
            color_discrete_map={"RED":"#EF4444","ORANGE":"#F97316","GREEN":"#22C55E"},
            template='plotly_dark',
        )
        fig_donut.update_layout(
            height=320, paper_bgcolor="#0F172A", font_color="#CBD5E1",
            margin=dict(t=10,b=10,l=10,r=10),
        )
        fig_donut.update_traces(textinfo='label+percent', textfont_size=12)
        st.plotly_chart(fig_donut, use_container_width=True)

    with sunburst_col:
        st.markdown("**Population by Category & Zone**")
        fig_sun = px.sunburst(
            zone_df[['Ward_Category','Zone','Population_2025']],
            path=['Ward_Category','Zone'], values='Population_2025',
            color='Zone',
            color_discrete_map={"RED":"#EF4444","ORANGE":"#F97316","GREEN":"#22C55E"},
            template='plotly_dark',
        )
        fig_sun.update_layout(
            height=320, paper_bgcolor="#0F172A", font_color="#CBD5E1",
            margin=dict(t=10,b=10,l=10,r=10),
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    zone_pop = {z: int(zone_df[zone_df['Zone']==z]['Population_2025'].sum()) for z in ['RED','ORANGE','GREEN']}
    dom_zone = max(zone_pop, key=zone_pop.get)
    _summary_card("🍩", "Zone Distribution Summary",
        f"Under current risk (<b>{risk_level}</b>, {risk_prob:.1%} probability): "
        f"<b>{zone_summary['red']['wards']}</b> RED wards, "
        f"<b>{zone_summary['orange']['wards']}</b> ORANGE, "
        f"<b>{zone_summary['green']['wards']}</b> GREEN. "
        f"By population the dominant zone is <b>{dom_zone}</b>. "
        f"The sunburst shows ward category breakdown — Eastern Wards carry the most "
        f"population at risk due to lower elevation and creek proximity.")

# ============================================================
# ============================================================
# TAB 6 — SATELLITE CV (Simplified)
# ============================================================

with tab6:
    st.subheader("Satellite Imagery Analysis — Before & After Flood Comparison")
    st.caption(
        "When the ANN model or Real-Time API predicts HIGH flood risk, "
        "the system instantly displays before and after flood images of the area "
        "with RGB-based flood highlighting to enhance understanding."
    )

    @st.cache_data(show_spinner=False, ttl=60)  # Cache for 60 seconds only
    def _run_cv_cached(_a, _b, _rl):
        import sys as _sys, importlib
        _sys.path.insert(0, str(BASE_DIR))
        try:
            import cv.water_detection as _cv_mod
            importlib.reload(_cv_mod)
            return _cv_mod.run_cv_validation(
                after_path  = str(_a) if Path(_a).exists() else None,
                before_path = str(_b) if Path(_b).exists() else None,
                risk_level  = _rl,
            )
        except Exception as _e:
            return {"error": str(_e), "_demo_mode": True,
                    "water_change_pct": 0, "flood_area_km2": 0,
                    "flood_pixel_count": 0, "flood_confirmed": False,
                    "highlight_applied": False,
                    "severity": {"level":"N/A","emoji":"","action":""},
                    "change_severity": {"level":"N/A","emoji":"","action":""},
                    "validation_summary": f"CV module error: {_e}",
                    "before": {"water_coverage_pct": 0, "flood_detected": False},
                    "after":  {"water_coverage_pct": 0, "flood_detected": False}}

    # Run CV analysis - regenerates images on every HIGH risk
    cv_res  = _run_cv_cached(CV_AFTER_PATH, CV_BEFORE_PATH, risk_level)
    cv_err  = cv_res.get("error")

    if cv_err:
        st.error("CV module error: " + str(cv_err))
    
    # Force regeneration if images don't exist but should (HIGH risk)
    if risk_level == "HIGH" and not CV_HL_PATH.exists():
        # Clear cache and try again
        _run_cv_cached.clear()
        cv_res = _run_cv_cached(CV_AFTER_PATH, CV_BEFORE_PATH, risk_level)

    st.markdown("---")
    
    # ============================================================
    # MAIN IMAGE DISPLAY — CONDITIONAL ON RISK LEVEL
    # ============================================================
    
    
    if risk_level == "HIGH":
        # HIGH RISK: Show both before and after images side-by-side with flood highlighting
        st.markdown("### 🔴 HIGH RISK DETECTED — Before & After Flood Comparison")
        st.markdown(
            "<div style='background:#FFE5E5;border-left:5px solid #C00000;"
            "padding:12px;border-radius:6px;margin-bottom:16px'>"
            "<b style='color:#C00000;font-size:1.1em'>⚠️ CRITICAL FLOOD ALERT</b><br>"
            "<span style='color:#7F1D1D'>The ANN model has detected HIGH flood risk. "
            "Satellite imagery shows before and after conditions with RGB-based flood water highlighting "
            "to help you understand the flood extent instantly.</span>"
            "</div>", unsafe_allow_html=True)
        
        img_l, img_r = st.columns(2)
        
        with img_l:
            st.markdown("#### 📅 Before Flood — Pre-event Conditions")
            if CV_ENH_PATH.exists():
                st.image(str(CV_ENH_PATH),
                         caption="Before flood: Sentinel-2 RGB image showing normal conditions",
                         use_column_width=True)
            elif CV_BEFORE_PATH.exists():
                st.image(str(CV_BEFORE_PATH), 
                         caption="Before flood: Normal area conditions", 
                         use_column_width=True)
            else:
                st.warning("Before-flood image not available. Place image in cv/sample_images/ directory.")
        
        with img_r:
            st.markdown("#### 🌊 After Flood — CYAN Highlights Show Flood Water")
            st.markdown(
                "<div style='background:#CFFAFE;border-left:5px solid #0891B2;"
                "padding:10px;border-radius:6px;margin-bottom:12px'>"
                "<b style='color:#0891B2'>RGB Flood Highlighting Applied</b><br>"
                "<span style='color:#164E63;font-size:0.9em'>CYAN pixels = detected flood water using NDWI analysis</span>"
                "</div>", unsafe_allow_html=True)
            
            if CV_HL_PATH.exists():
                st.image(str(CV_HL_PATH),
                         caption="After flood: CYAN = new flood water. Yellow glow = flood boundary. Other colors = real RGB.",
                         use_column_width=True)
            elif CV_AFTER_PATH.exists():
                st.image(str(CV_AFTER_PATH),
                         caption="After flood image (processing highlights...)", 
                         use_column_width=True)
            else:
                st.warning("After-flood image not available. Run scripts/after_image.py to generate.")
        
        # Optional: Show side-by-side panel if available
        if CV_PANEL_PATH.exists():
            st.markdown("---")
            st.markdown("#### 🔄 Side-by-Side Comparison Panel")
            st.image(str(CV_PANEL_PATH),
                     caption="Left: Before flood | Right: After flood with CYAN flood highlights",
                     use_column_width=True)
    
    elif risk_level in ("MEDIUM", "LOW"):
        # MEDIUM/LOW RISK: Show only before image
        st.markdown(f"### 🟢 {risk_level} RISK — Normal Area Conditions")
        
        if risk_level == "MEDIUM":
            st.markdown(
                "<div style='background:#FFF3E0;border-left:5px solid #E65100;"
                "padding:12px;border-radius:6px;margin-bottom:16px'>"
                "<b style='color:#E65100;font-size:1.1em'>🟠 MEDIUM RISK</b><br>"
                "<span style='color:#7C2D12'>Current risk level is MEDIUM. "
                "The system shows normal area conditions for monitoring. "
                "Before/after comparison will activate automatically if risk escalates to HIGH.</span>"
                "</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='background:#E8F5E9;border-left:5px solid #1B5E20;"
                "padding:12px;border-radius:6px;margin-bottom:16px'>"
                "<b style='color:#1B5E20;font-size:1.1em'>🟢 LOW RISK</b><br>"
                "<span style='color:#1B5E20'>Current risk level is LOW. "
                "The system shows normal area conditions for routine monitoring. "
                "Advanced satellite analysis will activate if risk increases.</span>"
                "</div>", unsafe_allow_html=True)
        
        # Show before image centered
        center_col1, center_img, center_col2 = st.columns([1, 3, 1])
        with center_img:
            st.markdown("#### 📍 Current Area Status — Normal Conditions")
            if CV_ENH_PATH.exists():
                st.image(str(CV_ENH_PATH),
                         caption="Satellite view: Normal area conditions (Sentinel-2 RGB)",
                         use_column_width=True)
            elif CV_BEFORE_PATH.exists():
                st.image(str(CV_BEFORE_PATH), 
                         caption="Satellite view: Normal area conditions", 
                         use_column_width=True)
            else:
                st.info("Satellite image not available. Place image in cv/sample_images/ directory.")
        
        st.markdown("---")
        st.info("💡 **Note:** Before/after flood comparison with RGB highlighting will automatically "
                "activate when the ANN model or Real-Time API detects HIGH flood risk.")


# ============================================================
# TAB 7 — AI SITUATION REPORT  (unchanged from document)
# ============================================================
with tab7:
    st.subheader("AI-Generated Situation Report")
    st.caption("Gemini AI reads ALL zone data — flood probability, ward populations, infrastructure status — and writes a professional emergency report.")

    st.markdown("#### Automated Summary")
    auto_col1, auto_col2 = st.columns(2)
    flood_bg     = "#FCA5A5" if risk_level=="HIGH" else "#FCD34D" if risk_level=="MEDIUM" else "#86EFAC"
    flood_border = "#C00000" if risk_level=="HIGH" else "#E65100" if risk_level=="MEDIUM" else "#1B5E20"

    with auto_col1:
        st.markdown(
            f"<div style='background:{flood_bg};color:#0F172A;border-left:6px solid {flood_border};"
            f"padding:16px;border-radius:8px'>"
            f"<h4 style='color:{flood_border}'>Flood Risk: {risk_level}</h4>"
            f"<p>Probability: <strong>{risk_prob:.1%}</strong></p>"
            f"<p>Rainfall: <strong>{rainfall:.1f} mm/hr</strong></p>"
            f"<p>Soil Saturation: <strong>{soil_moisture:.1%}</strong></p>"
            f"<p>Alert: <strong>{'SEND ALERT' if alert_decision else 'MONITOR'}</strong></p>"
            f"</div>", unsafe_allow_html=True)

    with auto_col2:
        st.markdown(
            f"<div style='background:#93C5FD;color:#0F172A;border-left:6px solid #1D4ED8;"
            f"padding:16px;border-radius:8px'>"
            f"<h4 style='color:#1565C0'>Population Impact</h4>"
            f"<p>🔴 <strong>{zone_summary['red']['wards']} wards</strong> HIGH — "
            f"<strong>{zone_summary['red']['population']:,}</strong> people</p>"
            f"<p>🟠 <strong>{zone_summary['orange']['wards']} wards</strong> MEDIUM — "
            f"<strong>{zone_summary['orange']['population']:,}</strong> people</p>"
            f"<p>🟢 <strong>{zone_summary['green']['wards']} wards</strong> LOW — "
            f"<strong>{zone_summary['green']['population']:,}</strong> people</p>"
            f"<p>Total at risk: <strong>{total_at_risk:,}</strong> ({total_at_risk/total_pop*100:.1f}%)</p>"
            f"</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='background:#C4B5FD;color:#0F172A;border-left:6px solid #6A1B9A;"
        f"padding:16px;border-radius:8px;margin-top:12px'>"
        f"<h4 style='color:#6A1B9A'>Critical Infrastructure Status</h4>"
        f"<p>🔴 <strong>{len(red_infra)}</strong> HIGH: "
        f"{', '.join(f'{v} {infra_icons.get(k,k)}' for k,v in infra_counts(red_infra).items()) or 'None'}</p>"
        f"<p>🟠 <strong>{len(orange_infra)}</strong> MEDIUM: "
        f"{', '.join(f'{v} {infra_icons.get(k,k)}' for k,v in infra_counts(orange_infra).items()) or 'None'}</p>"
        f"<p>🟢 <strong>{len(green_infra)}</strong> LOW: "
        f"{', '.join(f'{v} {infra_icons.get(k,k)}' for k,v in infra_counts(green_infra).items()) or 'None'}</p>"
        f"</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Gemini AI Comprehensive Report")
    if not enable_gemini:
        st.info("Enable Gemini AI in the sidebar and enter your API key to generate a full situation report.")
    else:
        if st.button("Generate AI Situation Report", type="primary"):
            with st.spinner("Gemini is analysing all zone data and writing the report..."):
                summary_text, err = generate_gemini_full_summary(
                    risk_level, risk_prob, rainfall, soil_moisture, elevation_risk,
                    alert_decision, zone_summary, infra_summary,
                    st.session_state.get('gemini_api_key', ''))
                if summary_text:
                    st.session_state['ai_full_summary'] = summary_text
                else:
                    st.error(err)
        if st.session_state.get('ai_full_summary'):
            st.markdown(st.session_state['ai_full_summary'])
            st.download_button("Download Report", st.session_state['ai_full_summary'],
                file_name=f"RainGuardAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain")

    if st.session_state.alert_history:
        st.markdown("---")
        st.subheader("Alert History")
        st.dataframe(pd.DataFrame(st.session_state.alert_history), use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    f"RainGuardAI  |  Risk: {risk_level} ({risk_prob:.1%})  |  "
    f"Population at risk: {total_at_risk:,}  |  "
    f"Updated: {datetime.now().strftime('%H:%M:%S')}  |  "
    "ML + GIS + GenAI + RL + CV"
)