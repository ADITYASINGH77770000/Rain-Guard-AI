<div align="center">

<img src="Project/dashboard/assets/RainGuard.png" alt="RainGuard AI" width="700"/>

<br/>

# 🌊 RainGuard AI — Flood Early Warning System

### *AI-Powered Flood Risk Intelligence for Mumbai, Maharashtra*

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ANN%20Model-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-CV%20Analysis-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![GeoPandas](https://img.shields.io/badge/GeoPandas-GIS%20Mapping-139C5A?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Gemini%20AI-GenAI%20Reports-4285F4?style=for-the-badge&logo=google&logoColor=white)

<br/>

> **RainGuard AI** is a full-stack flood early warning system that combines Artificial Neural Networks, Reinforcement Learning, GIS spatial analysis, Computer Vision on Sentinel-2 satellite imagery, and Generative AI — all in a single Streamlit dashboard — to predict flood risk and drive emergency response for Mumbai's 24 wards.

<br/>

![Demo GIF](demo.gif)

</div>

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🏗️ System Architecture](#️-system-architecture)
- [✨ Features](#-features)
- [📂 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [⚙️ Configuration](#️-configuration)
- [🧠 ML Models](#-ml-models)
- [🛰️ Satellite CV Module](#️-satellite-cv-module)
- [💬 GenAI RAG Chatbot](#-genai-rag-chatbot)
- [📡 Alert System](#-alert-system)
- [📊 Dashboard Tabs](#-dashboard-tabs)
- [🗂️ Data Sources](#️-data-sources)
- [📦 Requirements](#-requirements)
- [🤝 Contributing](#-contributing)

---

## 🎯 Project Overview

Mumbai faces recurring, devastating floods during monsoon season. Traditional early-warning systems lack real-time AI analysis, granular ward-level mapping, and automated emergency dispatch. **RainGuard AI** addresses this with a multi-phase AI pipeline:

| Phase | Component | Purpose |
|-------|-----------|---------|
| **Phase 1** | ANN Flood Model + Live Weather | Predict flood probability in real time |
| **Phase 2** | GIS Zone Analysis | Map risk across wards and infrastructure |
| **Phase 3** | RL Alert Policy | Smart alert dispatch decisions |
| **CV Module** | Satellite Image Analysis | Visual ground-truth via Sentinel-2 |
| **GenAI Module** | RAG Chatbot + Gemini Reports | Conversational support + executive briefings |

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          RainGuard AI Dashboard          │
                        │           (Streamlit Web App)            │
                        └────────────────┬────────────────────────┘
                                         │
           ┌─────────────────────────────┼─────────────────────────────┐
           │                             │                             │
    ┌──────▼──────┐             ┌────────▼────────┐           ┌───────▼───────┐
    │  Live Data   │             │   ANN Model     │           │  RL Alert     │
    │  Ingestion   │             │  (Keras .keras) │           │  Policy       │
    │  Open-Meteo  │             │  Flood Risk     │           │  (sklearn pkl)│
    │  WeatherAPI  │─────────────▶  Probability    │──────────▶  Dispatch      │
    │  Manual      │             │  LOW/MED/HIGH   │           │  Decision     │
    └─────────────┘             └────────┬────────┘           └───────┬───────┘
                                         │                             │
           ┌─────────────────────────────┼─────────────────────────────┐
           │                             │                             │
    ┌──────▼──────┐             ┌────────▼────────┐           ┌───────▼───────┐
    │  Population  │             │ Infrastructure  │           │  Satellite    │
    │  Zone Maps   │             │  Risk Mapping   │           │  CV Module    │
    │  24 Wards    │             │  Hospital /     │           │  Sentinel-2   │
    │  RED/ORANGE  │             │  Shelter /      │           │  NDWI Flood   │
    │  GREEN       │             │  Police / Fire  │           │  Highlighting │
    └─────────────┘             └────────┬────────┘           └───────────────┘
                                         │
                                ┌────────▼────────┐
                                │  GenAI RAG      │
                                │  Chatbot        │
                                │  FAISS Index    │
                                │  NDMA Guidelines│
                                │  Gemini Report  │
                                └─────────────────┘
```

---

## ✨ Features

### 🧠 1. ANN Flood Prediction Engine
- **Keras Sequential Neural Network** trained on Mumbai weather data (2016–2019)
- Four input features: `Rainfall Intensity`, `Soil Moisture Index`, `Elevation Risk`, `Humidity`
- Physics-based binary labeling with class-weight balancing for rare flood events
- Outputs a continuous flood probability score → mapped to **LOW / MEDIUM / HIGH**
- Automatic fallback to formula-based scoring if model files are missing
- Scaler (StandardScaler) saved alongside model for consistent inference

### 🤖 2. Reinforcement Learning Alert Policy
- **Logistic Regression policy** trained on historical alert-reward data
- `HIGH` risk → forced alert (safety-critical override of RL model)
- `MEDIUM` risk → RL decides: **SEND ALERT** or **CONTINUE MONITORING**
- `LOW` risk → suppress alerts, only routine monitoring
- Full audit trail maintained in session-state alert history

### 🌦️ 3. Real-Time Weather Integration
- **OpenStreetMap / Open-Meteo** — free, no API key required
- **WeatherAPI** — detailed current conditions with API key
- **QuoaMean API** — alternative weather provider support
- **Reverse Geocoding** via Nominatim for human-readable location display
- Elevation lookup via Open-Elevation API for terrain risk estimation
- Soil moisture estimated from humidity + rainfall + temperature formula

### 👥 4. Ward-Level Population Zone Analysis
- Classifies all **24 Mumbai wards** into RED / ORANGE / GREEN risk zones
- Population density weighting: high-density wards amplify flood impact score
- Interactive PyDeck scatter map with proportional circle sizes
- Color-coded ward summary table with population counts and area data
- Real-time zone breakdown cards for RED, ORANGE, and GREEN wards

### 🏥 5. Critical Infrastructure Risk Mapping
- Maps **Hospitals, Shelters, Police Stations, Fire Stations, Waterways**
- Criticality-weighted risk multipliers per infrastructure type:
  - Healthcare: 1.10× | Shelter: 1.05× | Police/Fire: 1.00× | Waterways: 0.70×
- GeoPackage (`.gpkg`) data loaded via GeoPandas for spatial accuracy
- Interactive map with zone-colored asset points and area metadata

### 🗺️ 6. Combined 2D Operations Map
- Unified operational view: **wards + infrastructure + GIS flood polygons**
- GeoJSON low-lying zones overlaid with risk-colored shading
- Designed for rapid command decision-making during active flood events

### 📐 7. 3D Column Zone Map
- **PyDeck ColumnLayer** — ward columns extruded by population height
- Right-drag to rotate; scroll to zoom; click for ward tooltip
- Infrastructure scatter points layered above the terrain
- GIS flood polygon base with translucent risk overlay

### 📊 8. Visual Analytics Dashboard
Six interactive Plotly charts:
1. Ward population bar chart by risk zone
2. Population density vs. risk score bubble chart
3. Infrastructure risk heatmap (asset type × zone)
4. Historical rainfall & humidity time series (2016–2019) with flood event markers
5. Ward area vs. population color bar
6. Risk zone distribution donut + sunburst charts

Each chart includes an **AI-generated summary card** explaining key insights.

### 🛰️ 9. Satellite Computer Vision Module
- **Sentinel-2 true-colour** before/after flood comparison
- NDWI pixel classification: `(G − R) / (G + R)` per pixel
- Flood water pixels painted **CYAN** — real RGB preserved for all other pixels
- Yellow glow drawn at flood boundary for visual emphasis
- Activates automatically on **HIGH** risk; shows normal conditions for LOW/MEDIUM
- Outputs: highlighted image, before/after panel, flood mask, water coverage %

### 💬 10. GenAI RAG Chatbot
- **FAISS vector index** built from NDMA flood management guidelines PDF
- **Sentence-Transformers** for semantic embedding of query and documents
- **LangChain** retrieval chain for context-augmented responses
- Streamlit chat interface with conversation history

### 📋 11. Gemini AI Situation Reports
- **Gemini 2.5 Flash** generates professional emergency situation reports
- Structured sections: Situation Overview, Population at Risk, Infrastructure Status, Immediate Actions, 6-Hour Outlook, NDMA Compliance Notes
- Downloadable `.txt` report with timestamp
- Integrates all zone data, ward names, infrastructure counts in prompt context

### 📡 12. Multi-Channel Alert Dispatch
- **Telegram Bot** — sends formatted alert with markdown tables to chat/channel
- **SMTP Email** — dispatches via Gmail or any SMTP server
- Comprehensive alert message includes: risk level, affected wards, population at risk, infrastructure summary, weather conditions, recommended actions
- Confirmation UX: preview before send, cancel option, one-time dispatch per event

---

## 📂 Project Structure

```
Rain-Guard-AI/
├── Project/
│   ├── dashboard/                  # Streamlit web application
│   │   ├── app.py                  # Main dashboard entry point
│   │   ├── introduction_page.py    # Feature overview with Lottie animations
│   │   ├── chatbot_page.py         # RAG chatbot interface
│   │   ├── auth_page.py            # User authentication
│   │   ├── audio_player.py         # Background music controller
│   │   └── assets/
│   │       ├── RainGuard.png       # Project logo
│   │       ├── Intestellar.mp3     # Background ambient audio
│   │       └── Animations/         # Lottie JSON animation files
│   │
│   ├── model/                      # ANN flood prediction model
│   │   ├── train_ann.py            # Training script (Keras + sklearn)
│   │   ├── ann_flood_model.keras   # Trained model weights
│   │   └── scaler.pkl              # StandardScaler for inference
│   │
│   ├── Phase 3/
│   │   └── rl/                     # Reinforcement Learning alert policy
│   │       ├── train_alert_policy.py  # Logistic Regression RL policy training
│   │       ├── alert_decision.py      # Policy inference helper
│   │       └── alert_policy.pkl       # Trained RL policy model
│   │
│   ├── Phase2/                     # GIS analysis scripts
│   │   ├── dem_low_lying.py        # DEM-based low-lying zone extraction
│   │   ├── infrastructure_overlay.py  # Infrastructure spatial overlay
│   │   └── population_lulc_exposure.py  # Population + LULC exposure analysis
│   │
│   ├── cv/                         # Computer Vision module
│   │   ├── water_detection.py      # NDWI-based flood highlighting
│   │   ├── train_cv_model.py       # CV model training utilities
│   │   ├── calibration.json        # Pixel calibration parameters
│   │   └── output/                 # Generated CV analysis images
│   │       ├── highlighted_after.jpg   # CYAN flood-highlighted image
│   │       ├── before_after_panel.jpg  # Side-by-side comparison panel
│   │       ├── flood_mask.jpg          # Binary flood mask
│   │       ├── water_mask_after.jpg    # Water segmentation mask
│   │       └── annotated_*.jpg         # Annotated satellite images
│   │
│   ├── genai_rag/                  # GenAI + RAG chatbot
│   │   ├── build_index.py          # FAISS index builder from PDF
│   │   ├── query_rag.py            # RAG query pipeline
│   │   ├── knowledge_base/
│   │   │   └── ndma guidelines.pdf # NDMA flood management guidelines
│   │   └── ndma_index/
│   │       ├── index.faiss         # FAISS vector store
│   │       └── index.pkl           # Document metadata index
│   │
│   ├── data/
│   │   ├── preprocess.py           # Data preprocessing pipeline
│   │   ├── raw/
│   │   │   ├── Demographics/
│   │   │   │   └── mumbai_wardwise_population_data.csv
│   │   │   └── hydrology/
│   │   │       └── mah_waterways.gpkg
│   │   ├── processed/
│   │   │   ├── Core_features.csv   # Historical weather + flood labels
│   │   │   ├── model_input.csv     # Processed ANN training data
│   │   │   ├── flood_impact_zones.geojson
│   │   │   └── low_lying_zones.geojson
│   │   └── Critical Centers/
│   │       ├── Mah_healthcare.gpkg
│   │       ├── mah_shelter.gpkg
│   │       ├── mah_police.gpkg
│   │       └── mah_firestat.gpkg
│   │
│   ├── scripts/
│   │   ├── before_image.py         # Download pre-flood Sentinel-2 image
│   │   └── after_image.py          # Download post-flood Sentinel-2 image
│   │
│   ├── output/                     # Raw satellite imagery outputs
│   │   ├── after_flood_r10m.jpg    # After flood at 10m resolution
│   │   ├── after_flood_r20m.jpg    # After flood at 20m resolution
│   │   └── after_flood_r60m.jpg    # After flood at 60m resolution
│   │
│   ├── config/
│   │   └── alert_rule.json         # Alert threshold configuration
│   │
│   └── req.txt                     # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Rain-Guard-AI.git
cd Rain-Guard-AI
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r Project/req.txt
```

> **Note:** TensorFlow installation may vary by platform. For GPU support, install `tensorflow-gpu` instead.

### 4. Run the Dashboard
```bash
cd Project/dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ⚙️ Configuration

### Logo Path
The logo path in `app.py` defaults to a local Windows path. Update it for your environment:
```python
# In Project/dashboard/app.py, line ~38
CUSTOM_LOGO_PATH = r"Project/dashboard/assets/RainGuard.png"
```

### Alert Configuration (`config/alert_rule.json`)
```json
{
  "high_threshold": 0.75,
  "medium_threshold": 0.50,
  "auto_alert_on_high": true
}
```

### API Keys (entered in the Streamlit sidebar)
| API | Purpose | Required? |
|-----|---------|-----------|
| Gemini API Key | AI situation reports | Optional |
| WeatherAPI Key | Live weather data | Optional (Open-Meteo is free fallback) |
| Telegram Bot Token + Chat ID | Alert dispatch | Optional |
| SMTP credentials | Email alerts | Optional |

---

## 🧠 ML Models

### ANN Flood Risk Model

**Architecture:**
```
Input (4 features) → Dense(64, ReLU) → BatchNorm → Dropout(0.3)
                  → Dense(32, ReLU) → BatchNorm → Dropout(0.2)
                  → Dense(16, ReLU) → Dense(1, Sigmoid)
```

**Training Details:**
- Optimizer: Adam with ReduceLROnPlateau
- Early stopping on validation loss (patience=15)
- Class-weight balancing to handle flood event rarity
- Features: `Intensity`, `soil_moisture_index`, `elevation_risk`, `humidity`

**Retrain the model:**
```bash
cd Project/model
python train_ann.py
```

### RL Alert Policy

A **Logistic Regression classifier** trained on historical `(risk_score, alert_sent, reward)` triples acts as the alert dispatch policy.

**Retrain the policy:**
```bash
cd "Project/Phase 3/rl"
python train_alert_policy.py
```

---

## 🛰️ Satellite CV Module

The Computer Vision module performs **zero-QGIS flood detection** using pure Python:

```
Sentinel-2 Images (R=B4, G=B3, B=B2)
         │
         ▼
  Gamma Enhancement (dark images ~21/255 → ~126/255)
         │
         ▼
  NDWI = (G - R) / (G + R) per pixel
         │
         ▼
  Flood pixels = NDWI increased before→after (new water bodies)
         │
         ▼
  Paint flood pixels CYAN [0, 200, 255]
  Preserve real RGB for all other pixels
         │
         ▼
  Build before|after comparison panel
  Calculate flood area (km²) and land coverage (%)
```

**Download satellite imagery:**
```bash
cd Project/scripts
python before_image.py   # Downloads pre-flood Sentinel-2 image
python after_image.py    # Downloads post-flood Sentinel-2 image
```

---

## 💬 GenAI RAG Chatbot

The chatbot answers flood management questions using NDMA guidelines as a knowledge base.

**Build the FAISS index (run once):**
```bash
cd Project/genai_rag
python build_index.py
```

This processes `knowledge_base/ndma guidelines.pdf` into `ndma_index/index.faiss`.

**Test RAG queries standalone:**
```bash
python query_rag.py
```

---

## 📡 Alert System

### Telegram Setup
1. Create a bot via [@BotFather](https://t.me/BotFather) and copy the token
2. Get your chat ID by messaging [@userinfobot](https://t.me/userinfobot)
3. Enter both in the Streamlit sidebar under **Notifications**

### Email (SMTP) Setup
1. Enable 2FA on your Gmail account
2. Generate an [App Password](https://myaccount.google.com/apppasswords)
3. Use `smtp.gmail.com`, port `587`, and the app password in the sidebar

### Alert Message Format
```
🚨 RAINGUARD FLOOD ALERT - HIGH RISK
═══════════════════════════════════════

📊 SITUATION OVERVIEW
• Timestamp: 2025-07-15 14:32:01
• Risk Level: HIGH
• Flood Probability: 89.3%

📍 AFFECTED AREAS
RED ZONE: Wards K/E, L, M/E, M/W, N, S, T
• Population at Risk: 2,841,200

🏥 CRITICAL INFRASTRUCTURE — RED ZONE
• Hospital: 12
• Shelter: 8
• Police Station: 6
• Fire Station: 4

⚠️ RECOMMENDED ACTIONS
1. IMMEDIATE EVACUATION of RED zone wards
2. Alert emergency services in affected areas
...
```

---

## 📊 Dashboard Tabs

| Tab | Icon | Description |
|-----|------|-------------|
| Introduction | 🏠 | Animated feature overview with Lottie |
| Population Zones | 📍 | Ward-level risk map + summary table |
| Infrastructure | 🏥 | Critical asset risk map + zone cards |
| Combined Map | 🗺️ | Unified 2D operational map |
| 3D Zone Map | 🌐 | PyDeck 3D column visualization |
| Visual Analytics | 📊 | Six interactive Plotly charts |
| Satellite CV | 🛰️ | Before/after flood imagery analysis |
| AI Situation Report | 🤖 | Gemini AI emergency report generator |
| Chatbot Facility | 💬 | RAG chatbot with NDMA knowledge base |

---

## 🗂️ Data Sources

| Data | Source | Format |
|------|--------|--------|
| Mumbai ward population (2025) | Municipal census extrapolation | CSV |
| Critical infrastructure (healthcare, shelters, police, fire) | OpenStreetMap + Maharashtra GIS | GeoPackage (.gpkg) |
| Waterways | Maharashtra hydrology data | GeoPackage (.gpkg) |
| Flood impact zones | DEM + LULC analysis | GeoJSON |
| Low-lying zones | SRTM DEM processing | GeoJSON |
| Historical weather + flood labels | IMD + custom processing | CSV |
| Satellite imagery | Sentinel-2 (ESA Copernicus) | JPEG (RGB composite) |
| NDMA guidelines | National Disaster Management Authority | PDF |

---

## 📦 Requirements

```
streamlit>=1.32.0
tensorflow>=2.15.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
geopandas>=0.14.0
rasterio>=1.3.0
shapely>=2.0.0
pydeck>=0.8.0
plotly>=5.0.0
joblib>=1.3.0
langchain>=0.1.0
langchain-community>=0.0.20
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
pypdf>=4.0.0
requests>=2.31.0
opencv-python>=4.9.0
google-generativeai>=0.4.0
twilio>=8.0.0
Pillow>=10.0.0
```

Install all with:
```bash
pip install -r Project/req.txt
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get involved:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes: `git commit -m 'Add: your feature description'`
4. **Push** to the branch: `git push origin feature/your-feature-name`
5. **Open** a Pull Request

### Areas for Contribution
- Additional city/region support beyond Mumbai
- Integration with IMD (India Meteorological Department) live APIs
- Mobile-responsive dashboard improvements
- More advanced CV flood segmentation (U-Net / DeepLab)
- Time-series forecasting (LSTM/Prophet) for 24-hour outlook

---

<div align="center">



<br/>

**Built with ❤️ for disaster preparedness and community safety**

*RainGuard AI · Flood Early Warning System · Mumbai, Maharashtra*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-red?style=flat-square)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>
