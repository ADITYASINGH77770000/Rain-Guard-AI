# import json
# from pathlib import Path

# import streamlit as st
# import streamlit.components.v1 as components


# ANIMATION_DIR = Path(__file__).resolve().parent / "assets" / "Animations"


# def _render_lottie(json_path: Path, height: int = 220):
#     try:
#         animation_data = json.loads(json_path.read_text(encoding="utf-8"))
#     except Exception:
#         st.warning("Animation JSON could not be loaded.")
#         return

#     html = f"""
#     <div id="lottie-holder" style="width:100%;height:{height}px;"></div>
#     <script src="https://cdnjs.cloudflare.com/ajax/libs/bodymovin/5.12.2/lottie.min.js"></script>
#     <script>
#       const animData = {json.dumps(animation_data)};
#       lottie.loadAnimation({{
#         container: document.getElementById("lottie-holder"),
#         renderer: "svg",
#         loop: true,
#         autoplay: true,
#         animationData: animData
#       }});
#     </script>
#     """
#     components.html(html, height=height)


# def render_introduction_page():
#     st.markdown(
#         """
#         <style>
#         .feature-card {
#             background: #E2E8F0;
#             border-left: 6px solid #1D4ED8;
#             border-radius: 10px;
#             padding: 14px 16px;
#             color: #0F172A;
#             box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12);
#         }
#         .feature-title {
#             margin: 0 0 8px 0;
#             font-weight: 800;
#             color: #0B3A75;
#             font-size: 1.02rem;
#         }
#         .feature-list {
#             margin: 0;
#             padding-left: 18px;
#             color: #0F172A;
#         }
#         .feature-bullet {
#             margin: 3px 0;
#             line-height: 1.5;
#             font-size: 0.95rem;
#         }
#         .feature-media-card {
#             background: #0F172A;
#             border: 1px solid #1D4ED8;
#             border-radius: 10px;
#             padding: 14px 16px;
#             color: #E2E8F0;
#             box-shadow: 0 0 14px rgba(37, 99, 235, 0.25);
#         }
#         .feature-media-card p {
#             margin: 0 0 10px 0;
#             line-height: 1.45;
#         }
#         .neon-line {
#             height: 3px;
#             margin: 14px 0 18px 0;
#             border-radius: 999px;
#             background: linear-gradient(90deg, #22D3EE, #2563EB, #38BDF8, #22D3EE);
#             box-shadow: 0 0 10px #22D3EE, 0 0 18px #2563EB;
#             animation: neonFlow 2.2s linear infinite;
#             background-size: 240% 100%;
#         }
#         @keyframes neonFlow {
#             0% { background-position: 0% 0; }
#             100% { background-position: 200% 0; }
#         }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

#     features = [
#         {
#             "title": "1) Flood Prediction Engine (ANN + Risk Logic)",
#             "lines": [
#                 "This module predicts flood probability using rainfall, soil saturation, elevation risk, and humidity.",
#                 "A trained ANN model is loaded from local files when available for realistic scoring.",
#                 "If model assets are missing, the app uses a fallback formula so prediction still works.",
#                 "The score is converted into LOW, MEDIUM, or HIGH risk labels for quick interpretation.",
#                 "This gives a fast first warning before deeper map and asset impact analysis begins.",
#                 "The risk output drives alert decisions, zone classification, and executive summaries.",
#             ],
#             "accent": "#1D4ED8",
#             "animation_json": ANIMATION_DIR / "Artificial intelligence digital technology.json",
#         },
#         {
#             "title": "2) Live Data Ingestion (Weather + Elevation)",
#             "lines": [
#                 "This module pulls real-time weather using Open-Meteo when API mode is enabled.",
#                 "Latitude and longitude controls allow quick relocation of analysis to other areas.",
#                 "Rainfall and humidity are refreshed and normalized for direct model input use.",
#                 "Terrain exposure is estimated with elevation lookup to improve flood context.",
#                 "Manual input mode remains available for testing emergency scenarios quickly.",
#                 "This ensures the system can run in both connected and controlled-demo conditions.",
#             ],
#             "accent": "#0EA5E9",
#             "animation_json": ANIMATION_DIR / "Cloud robotics abstract.json",
#         },
#         {
#             "title": "3) Population Zone Analysis (Ward-Level Risk)",
#             "lines": [
#                 "Ward-level population density is combined with flood probability to compute exposure.",
#                 "Each ward is placed into RED, ORANGE, or GREEN based on computed risk score.",
#                 "The map uses proportional circles so larger populations are visually emphasized.",
#                 "Tables provide ward category, population counts, and zone color coding.",
#                 "This helps authorities identify where more people are likely to be affected first.",
#                 "It is designed for fast planning of evacuation and support resource prioritization.",
#             ],
#             "accent": "#2563EB",
#             "animation_json": ANIMATION_DIR / "maps.json",
#         },
#         {
#             "title": "4) Critical Infrastructure Risk Mapping",
#             "lines": [
#                 "Hospitals, shelters, police stations, fire stations, and waterways are weighted by criticality.",
#                 "The module estimates risk impact for each asset category using risk probability multipliers.",
#                 "Asset points are displayed on map with zone colors for immediate operational visibility.",
#                 "Zone-wise tables list how many critical services are at high or medium risk.",
#                 "This directly supports continuity planning for life-saving and emergency services.",
#                 "It also provides a structured basis for staged alerting and field coordination.",
#             ],
#             "accent": "#1E40AF",
#             "animation_json": ANIMATION_DIR / "Homepage -map dark blue 1.json",
#         },
#         {
#             "title": "5) Combined Zone Operations Map",
#             "lines": [
#                 "This screen merges ward population zones, infrastructure points, and GIS flood polygons.",
#                 "A unified map gives command teams one operational picture during active events.",
#                 "Legend and summary cards reduce interpretation time during high-pressure decisions.",
#                 "Color layers are tuned so high-risk zones and critical assets stand out immediately.",
#                 "Population-at-risk metrics are shown alongside spatial context for action planning.",
#                 "This creates a bridge between model outputs and practical ground response workflows.",
#             ],
#             "accent": "#1D4ED8",
#             "animation_json": ANIMATION_DIR / "Global network.json",
#         },
#         {
#             "title": "6) AI Situation Report + Notification Workflow",
#             "lines": [
#                 "The AI report module converts numeric outputs into a readable emergency narrative.",
#                 "Risk level, impacted wards, and infrastructure summaries are packaged in one report.",
#                 "Gemini integration can generate structured recommendations and outlook sections.",
#                 "Telegram and SMTP options allow direct alert transmission from the same interface.",
#                 "The alert history table preserves communication traceability during operations.",
#                 "This closes the loop from detection to communication and stakeholder decision support.",
#             ],
#             "accent": "#0F766E",
#             "animation_json": ANIMATION_DIR / "Assistant-Bot.json",
#         },
#     ]

#     for i, feature in enumerate(features):
#         details_html = (
#             f"<div class='feature-card' style='border-left-color:{feature['accent']}'>"
#             f"<p class='feature-title'>{feature['title']}</p>"
#             + "<ul class='feature-list'>"
#             + "".join([f"<li class='feature-bullet'>{line}</li>" for line in feature["lines"]])
#             + "</ul>"
#             + "</div>"
#         )
#         media_path = feature.get("animation_json")
#         show_media = bool(media_path) and Path(media_path).exists()

#         if i % 2 == 0:
#             col_left, col_right = st.columns([2.4, 1.1], gap="large")
#             with col_left:
#                 st.markdown(details_html, unsafe_allow_html=True)
#             with col_right:
#                 if show_media:
#                     _render_lottie(Path(media_path))
#                 else:
#                     st.warning("Animation file not found for this feature.")
#         else:
#             col_left, col_right = st.columns([1.1, 2.4], gap="large")
#             with col_left:
#                 if show_media:
#                     _render_lottie(Path(media_path))
#                 else:
#                     st.warning("Animation file not found for this feature.")
#             with col_right:
#                 st.markdown(details_html, unsafe_allow_html=True)

#         st.markdown("<div class='neon-line'></div>", unsafe_allow_html=True)


import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


ANIMATION_DIR = Path(__file__).resolve().parent / "assets" / "Animations"


def _render_lottie(json_path: Path, height: int = 220):
    try:
        animation_data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        st.warning("Animation JSON could not be loaded.")
        return

    html = f"""
    <div id="lottie-holder" style="width:100%;height:{height}px;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bodymovin/5.12.2/lottie.min.js"></script>
    <script>
      const animData = {json.dumps(animation_data)};
      lottie.loadAnimation({{
        container: document.getElementById("lottie-holder"),
        renderer: "svg",
        loop: true,
        autoplay: true,
        animationData: animData
      }});
    </script>
    """
    components.html(html, height=height)


def render_introduction_page():
    st.markdown(
        """
        <style>
        .feature-card {
            background: #E2E8F0;
            border-left: 6px solid #1D4ED8;
            border-radius: 10px;
            padding: 14px 16px;
            color: #0F172A;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12);
        }
        .feature-title {
            margin: 0 0 8px 0;
            font-weight: 800;
            color: #0B3A75;
            font-size: 1.02rem;
        }
        .feature-list {
            margin: 0;
            padding-left: 18px;
            color: #0F172A;
        }
        .feature-bullet {
            margin: 3px 0;
            line-height: 1.5;
            font-size: 0.95rem;
        }
        .feature-media-card {
            background: #0F172A;
            border: 1px solid #1D4ED8;
            border-radius: 10px;
            padding: 14px 16px;
            color: #E2E8F0;
            box-shadow: 0 0 14px rgba(37, 99, 235, 0.25);
        }
        .feature-media-card p {
            margin: 0 0 10px 0;
            line-height: 1.45;
        }
        .neon-line {
            height: 3px;
            margin: 14px 0 18px 0;
            border-radius: 999px;
            background: linear-gradient(90deg, #22D3EE, #2563EB, #38BDF8, #22D3EE);
            box-shadow: 0 0 10px #22D3EE, 0 0 18px #2563EB;
            animation: neonFlow 2.2s linear infinite;
            background-size: 240% 100%;
        }
        @keyframes neonFlow {
            0% { background-position: 0% 0; }
            100% { background-position: 200% 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    features = [
        {
            "title": "1) Flood Prediction Engine (ANN + Risk Logic)",
            "lines": [
                "This module predicts flood probability using rainfall, soil saturation, elevation risk, and humidity.",
                "A trained ANN model is loaded from local files when available for realistic scoring.",
                "If model assets are missing, the app uses a fallback formula so prediction still works.",
                "The score is converted into LOW, MEDIUM, or HIGH risk labels for quick interpretation.",
                "This gives a fast first warning before deeper map and asset impact analysis begins.",
                "The risk output drives alert decisions, zone classification, and executive summaries.",
            ],
            "accent": "#1D4ED8",
            "animation_json": ANIMATION_DIR / "Artificial intelligence digital technology.json",
        },
        {
            "title": "2) Live Data Ingestion (Weather + Elevation)",
            "lines": [
                "This module pulls real-time weather using Open-Meteo when API mode is enabled.",
                "Latitude and longitude controls allow quick relocation of analysis to other areas.",
                "Rainfall and humidity are refreshed and normalized for direct model input use.",
                "Terrain exposure is estimated with elevation lookup to improve flood context.",
                "Manual input mode remains available for testing emergency scenarios quickly.",
                "This ensures the system can run in both connected and controlled-demo conditions.",
            ],
            "accent": "#0EA5E9",
            "animation_json": ANIMATION_DIR / "Cloud robotics abstract.json",
        },
        {
            "title": "3) Population Zone Analysis (Ward-Level Risk)",
            "lines": [
                "Ward-level population density is combined with flood probability to compute exposure.",
                "Each ward is placed into RED, ORANGE, or GREEN based on computed risk score.",
                "The map uses proportional circles so larger populations are visually emphasized.",
                "Tables provide ward category, population counts, and zone color coding.",
                "This helps authorities identify where more people are likely to be affected first.",
                "It is designed for fast planning of evacuation and support resource prioritization.",
            ],
            "accent": "#2563EB",
            "animation_json": ANIMATION_DIR / "maps.json",
        },
        {
            "title": "4) Critical Infrastructure Risk Mapping",
            "lines": [
                "Hospitals, shelters, police stations, fire stations, and waterways are weighted by criticality.",
                "The module estimates risk impact for each asset category using risk probability multipliers.",
                "Asset points are displayed on map with zone colors for immediate operational visibility.",
                "Zone-wise tables list how many critical services are at high or medium risk.",
                "This directly supports continuity planning for life-saving and emergency services.",
                "It also provides a structured basis for staged alerting and field coordination.",
            ],
            "accent": "#1E40AF",
            "animation_json": ANIMATION_DIR / "Homepage -map dark blue 1.json",
        },
        {
            "title": "5) Combined Zone Operations Map",
            "lines": [
                "This screen merges ward population zones, infrastructure points, and GIS flood polygons.",
                "A unified map gives command teams one operational picture during active events.",
                "Legend and summary cards reduce interpretation time during high-pressure decisions.",
                "Color layers are tuned so high-risk zones and critical assets stand out immediately.",
                "Population-at-risk metrics are shown alongside spatial context for action planning.",
                "This creates a bridge between model outputs and practical ground response workflows.",
            ],
            "accent": "#1D4ED8",
            "animation_json": ANIMATION_DIR / "Global network.json",
        },
        {
            "title": "6) Visual Analytics Dashboard (Interactive Charts)",
            "lines": [
                "This module transforms raw zone data into six interactive Plotly visualizations.",
                "Ward population bars, density scatter plots, and infrastructure heatmaps reveal patterns.",
                "Historical rainfall time series from 2016-2019 shows confirmed flood event markers.",
                "Each chart includes AI-generated summaries explaining key insights and risk drivers.",
                "Donut charts and sunburst diagrams break down zone distribution by category.",
                "This supports data-driven briefings and helps stakeholders understand risk composition.",
            ],
            "accent": "#7C3AED",
            "animation_json": ANIMATION_DIR / "AI data.json",
        },
        {
            "title": "7) Satellite CV Analysis (Before/After Flood Detection)",
            "lines": [
                "This module performs RGB-based flood water detection on Sentinel-2 satellite imagery.",
                "Before and after images are compared using NDWI analysis to identify new water bodies.",
                "Flood water pixels are highlighted in CYAN for instant visual recognition.",
                "The comparison activates automatically when the ANN model detects HIGH flood risk.",
                "For MEDIUM and LOW risk levels, only normal area conditions are shown.",
                "This provides ground-truth validation of model predictions using real satellite data.",
            ],
            "accent": "#0891B2",
            "animation_json": ANIMATION_DIR / "Technology isometric ai robot brain.json",
        },
        {
            "title": "8) AI Situation Report + Notification Workflow",
            "lines": [
                "The AI report module converts numeric outputs into a readable emergency narrative.",
                "Risk level, impacted wards, and infrastructure summaries are packaged in one report.",
                "Gemini integration can generate structured recommendations and outlook sections.",
                "Telegram and SMTP options allow direct alert transmission from the same interface.",
                "The alert history table preserves communication traceability during operations.",
                "This closes the loop from detection to communication and stakeholder decision support.",
            ],
            "accent": "#0F766E",
            "animation_json": ANIMATION_DIR / "Assistant-Bot.json",
        },
    ]

    for i, feature in enumerate(features):
        details_html = (
            f"<div class='feature-card' style='border-left-color:{feature['accent']}'>"
            f"<p class='feature-title'>{feature['title']}</p>"
            + "<ul class='feature-list'>"
            + "".join([f"<li class='feature-bullet'>{line}</li>" for line in feature["lines"]])
            + "</ul>"
            + "</div>"
        )
        media_path = feature.get("animation_json")
        show_media = bool(media_path) and Path(media_path).exists()

        if i % 2 == 0:
            col_left, col_right = st.columns([2.4, 1.1], gap="large")
            with col_left:
                st.markdown(details_html, unsafe_allow_html=True)
            with col_right:
                if show_media:
                    _render_lottie(Path(media_path))
                else:
                    st.warning("Animation file not found for this feature.")
        else:
            col_left, col_right = st.columns([1.1, 2.4], gap="large")
            with col_left:
                if show_media:
                    _render_lottie(Path(media_path))
                else:
                    st.warning("Animation file not found for this feature.")
            with col_right:
                st.markdown(details_html, unsafe_allow_html=True)

        st.markdown("<div class='neon-line'></div>", unsafe_allow_html=True)