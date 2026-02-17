"""
Background music player module for RainGuardAI dashboard.
Handles audio playback with mute toggle functionality.
"""

import base64
from pathlib import Path
import streamlit as st


def render_background_music(audio_file_path: str) -> None:
    """
    Render background music player with mute state synchronization.

    Args:
        audio_file_path: Path to the audio file (e.g., 'assets/bg_music.mp3')
    """
    # Convert to Path object and make absolute if needed
    audio_path = Path(audio_file_path)
    if not audio_path.is_absolute():
        audio_path = Path(__file__).resolve().parent / audio_file_path

    # Check if file exists
    if not audio_path.exists():
        return

    # Read and encode audio file to base64
    try:
        with open(audio_path, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode()
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # Determine MIME type from file extension
    ext = audio_path.suffix.lower()
    mime_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }
    mime_type = mime_types.get(ext, "audio/mpeg")

    # Get mute state from session
    is_muted = st.session_state.get("audio_muted", False)

    # Create audio HTML with JavaScript control
    audio_html = f"""
    <audio
        id="bg_music"
        autoplay
        loop
        style="display: none;"
    >
        <source src="data:{mime_type};base64,{audio_data}" type="{mime_type}">
    </audio>

    <script>
        // Initialize audio element
        const bgMusic = document.getElementById('bg_music');

        // Set initial mute state
        const isMuted = {str(is_muted).lower()};
        if (bgMusic) {{
            bgMusic.volume = isMuted ? 0 : 0.3;
            bgMusic.play().catch(e => console.log("Autoplay prevented:", e));
        }}

        // Listen for mute state changes
        window.addEventListener('message', (event) => {{
            if (bgMusic && event.data.type === 'st:componentWasMounted') {{
                const mute = event.data.muted;
                bgMusic.volume = mute ? 0 : 0.3;
                if (mute && !bgMusic.paused) {{
                    bgMusic.pause();
                }} else if (!mute && bgMusic.paused) {{
                    bgMusic.play();
                }}
            }}
        }});

        // Attempt to play audio when user interacts with page
        document.addEventListener('click', () => {{
            if (bgMusic && bgMusic.paused && !{str(is_muted).lower()}) {{
                bgMusic.play().catch(e => console.log("Play failed:", e));
            }}
        }}, {{once: true}});
    </script>
    """

    st.markdown(audio_html, unsafe_allow_html=True)
