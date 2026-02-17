from __future__ import annotations

from typing import Generator

import streamlit as st


DISASTER_KEYWORDS = {
    "flood",
    "flash flood",
    "urban flood",
    "cyclone",
    "hurricane",
    "storm surge",
    "earthquake",
    "aftershock",
    "tsunami",
    "landslide",
    "mudslide",
    "avalanche",
    "wildfire",
    "forest fire",
    "heatwave",
    "cold wave",
    "drought",
    "lightning",
    "thunderstorm",
    "cloudburst",
    "disaster",
    "evacuation",
    "shelter",
    "emergency kit",
    "rescue",
    "first aid",
    "hazard",
    "mitigation",
    "preparedness",
}


def _yield_text_chunks(text: str, chunk_size: int = 60) -> Generator[str, None, None]:
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i : i + chunk_size]) + " "


def _is_disaster_related(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    return any(keyword in q for keyword in DISASTER_KEYWORDS)


def _disaster_prompt(user_query: str) -> str:
    return f"""
You are DisasterProtector, an expert disaster preparedness and response advisor.
You must answer ONLY disaster-related questions.

Rules:
1. If the user asks anything non-disaster-related, refuse briefly and ask for a disaster-related question.
2. Give practical guidance with clear steps and priorities.
3. Prefer structured format:
   - Immediate actions (first 10 minutes)
   - Next actions (next 1-6 hours)
   - Safety checks and what to avoid
4. Keep advice concise, actionable, and safety-first.
5. Do not provide illegal, harmful, or unsafe instructions.

User question:
{user_query}
"""


def _stream_gemini_answer(user_query: str, gemini_api_key: str) -> Generator[str, None, None]:
    try:
        import google.generativeai as genai
    except ImportError:
        yield "Gemini SDK is not installed. Run: pip install google-generativeai"
        return

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response_stream = model.generate_content(_disaster_prompt(user_query), stream=True)

        any_chunk = False
        for chunk in response_stream:
            text = getattr(chunk, "text", None)
            if text:
                any_chunk = True
                yield text

        if not any_chunk:
            yield "I could not generate a response. Please try again."
    except Exception as exc:
        yield f"Gemini error: {exc}"


def render_chatbot_page() -> None:
    st.subheader("Disaster Expert")
   

    st.markdown(
        """
        - Ask about floods, earthquakes, cyclones, landslides, wildfire, heatwaves, evacuation, shelters, and emergency kits.
        - Non-disaster topics are intentionally blocked.
        - In life-threatening emergencies, contact local emergency services immediately.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Chatbot Settings")
    chatbot_api_key = st.sidebar.text_input(
        "Gemini API Key (Chatbot)",
        type="password",
        value=st.session_state.get("chatbot_gemini_api_key", ""),
    )
    st.session_state["chatbot_gemini_api_key"] = chatbot_api_key

    history_key = "chatbot_messages"
    if history_key not in st.session_state:
        st.session_state[history_key] = [
            {
                "role": "assistant",
                "content": (
                    "I am your disaster expert assistant. Ask me about floods, earthquakes, cyclones, evacuation, "
                    "preparedness, and emergency actions."
                ),
            }
        ]

    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a disaster-related question...")
    if not user_query:
        return

    st.session_state[history_key].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not _is_disaster_related(user_query):
            refusal = (
                "I can only assist with disaster-related guidance. "
                "Please ask about hazards like floods, earthquakes, cyclones, landslides, wildfire, "
                "evacuation, shelter planning, or emergency preparedness."
            )
            full_text = st.write_stream(_yield_text_chunks(refusal, chunk_size=18))
        elif not chatbot_api_key.strip():
            missing_key_msg = "Please enter your Gemini API key in the sidebar to use the chatbot."
            full_text = st.write_stream(_yield_text_chunks(missing_key_msg, chunk_size=14))
        else:
            full_text = st.write_stream(_stream_gemini_answer(user_query, chatbot_api_key.strip()))

    st.session_state[history_key].append({"role": "assistant", "content": full_text})
