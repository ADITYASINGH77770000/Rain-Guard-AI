# from __future__ import annotations

# import html
# import hashlib
# import json
# import secrets
# from datetime import datetime
# from pathlib import Path

# import streamlit as st
# import streamlit.components.v1 as components


# USERS_DB_PATH = Path(__file__).resolve().parent / "auth_users.json"


# def _load_users() -> dict:
#     if not USERS_DB_PATH.exists():
#         return {}
#     try:
#         return json.loads(USERS_DB_PATH.read_text(encoding="utf-8"))
#     except Exception:
#         return {}


# def _save_users(users: dict) -> None:
#     USERS_DB_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")


# def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
#     salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
#     pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
#     return salt.hex(), pwd_hash.hex()


# def _verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
#     _, actual_hash_hex = _hash_password(password, salt_hex=salt_hex)
#     return secrets.compare_digest(actual_hash_hex, expected_hash_hex)


# def init_auth_state() -> None:
#     if "auth_logged_in" not in st.session_state:
#         st.session_state["auth_logged_in"] = False
#     if "auth_user_id" not in st.session_state:
#         st.session_state["auth_user_id"] = ""
#     if "auth_name" not in st.session_state:
#         st.session_state["auth_name"] = ""
#     if "audio_muted" not in st.session_state:
#         st.session_state["audio_muted"] = False


# def _set_logged_in(user_id: str, name: str) -> None:
#     st.session_state["auth_logged_in"] = True
#     st.session_state["auth_user_id"] = user_id
#     st.session_state["auth_name"] = name


# def logout_user() -> None:
#     st.session_state["auth_logged_in"] = False
#     st.session_state["auth_user_id"] = ""
#     st.session_state["auth_name"] = ""


# def _register_user(name: str, user_id: str, password: str) -> tuple[bool, str]:
#     users = _load_users()
#     norm_id = user_id.strip().lower()
#     if norm_id in users:
#         return False, "User ID already exists. Please choose another."

#     salt_hex, hash_hex = _hash_password(password)
#     users[norm_id] = {
#         "name": name.strip(),
#         "salt": salt_hex,
#         "password_hash": hash_hex,
#         "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     }
#     _save_users(users)
#     return True, "Registration successful. You can now log in."


# def _authenticate(user_id: str, password: str) -> tuple[bool, str]:
#     users = _load_users()
#     norm_id = user_id.strip().lower()
#     user = users.get(norm_id)
#     if not user:
#         return False, "User ID not found."

#     if not _verify_password(password, user.get("salt", ""), user.get("password_hash", "")):
#         return False, "Invalid password."

#     _set_logged_in(norm_id, user.get("name", norm_id))
#     return True, "Login successful."


# def render_auth_page() -> None:
#     st.markdown("### Account Access")
#     st.caption("Register a new account, then log in to access the platform.")

#     tab_register, tab_login = st.tabs(["Register (New User)", "Login"])

#     with tab_register:
#         with st.form("register_form", clear_on_submit=True):
#             reg_name = st.text_input("Full Name")
#             reg_user_id = st.text_input("User ID")
#             reg_password = st.text_input("Password", type="password")
#             reg_confirm = st.text_input("Confirm Password", type="password")
#             reg_submit = st.form_submit_button("Register")

#         if reg_submit:
#             if not reg_name.strip() or not reg_user_id.strip() or not reg_password:
#                 st.error("Please fill all required fields.")
#             elif len(reg_password) < 6:
#                 st.error("Password must be at least 6 characters.")
#             elif reg_password != reg_confirm:
#                 st.error("Passwords do not match.")
#             else:
#                 ok, msg = _register_user(reg_name, reg_user_id, reg_password)
#                 if ok:
#                     st.success(msg)
#                 else:
#                     st.error(msg)

#     with tab_login:
#         with st.form("login_form", clear_on_submit=False):
#             login_user_id = st.text_input("User ID", key="login_user_id_input")
#             login_password = st.text_input("Password", type="password", key="login_password_input")
#             login_submit = st.form_submit_button("Login")

#         if login_submit:
#             if not login_user_id.strip() or not login_password:
#                 st.error("Please enter User ID and Password.")
#             else:
#                 ok, msg = _authenticate(login_user_id, login_password)
#                 if ok:
#                     st.success(msg)
#                     st.rerun()
#                 else:
#                     st.error(msg)


# def ensure_authenticated() -> None:
#     init_auth_state()
#     if st.session_state.get("auth_logged_in"):
#         return
#     render_auth_page()
#     st.stop()


# def render_sidebar_user_footer() -> None:
#     if not st.session_state.get("auth_logged_in"):
#         return

#     signed_name = html.escape(st.session_state.get("auth_name", ""))
#     signed_user_id = html.escape(st.session_state.get("auth_user_id", ""))

#     st.sidebar.markdown(
#         """
#         <style>
#         .account-neon-line {
#             height: 2px;
#             margin: 8px 0 10px 0;
#             border-radius: 999px;
#             background: linear-gradient(90deg, #22D3EE, #2563EB, #38BDF8, #22D3EE);
#             background-size: 240% 100%;
#             box-shadow: 0 0 8px rgba(34, 211, 238, 0.75), 0 0 16px rgba(37, 99, 235, 0.55);
#             animation: accountNeonFlow 2.2s linear infinite;
#         }
#         @keyframes accountNeonFlow {
#             0% { background-position: 0% 0; }
#             100% { background-position: 200% 0; }
#         }
#         .account-footer-box {
#             background: #0F172A;
#             border: 1px solid #1D4ED8;
#             border-radius: 10px;
#             padding: 10px 12px;
#             margin-bottom: 8px;
#         }
#         .account-inline {
#             color: #E0F2FE;
#             font-size: 0.82rem;
#             margin: 0;
#         }
#         .account-inline strong {
#             color: #93C5FD;
#             font-weight: 700;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

#     st.sidebar.markdown("<div class='account-neon-line'></div>", unsafe_allow_html=True)
#     st.sidebar.markdown(
#         f"""
#         <div class="account-footer-box">
#             <p class="account-inline"><strong>Signed In As:</strong> {signed_name}<br><strong>User ID:</strong> {signed_user_id}</p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     # Mute toggle button
#     col1, col2 = st.sidebar.columns(2)

#     with col1:
#         logout_clicked = st.button("Logout", use_container_width=True)

#     with col2:
#         mute_icon = "🔇" if st.session_state.get("audio_muted") else "🔊"
#         mute_clicked = st.button(mute_icon, use_container_width=True, help="Toggle background music")

#     if mute_clicked:
#         st.session_state["audio_muted"] = not st.session_state.get("audio_muted")
#         st.rerun()

#     if logout_clicked:
#         logout_user()
#         components.html(
#             """
#             <script>
#               window.open('', '_self');
#               window.close();
#             </script>
#             """,
#             height=0,
#         )
#         st.sidebar.info("Logged out. If this tab does not close automatically, close it manually.")
#         st.rerun()


from __future__ import annotations

import html
import hashlib
import json
import secrets
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


USERS_DB_PATH = Path(__file__).resolve().parent / "auth_users.json"


def _load_users() -> dict:
    if not USERS_DB_PATH.exists():
        return {}
    try:
        return json.loads(USERS_DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    USERS_DB_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return salt.hex(), pwd_hash.hex()


def _verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    _, actual_hash_hex = _hash_password(password, salt_hex=salt_hex)
    return secrets.compare_digest(actual_hash_hex, expected_hash_hex)


def init_auth_state() -> None:
    defaults = {
        "auth_logged_in": False,
        "auth_user_id":   "",
        "auth_name":      "",
        "audio_muted":    False,
        # tracks whether music has been started for this session
        "music_started":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _set_logged_in(user_id: str, name: str) -> None:
    st.session_state["auth_logged_in"] = True
    st.session_state["auth_user_id"]   = user_id
    st.session_state["auth_name"]      = name
    # ── flag so app.py knows to start music on the very next render ──
    st.session_state["music_started"]  = False   # reset → triggers fresh play


def logout_user() -> None:
    st.session_state["auth_logged_in"] = False
    st.session_state["auth_user_id"]   = ""
    st.session_state["auth_name"]      = ""
    st.session_state["music_started"]  = False


def _register_user(name: str, user_id: str, password: str) -> tuple[bool, str]:
    users   = _load_users()
    norm_id = user_id.strip().lower()
    if norm_id in users:
        return False, "User ID already exists. Please choose another."

    salt_hex, hash_hex = _hash_password(password)
    users[norm_id] = {
        "name":         name.strip(),
        "salt":         salt_hex,
        "password_hash": hash_hex,
        "created_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_users(users)
    return True, "Registration successful. You can now log in."


def _authenticate(user_id: str, password: str) -> tuple[bool, str]:
    users   = _load_users()
    norm_id = user_id.strip().lower()
    user    = users.get(norm_id)
    if not user:
        return False, "User ID not found."

    if not _verify_password(password, user.get("salt", ""), user.get("password_hash", "")):
        return False, "Invalid password."

    _set_logged_in(norm_id, user.get("name", norm_id))
    return True, "Login successful."


def render_auth_page() -> None:
    st.markdown("### Account Access")
    st.caption("Register a new account, then log in to access the platform.")

    tab_register, tab_login = st.tabs(["Register (New User)", "Login"])

    with tab_register:
        with st.form("register_form", clear_on_submit=True):
            reg_name     = st.text_input("Full Name")
            reg_user_id  = st.text_input("User ID")
            reg_password = st.text_input("Password", type="password")
            reg_confirm  = st.text_input("Confirm Password", type="password")
            reg_submit   = st.form_submit_button("Register")

        if reg_submit:
            if not reg_name.strip() or not reg_user_id.strip() or not reg_password:
                st.error("Please fill all required fields.")
            elif len(reg_password) < 6:
                st.error("Password must be at least 6 characters.")
            elif reg_password != reg_confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = _register_user(reg_name, reg_user_id, reg_password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            login_user_id  = st.text_input("User ID",  key="login_user_id_input")
            login_password = st.text_input("Password", type="password", key="login_password_input")
            login_submit   = st.form_submit_button("Login")

        if login_submit:
            if not login_user_id.strip() or not login_password:
                st.error("Please enter User ID and Password.")
            else:
                ok, msg = _authenticate(login_user_id, login_password)
                if ok:
                    st.success(msg)
                    # ── rerun: next render sees auth_logged_in=True
                    # app.py calls inject_bg_music() right after
                    # ensure_authenticated() returns — music starts.
                    st.rerun()
                else:
                    st.error(msg)


def ensure_authenticated() -> None:
    init_auth_state()
    if st.session_state.get("auth_logged_in"):
        return          # ← logged in: app.py continues, music plays
    render_auth_page()
    st.stop()           # ← not logged in: stop here, show login only


def render_sidebar_user_footer() -> None:
    if not st.session_state.get("auth_logged_in"):
        return

    signed_name    = html.escape(st.session_state.get("auth_name",    ""))
    signed_user_id = html.escape(st.session_state.get("auth_user_id", ""))

    st.sidebar.markdown(
        """
        <style>
        .account-neon-line {
            height: 2px; margin: 8px 0 10px 0; border-radius: 999px;
            background: linear-gradient(90deg,#22D3EE,#2563EB,#38BDF8,#22D3EE);
            background-size: 240% 100%;
            box-shadow: 0 0 8px rgba(34,211,238,0.75), 0 0 16px rgba(37,99,235,0.55);
            animation: accountNeonFlow 2.2s linear infinite;
        }
        @keyframes accountNeonFlow {
            0%   { background-position: 0%   0; }
            100% { background-position: 200% 0; }
        }
        .account-footer-box {
            background: #0F172A; border: 1px solid #1D4ED8;
            border-radius: 10px; padding: 10px 12px; margin-bottom: 8px;
        }
        .account-inline { color:#E0F2FE; font-size:0.82rem; margin:0; }
        .account-inline strong { color:#93C5FD; font-weight:700; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<div class='account-neon-line'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div class="account-footer-box">
            <p class="account-inline">
                <strong>Signed In As:</strong> {signed_name}<br>
                <strong>User ID:</strong> {signed_user_id}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Mute / Logout buttons ─────────────────────────────────────────
    # Use on_click callbacks so state flips BEFORE the rerun,
    # guaranteeing inject_bg_music() in app.py sees the updated value.

    def _toggle_mute():
        st.session_state["audio_muted"] = not st.session_state.get("audio_muted", False)

    def _do_logout():
        logout_user()

    col1, col2 = st.sidebar.columns(2)

    is_muted  = st.session_state.get("audio_muted", False)
    mute_icon = "🔇 Mute" if not is_muted else "🔊 Unmute"
    mute_help = "Click to mute background music" if not is_muted else "Click to unmute background music"

    with col1:
        st.button("Logout", use_container_width=True, on_click=_do_logout, key="btn_logout")

    with col2:
        st.button(mute_icon, use_container_width=True, on_click=_toggle_mute,
                  help=mute_help, key="btn_mute_toggle")

    # Show current music state clearly
    if is_muted:
        st.sidebar.caption("🔇 Music is muted")
    else:
        st.sidebar.caption("🔊 Music playing — Interstellar OST")

    # Handle logout (rerun after _do_logout callback)
    if not st.session_state.get("auth_logged_in"):
        components.html(
            """<script>window.open('','_self'); window.close();</script>""",
            height=0,
        )
        st.sidebar.info("Logged out. Close this tab if it does not close automatically.")
        st.rerun()