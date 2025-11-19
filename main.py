# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import json
import matplotlib.pyplot as plt
from Gestenerkennung_mit_Python import detect_gesture_from_csv
from actions import execute_action
import subprocess
import re

# === NEUE FUNKTION: Apps vom Android-Gerät abfragen (mit Anzeigenamen als Schlüssel) ===
@st.cache_data(ttl=600)  # Cache für 10 Minuten
def get_installed_apps():
    try:
        # Paketnamen abfragen (JETZT OHNE -3, also auch System-Apps)
        result = subprocess.run(
            ["adb", "shell", "pm", "list", "packages"],  # <- Hier entfernen wir -3
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            packages = result.stdout.strip().split('\n')
            package_list = [pkg.replace('package:', '') for pkg in packages if pkg.startswith('package:')]
            
            # Für jedes Paket den Anzeigenamen über dumpsys finden
            app_dict = {}
            for pkg in package_list:
                # Versuche, den Anzeigenamen zu finden
                name_result = subprocess.run(
                    ["adb", "shell", "dumpsys", "package", pkg],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if name_result.returncode == 0:
                    dumpsys_output = name_result.stdout
                    # Suche nach dem Label (z. B. `applicationLabel='Chrome'`)
                    match = re.search(r"applicationLabel='([^']+)'", dumpsys_output)
                    if match:
                        app_name = match.group(1)
                    else:
                        # Falls kein applicationLabel gefunden, verwende den Paketnamen als Anzeigename
                        app_name = pkg
                    app_dict[app_name] = pkg  # Anzeigename -> Paketname
            return app_dict
        else:
            st.error("❌ Fehler beim Abrufen der Apps: " + result.stderr)
            return {}
    except subprocess.TimeoutExpired:
        st.error("❌ ADB-Befehl dauerte zu lange (Timeout).")
        return {}
    except FileNotFoundError:
        st.error("❌ ADB nicht gefunden. Stelle sicher, dass ADB installiert und im Pfad ist.")
        return {}
    except Exception as e:
        st.error(f"❌ Fehler: {str(e)}")
        return {}

# Weitere Funktion: Hauptaktivität finden
def get_main_activity(package):
    try:
        result = subprocess.run(
            ["adb", "shell", "cmd", "package", "resolve-activity", "--brief", package],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if package in line:
                    return line.strip()
        return None
    except Exception:
        return None

# === ENDE NEUE FUNKTION ===

# === NEUE FUNKTION: ADB-Verbindungsstatus abfragen ===
@st.cache_data(ttl=30) # Cache für 30 Sekunden
def get_adb_status():
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            # Filtere Header und extrahiere Geräte
            devices = [line.split('\t')[0] for line in lines[1:] if '\t' in line and 'device' in line and 'offline' not in line]
            return True, devices
        else:
            return False, []
    except FileNotFoundError:
        return False, []

# === ENDE NEUE FUNKTION ===

CONFIG_FILE = "config.json"

default_config = {
    "gesten": {
        "Kreis": {"type": "url", "target": "https://www.hs-hannover.de                  "},
        "Rechteck": {"type": "url", "target": "https://youtube.com                  "},
        "Quadrat": {"type": "android_app", "target": "com.android.chrome/com.google.android.apps.chrome.Main"}
    }
}

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2)

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

st.set_page_config(page_title="Gestenerkennung – Bachelorarbeit", layout="wide")

# --- EINSTELLUNGEN IN DER SIDEBAR ALS EXPANDER ---
st.sidebar.header("⚙️ Einstellungen")

# Kontrast-Modus im Session State initialisieren
if 'high_contrast' not in st.session_state:
    st.session_state.high_contrast = False

# Schriftgröße im Session State initialisieren
if 'font_size' not in st.session_state:
    st.session_state.font_size = 16  # Standardgröße in px

with st.sidebar.expander("anzeigen", expanded=False):
    # Schriftgröße anpassen
    st.subheader("Schriftgröße")
    col_inc, col_mid, col_dec = st.columns([1, 2, 1])
    with col_inc:
        if st.button("A+", key="font_inc_btn"):
            if st.session_state.font_size < 30:
                st.session_state.font_size += 2
    with col_mid:
        st.write(f"**{st.session_state.font_size}px**")
    with col_dec:
        if st.button("A–", key="font_dec_btn"):
            if st.session_state.font_size > 12:
                st.session_state.font_size -= 2

    # Kontrast-Modus
    st.subheader("Kontrast")
    if st.button("🌙 Hohen Kontrast umschalten", key="contrast_btn"):
        st.session_state.high_contrast = not st.session_state.high_contrast


# Dynamischer CSS-Stil basierend auf font_size und high_contrast
css_styles = f"""
<style>
    html, body, [class*="View"], .stApp {{
        font-size: {st.session_state.font_size}px !important;
    }}
    h1 {{ font-size: {st.session_state.font_size + 8}px !important; }}
    h2 {{ font-size: {st.session_state.font_size + 6}px !important; }}
    h3 {{ font-size: {st.session_state.font_size + 4}px !important; }}
    .stButton>button {{
        height: 50px;
        width: 100%;
        font-size: {st.session_state.font_size + 4}px;
    }}
"""

if st.session_state.high_contrast:
    # Füge CSS für hohen Kontrast hinzu (Schwarz & Gelb/Weiß)
    css_styles += """
    .stApp, [data-testid="stSidebar"] {
        background-color: black !important;
        color: white !important; /* Standardtext auf weiß */
    }
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: black !important;
    }
    /* Sidebar-Textfarben */
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div:not([data-testid="stHeader"]) {
        color: white !important;
    }
    /* Wichtige Überschriften in Gelb */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: yellow !important;
    }
    /* Eingabefelder und Buttons */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-testid="stTextInput"] input,
    [data-testid="stSidebar"] [data-testid="stButton"] button,
    [data-testid="stTextInput"] input,
    .st-emotion-cache-1kyxreq { /* Dies ist die Klasse für Texteingaben im Hauptbereich */
        background-color: #222 !important;
        color: white !important;
        border: 1px solid yellow !important;
    }
    /* Button-Hover-Effekt in Gelb */
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover,
    [data-testid="stButton"] button:hover {
        background-color: #444 !important;
        color: yellow !important;
        border-color: yellow !important;
    }
    /* Erfolg-, Warn- und Fehlermeldungen */
    .st-emotion-cache-1gwvy71, /* Erfolgscontainer */
    .st-emotion-cache-10oheav, /* Erfolgsicon */
    .st-emotion-cache-1gwvy71 p,
    .st-emotion-cache-1gwvy71 div { /* Erfolgstext */
        background-color: transparent !important;
        color: yellow !important;
        border: 1px solid yellow !important;
    }
    .st-emotion-cache-1nycj0l, /* Warnungcontainer */
    .st-emotion-cache-p5msec, /* Warnungicon */
    .st-emotion-cache-1nycj0l p,
    .st-emotion-cache-1nycj0l div { /* Warnungstext */
        background-color: transparent !important;
        color: yellow !important;
        border: 1px solid yellow !important;
    }
    .st-emotion-cache-1dtefog, /* Fehlercontainer */
    .st-emotion-cache-k37gc0, /* Fehlericon */
    .st-emotion-cache-1dtefog p,
    .st-emotion-cache-1dtefog div { /* Fehlertext */
        background-color: transparent !important;
        color: yellow !important;
        border: 1px solid yellow !important;
    }
    /* Allgemeine Links im Hauptbereich */
    .stApp a {
        color: yellow !important;
    }
    .stApp a:hover {
        color: #ffff99 !important; /* Helleres Gelb beim Hover */
    }
    """
css_styles += "</style>"

st.markdown(css_styles, unsafe_allow_html=True)

# Hauptseite
st.title("🎓 Gestenerkennungs-Dashboard")
st.markdown("Erkennt **Kreis, Rechteck, Quadrat** und zeigt die Bewegungsbahn an.")

# --- Hilfe & Info ---
with st.expander("❓ Hilfe & Info", expanded=False):
    st.markdown(
        """
        **Willkommen beim Gestenerkennungs-Dashboard!**

        Diese Anwendung dient zur Erkennung einfacher Gesten (Kreis, Rechteck, Quadrat), 
        die z.B. mit der Phyphox-App auf einem Smartphone aufgezeichnet wurden. 
        Die erkannte Geste kann dann eine Aktion auslösen, wie z.B. das Öffnen einer URL 
        oder das Starten einer App auf einem Android-Gerät.

        ---
        **ADB-Verbindung (Android Debug Bridge)**

        Um Aktionen auf einem Android-Gerät auszulösen, muss eine Verbindung über ADB hergestellt werden.
        Dies erfordert die Installation von ADB auf diesem Computer und die Aktivierung der 
        USB-Debugging-Option auf dem Android-Gerät.

        - [Offizielle ADB-Dokumentation (Englisch)](https://developer.android.com/tools/adb)
        - [ADB-Download (Teil des Android SDK Platform Tools)](https://developer.android.com/tools/releases/platform-tools)

        ---
        **Phyphox-Datei**

        - Verwenden Sie die [Phyphox-App](https://phyphox.org/de/) auf Ihrem Smartphone.
        - Starten Sie das Experiment "Lineare Beschleunigung" oder ein ähnliches, 
          das die Beschleunigungen in x- und y-Richtung aufzeichnet.
        - Führen Sie Ihre Geste (z.B. einen Kreis in der Luft) aus und starten/stoppen Sie die Aufzeichnung.
        - Exportieren Sie die Daten als **CSV-Datei** und laden Sie sie hier hoch.
        - Die Spaltennamen in der CSV-Datei müssen exakt folgendermaßen lauten:
            - `Time (s)`
            - `Linear Acceleration x (m/s^2)`
            - `Linear Acceleration y (m/s^2)`
        """
    )

# --- ADB-Verbindungsinformationen ---
adb_ok, connected_devices = get_adb_status()

if adb_ok:
    if connected_devices:
        st.success(f"✅ ADB-Verbindung: {len(connected_devices)} Gerät(e) verbunden. ({', '.join(connected_devices)})")
    else:
        st.warning("⚠️ ADB-Verbindung besteht, aber kein Android-Gerät ist verbunden oder autorisiert.")
else:
    st.error("❌ ADB-Verbindung fehlgeschlagen. Stelle sicher, dass ADB installiert ist und das Android-Gerät per USB angeschlossen und die USB-Debugging Option aktiviert ist.")

# Optional: Zeige eine kurze Anleitung
with st.expander("Hinweise zur ADB-Verbindung"):
     st.markdown("""
     1. Stelle sicher, dass [ADB](https://developer.android.com/tools/adb              ) auf deinem System installiert und im `Pfad` verfügbar ist.
     2. Verbinde dein Android-Gerät per USB-Kabel mit diesem Computer.
     3. Aktiviere auf deinem Android-Gerät die **USB-Debugging**-Option (normalerweise unter Einstellungen > Entwickleroptionen).
     4. Akzeptiere die ADB-Autorisierungsanfrage auf deinem Android-Gerät, wenn sie erscheint.
     5. Starte die Streamlit-App neu, nachdem alles eingerichtet ist.
     """)

# --- Konfiguration ---
st.header("⚙️ Geste → Aktion")
new_gesten = {}

for gesture in ["Kreis", "Rechteck", "Quadrat"]:
    # Wähle passendes Symbol
    if gesture == "Kreis":
        symbol = "🔵"
    elif gesture == "Rechteck":
        symbol = "▭"
    elif gesture == "Quadrat":
        symbol = "🟥"
    else:
        symbol = "❓"  # Fallback für unbekannte Gesten

    st.subheader(f"{symbol} **{gesture}**")

    current = config["gesten"].get(gesture, {"type": "url", "target": ""})
    action_type = st.selectbox(
        f"Typ ({gesture})",
        ["url", "android_app"],
        index=0 if current["type"] == "url" else 1,
        key=f"type_{gesture}"
    )

    if action_type == "url":
        target = st.text_input(
            f"Ziel ({gesture})",
            value=current["target"].strip(),
            key=f"target_{gesture}"
        )
    elif action_type == "android_app":
        # Apps abfragen (mit Anzeigenamen als Schlüssel)
        app_dict = get_installed_apps()  # Gibt {"Chrome": "com.android.chrome", ...}
        if app_dict:
            # Auswahlbox mit Anzeigenamen
            selected_app_name = st.selectbox(
                f"App wählen ({gesture})",
                options=list(app_dict.keys()),  # Liste der Anzeigenamen
                index=None,
                key=f"app_{gesture}"
            )
            if selected_app_name:
                selected_pkg = app_dict[selected_app_name]  # Paketname holen
                activity = get_main_activity(selected_pkg)
                if activity:
                    target = activity
                    st.success(f"Ausgewählt: {selected_app_name} → {activity}")
                else:
                    target = selected_pkg
                    st.warning(f"Keine Hauptaktivität gefunden für {selected_app_name}")
            else:
                target = current["target"].strip()
        else:
            st.warning("Keine Apps gefunden. Stelle sicher, dass ADB funktioniert.")
            target = current["target"].strip()
    else:
        target = current["target"].strip()

    new_gesten[gesture] = {"type": action_type, "target": target.strip()}

if st.button("✅ Speichern"):
    config["gesten"] = new_gesten           # Neue Geste zwischenspeichern
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)      # Datei speichern
    st.success("Gespeichert!")              # Meldung anzeigen 

# --- Upload & Plot ---
st.header("🔍 Datei hochladen")
uploaded = st.file_uploader("Phyphox-Datei (CSV oder Excel)", type=["csv", "xlsx"])

if uploaded:
    try:
        file_ext = uploaded.name.split('.')[-1].lower()
        temp_path = f"temp.{file_ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        result = detect_gesture_from_csv(temp_path)
        gesture = result["gesture"]
        sx = result["sx_cm"]
        sy = result["sy_cm"]

        st.subheader(f"Erkannte Geste: **{gesture}**")

        # Bewegungsbahn plotten
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(sx, sy, 'b-', linewidth=2)
        ax.set_xlabel('x-Position [cm]')
        ax.set_ylabel('y-Position [cm]')
        ax.set_title(f'Bewegungsbahn – Erkannte Form: {gesture}')
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        st.pyplot(fig)

        # Aktion ausführen
        if gesture in ["Kreis", "Rechteck", "Quadrat"]:
            success, msg = execute_action(gesture, config)
            if success:
                st.success(msg)
            else:
                st.error(msg)
        else:
            st.warning("❓ Keine bekannte Geste erkannt.")

    except Exception as e:
        st.error("❌ Fehler bei der Verarbeitung:")
        st.code(str(e))
        st.info("💡 Tipp: Spaltennamen müssen exakt sein:\n• Time (s)\n• Linear Acceleration x (m/s^2)\n• Linear Acceleration y (m/s^2)")