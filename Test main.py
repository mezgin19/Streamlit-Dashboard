# actions.py
import webbrowser

def execute_action(gesture, config):
    """
    Führt die konfigurierte Aktion für eine Geste aus.
    Unterstützt derzeit nur: 'url' (öffnet im Browser).
    Gibt (Erfolg, Nachricht) zurück.
    """
    if "gesten" not in config or gesture not in config["gesten"]:
        return False, "Keine Aktion für diese Geste konfiguriert."

    entry = config["gesten"][gesture]
    action_type = entry.get("type", "")
    target = entry.get("target", "").strip()

    if not target:
        return False, "Leeres Ziel."

    try:
        if action_type == "url":
            webbrowser.open(target)
            return True, f"✅ URL geöffnet: {target}"
        else:
            # Falls in der config.json ein android_app Eintrag verbleibt, wird ignoriert
            # oder eine Warnung ausgegeben.
            # Optional: return False, f"Nicht unterstützter Aktionstyp: {action_type}"
            # Oder einfach ignorieren und False zurückgeben:
            return False, f"Nicht unterstützter Aktionstyp: {action_type}. Nur 'url' wird unterstützt."

    except Exception as e:
        return False, f"Fehler beim Öffnen der URL: {str(e)}"

# Gestenerkennung_mit_Python.py
import numpy as np
from scipy import signal
from scipy.integrate import cumulative_trapezoid as cumtrapz
import pandas as pd

def detect_gesture_from_csv(file_path):
    """
    Erkennt Gesten (Kreis, Rechteck, Quadrat) aus Phyphox-Daten.
    Gibt ein Dict mit 'gesture', 'sx_cm', 'sy_cm' zurück.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8')
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Nur .csv oder .xlsx Dateien werden unterstützt.")
    
    required = ['Time (s)', 'Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)']
    if not all(col in df.columns for col in required):
        missing = [col for col in required if col not in df.columns]
        raise KeyError(f"Fehlende Spalten: {missing}. Erwartet: {required}")
    
    t = df['Time (s)'].values
    ax = df['Linear Acceleration x (m/s^2)'].values
    ay = df['Linear Acceleration y (m/s^2)'].values
    
    dt = np.mean(np.diff(t))
    fs = 1 / dt

    # Tiefpassfilter (12 Hz)
    b, a = signal.butter(2, 12 / (fs / 2), 'low')
    ax_f = signal.filtfilt(b, a, ax)
    ay_f = signal.filtfilt(b, a, ay)
    
    # Geschwindigkeit
    vx = cumtrapz(ax_f, t, initial=0)
    vy = cumtrapz(ay_f, t, initial=0)

    # Driftkorrektur Hochpassfilter
    bh, ah = signal.butter(2, 0.3 / (fs / 2), 'high')
    vx_f = signal.filtfilt(bh, ah, vx)
    vy_f = signal.filtfilt(bh, ah, vy)
    vx_f -= np.mean(vx_f)
    vy_f -= np.mean(vy_f)
    
    # Weg in cm
    sx = cumtrapz(vx_f, t, initial=0)
    sy = cumtrapz(vy_f, t, initial=0)
    sx_cm = 100 * (sx - sx[0])
    sy_cm = 100 * (sy - sy[0])

    # === Kreis-Erkennung (FFT) ===
    x = sx_cm - np.mean(sx_cm)
    y = sy_cm - np.mean(sy_cm)
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    halfN = len(X) // 2 + 1
    X, Y = X[:halfN], Y[:halfN]
    Pxx, Pyy = np.abs(X)**2, np.abs(Y)**2

    has_freq = (np.max(Pxx[1:]) > 4 * np.mean(Pxx[1:])) and (np.max(Pyy[1:]) > 4 * np.mean(Pyy[1:]))
    is_circle = False

    if has_freq:
        ix = np.argmax(Pxx[1:]) + 1
        iy = np.argmax(Pyy[1:]) + 1
        Ax, Ay = np.abs(X[ix]), np.abs(Y[iy])
        phi_x, phi_y = np.angle(X[ix]), np.angle(Y[iy])
        dphi = np.arctan2(np.sin(phi_y - phi_x), np.cos(phi_y - phi_x))
        amp_ratio = max(Ax, Ay) / min(Ax, Ay)
        is_circle = (amp_ratio <= 1.3) and (abs(abs(dphi) - np.pi/2) <= np.pi/3)

    # === Rechteck/Quadrat-Erkennung (Geometrie) ===
    dx = np.diff(sx_cm)
    dy = np.diff(sy_cm)
    direction = np.arctan2(dy, dx)
    d_ang = np.diff(direction)
    d_ang = np.arctan2(np.sin(d_ang), np.cos(d_ang))

    corner = (np.abs(d_ang) > np.deg2rad(45)) & (np.abs(d_ang) < np.deg2rad(135))
    num_c = np.sum(corner)

    curv = np.abs(d_ang[~corner])
    mean_curv = np.mean(curv) if len(curv) > 0 else 0

    x_sp = np.max(sx_cm) - np.min(sx_cm)
    y_sp = np.max(sy_cm) - np.min(sy_cm)
    aspect = max(x_sp, y_sp) / min(x_sp, y_sp) if min(x_sp, y_sp) > 1 else np.inf

    # Rechteck/Quadrat: Eckenzahl Vielfaches von 4
    n_cyc = round(num_c / 4)
    tol_rect = max(1, min(3, n_cyc))
    is_mult4 = (n_cyc >= 1) and (abs(num_c - 4 * n_cyc) <= tol_rect)
    is_rect = is_mult4 and (mean_curv < np.deg2rad(18)) and (aspect < 3.5)
    is_sq = is_rect and (aspect < 1.4)

    # === Entscheidung (Priorität: Quadrat > Rechteck > Kreis) ===
    if is_sq:
        gesture = "Quadrat"
    elif is_rect:
        gesture = "Rechteck"
    elif is_circle:
        gesture = "Kreis"
    else:
        gesture = "Unbekannt"

    return {
        "gesture": gesture,
        "sx_cm": sx_cm.tolist(),
        "sy_cm": sy_cm.tolist()
    }

# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import json
import matplotlib.pyplot as plt
from Gestenerkennung_mit_Python import detect_gesture_from_csv
from actions import execute_action

CONFIG_FILE = "config.json"

default_config = {
    "gesten": {
        "Kreis": {"type": "url", "target": "https://www.hs-hannover.de"},
        "Rechteck": {"type": "url", "target": "https://youtube.com"},
        "Quadrat": {"type": "url", "target": "https://google.com"} # Beispiel für Quadrat
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
        Die erkannte Geste kann dann eine Aktion auslösen, wie z.B. das Öffnen einer URL.

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
    # ADB-Option entfernt, nur noch 'url' verfügbar
    action_type = "url" # Da nur noch eine Option, kann sie festgelegt oder ausgeblendet werden
    # Falls du später z.B. 'firebase_action' hinzufügen willst:
    # action_type = st.selectbox(
    #     f"Typ ({gesture})",
    #     ["url", "firebase_action"], # Beispiel für Erweiterung
    #     index=0 if current["type"] == "url" else 1,
    #     key=f"type_{gesture}"
    # )
    st.text_input( # Verwende text_input statt selectbox, da Typ fest ist
        f"Typ ({gesture})",
        value=action_type,
        disabled=True, # Deaktiviert, da nur 'url' unterstützt wird
        key=f"type_{gesture}_disabled"
    )
    target = st.text_input(
        f"Ziel URL ({gesture})",
        value=current["target"].strip(),
        key=f"target_{gesture}"
    )

    # Speichere die neue Konfiguration, Typ ist immer 'url'
    new_gesten[gesture] = {"type": action_type, "target": target.strip()}

if st.button("✅ Speichern"):
    config["gesten"] = new_gesten           # Neue Geste zwischenspeichern
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)      # Datei speichern
    st.success("Gespeichert!")              # Meldung anzeigen
