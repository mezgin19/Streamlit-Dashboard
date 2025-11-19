# actions.py
import webbrowser
import subprocess

def execute_action(gesture, config):
    """
    Führt die konfigurierte Aktion für eine Geste aus.
    Unterstützt: 'url' (öffnet im Browser), 'android_app' (über ADB).
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
        elif action_type == "android_app":
            # ADB-Befehl: z. B. "com.android.chrome/com.google.android.apps.chrome.Main"
            cmd = ["adb", "shell", "am", "start", "-n", target]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                return True, f"📱 App gestartet: {target}"
            else:
                stderr = result.stderr.strip()
                # Versuche, häufige Fehler zu erklären
                if "error: device not found" in stderr.lower():
                    return False, "❌ ADB-Fehler: Kein Gerät verbunden. Stelle sicher, dass USB-Debugging aktiviert ist."
                elif "activity class" in stderr.lower() and "does not exist" in stderr.lower():
                    return False, f"❌ ADB-Fehler: Aktivität existiert nicht. Überprüfe den Aktivitätsnamen in '{target}'."
                elif "Bad component name" in stderr:
                    return False, f"❌ ADB-Fehler: Falsches Format. Verwende z. B. 'com.paket.name/com.paket.name.Hauptaktivität'. Aktuelles Ziel: '{target}'"
                else:
                    return False, f"❌ ADB-Fehler: {stderr}"
        else:
            return False, f"Unbekannter Aktionstyp: {action_type}"
    except subprocess.TimeoutExpired:
        return False, "❌ ADB-Befehl dauerte zu lange (Timeout)."
    except FileNotFoundError:
        return False, "❌ ADB nicht gefunden. Stelle sicher, dass ADB installiert und im Pfad ist."
    except Exception as e:
        return False, f"Fehler: {str(e)}"
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
