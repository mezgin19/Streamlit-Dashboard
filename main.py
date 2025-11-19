# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import json
import matplotlib.pyplot as plt
from Gestenerkennung_mit_Python import detect_gesture_from_csv
import webbrowser

CONFIG_FILE = "config.json"

default_config = {
    "Kreis": {"url": "https://www.hs-hannover.de"}
}

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2)

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

st.set_page_config(page_title="Gestenerkennung", layout="centered")
st.title("🎓 Gestenerkennungs-Dashboard")
st.markdown("Erkennt **Kreis** und zeigt die Bewegungsbahn an.")

# --- Konfiguration ---
st.header("🔵 Kreis ")
current_url = config.get("Kreis", {}).get("url", "https://www.hs-hannover.de")
new_url = st.text_input("Ziel-URL für Kreis", value=current_url.strip())

if st.button("✅ Speichern"):
    config["Kreis"] = {"url": new_url.strip()}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    st.success("Gespeichert!")

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
        if gesture == "Kreis":
            url = config["Kreis"]["url"]
            webbrowser.open(url)
            st.success(f"✅ URL geöffnet: {url}")
        else:
            st.warning("❓ Kein Kreis erkannt.")

    except Exception as e:
        st.error("❌ Fehler bei der Verarbeitung:")
        st.code(str(e))
        st.info("💡 Tipp: Spaltennamen müssen exakt sein:\n• Time (s)\n• Linear Acceleration x (m/s^2)\n• Linear Acceleration y (m/s^2)")