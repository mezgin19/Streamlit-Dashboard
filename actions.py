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