import numpy as np
from scipy import signal
from scipy.integrate import cumulative_trapezoid as cumtrapz
import pandas as pd

def detect_gesture_from_csv(file_path):
    """
    Erkennt nur die Geste 'Kreis' aus Phyphox-Daten.
    Gibt ein Dict mit 'gesture', 'sx_cm', 'sy_cm' zurück.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8')
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Nur .csv oder .xlsx/.xls Dateien werden unterstützt.")
    
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

    # Driftkorrektur
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

    gesture = "Kreis" if is_circle else "Unbekannt"

    return {
        "gesture": gesture,
        "sx_cm": sx_cm.tolist(),
        "sy_cm": sy_cm.tolist()
    }