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