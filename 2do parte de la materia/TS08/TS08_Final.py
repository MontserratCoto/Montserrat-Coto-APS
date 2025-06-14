# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 23:31:08 2025

@author: monse
"""


"""
TS8 - Filtrado No Lineal: Estimación y sustracción de línea de base con filtro de mediana

En esta tarea se utiliza un enfoque no lineal para eliminar el movimiento de línea de base
en señales de ECG. La estimación se hace aplicando dos filtros de mediana en cascada
(mediana de 200 ms seguida de una de 600 ms), y se resta dicha estimación a la señal original.
"""

# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy.signal import convolve



plt.close('all')

#############################
#%% Carga de la señal ECG
#############################

fs_ecg = 1000  # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()


# Variables del archivo
qrs_pattern = mat_struct['qrs_pattern1'].flatten()
heartbeat_normal = mat_struct['heartbeat_pattern1'].flatten()
heartbeat_ventricular = mat_struct['heartbeat_pattern2'].flatten()
qrs_detections = mat_struct['qrs_detections'].flatten()

# Normalización
ecg_one_lead = ecg_one_lead / np.std(ecg_one_lead)
heartbeat_normal = heartbeat_normal / np.std(heartbeat_normal)
heartbeat_ventricular = heartbeat_ventricular / np.std(heartbeat_ventricular)
qrs_pattern = qrs_pattern / np.std(qrs_pattern)

# Tiempo total señal completa
t_total = np.arange(len(ecg_one_lead)) / fs_ecg

###############################################
#%% Filtro de mediana doble (no lineal) sobre toda la señal
###############################################

win1 = 200  # 200 ms
win2 = 600  # 600 ms

# Asegurar ventanas impares (kernel_size en muestras)
if win1 % 2 == 0: win1 += 1
if win2 % 2 == 0: win2 += 1

# Aplicar filtro mediana doble a la señal completa
mediana1 = medfilt(ecg_one_lead, kernel_size=win1)
mediana2 = medfilt(mediana1, kernel_size=win2)

# Señal filtrada = señal original - línea base estimada
ecg_filtrada = ecg_one_lead - mediana2

print("\nValores de s (señal original), b^ (línea de base estimada) y x^ = s - b^ (señal filtrada):\n")
print("   s (original)     b^ (línea de base)     x^ (filtrada)")
for i in range(20):  # primeros 20 valores
    print(f"{ecg_one_lead[i]:12.6f}  {mediana2[i]:18.6f}  {ecg_filtrada[i]:15.6f}")


##########################################
#%% Gráfica completa (solo segmento para visualización clara)
##########################################
# Para visualizar, podés graficar un segmento temporal específico (ejemplo: 700000:745000)
start_plot = 700000
end_plot = 745000
t_segment = t_total[start_plot:end_plot]
ecg_segment = ecg_one_lead[start_plot:end_plot]
mediana2_segment = mediana2[start_plot:end_plot]
ecg_filtrada_segment = ecg_filtrada[start_plot:end_plot]

plt.figure(figsize=(12, 6))
plt.plot(t_segment, ecg_segment, label='ECG original', alpha=0.6)
plt.plot(t_segment, mediana2_segment, label='Línea de base estimada (b^)', linewidth=2)
plt.plot(t_segment, ecg_filtrada_segment, label='ECG limpio (x^)', color='green', linewidth=1.5)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud (mV)')
plt.title('Comparación completa (segmento) - original, línea de base y filtrado')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

##########################################################
#%% Gráfico centrado en un QRS detectado (±150 ms)
##########################################################

# Elegimos un QRS para centrar la ventana
i = 0
sample_center = int(qrs_detections[i])

window_qrs = 0.3  # segundos (±150 ms)
N_qrs = int(window_qrs * fs_ecg)

start_idx = sample_center - N_qrs // 2
end_idx = sample_center + N_qrs // 2

# Extraemos los segmentos de la señal original, línea base y señal filtrada
ecg_segment_qrs = ecg_one_lead[start_idx:end_idx]
mediana2_qrs = mediana2[start_idx:end_idx]
ecg_filtrada_qrs = ecg_filtrada[start_idx:end_idx]

# Tiempo relativo centrado en QRS
t_qrs = np.arange(-N_qrs//2, N_qrs//2) / fs_ecg

plt.figure(figsize=(10, 5))
plt.plot(t_qrs, ecg_segment_qrs, label='ECG Original', color='blue')
plt.plot(t_qrs, mediana2_qrs, label='Línea de base estimada', color='red')
plt.plot(t_qrs, ecg_filtrada_qrs, label='ECG Filtrada (Mediana)', color='green')
plt.title(f'Segmento centrado en QRS - Original vs Línea de base vs Filtrado (Mediana)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Gráfico equivalente a: "Estimación de línea de base por interpolación mediana (segmento)"

plt.figure(figsize=(12,5))
plt.plot(t_segment, ecg_segment, label='ECG original')
plt.plot(t_segment, mediana2_segment, color='orange', linewidth=2, label='Línea base estimada (Mediana)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.title('Estimación de línea de base por filtro de mediana (segmento)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Gráfico  "ECG con línea de base eliminada (Mediana) - Segmento"
plt.figure(figsize=(12,5))
plt.plot(t_segment, ecg_filtrada_segment, color='green', label='ECG sin línea de base (Mediana)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.title('ECG con línea de base eliminada (Mediana) - Segmento')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% SEGMENTO PQ con método de la mediana 

# Parámetros para detección del nivel PQ
n0 = 80  # muestras antes del QRS (ej: 80 ms)
window_size = 21  # ventana para la mediana (debe ser impar)

# Inicializamos listas para guardar puntos m_i y s_i
m_values = []
s_values = []

# Recorremos todos los QRS detectados
for ni in qrs_detections:
    start_idx = ni - n0 - window_size // 2
    end_idx = ni - n0 + window_size // 2 + 1  # +1 porque slicing es exclusivo
    
    # Validamos que no se salga del rango de la señal
    if start_idx < 0 or end_idx > len(ecg_one_lead):
        continue

    m_i = ni - n0  # índice central del punto PQ
    segment = ecg_one_lead[start_idx:end_idx]
    s_i = np.median(segment)  # mediana del segmento

    m_values.append(m_i)
    s_values.append(s_i)

# Convertimos a arrays y tiempo
m_values = np.array(m_values)
s_values = np.array(s_values)
t_m = m_values / fs_ecg

# Seleccionamos un segmento alrededor del primer m_i para graficar
first_idx = m_values[0]
segment_start = first_idx - 150
segment_end = first_idx + 150
if segment_start < 0: segment_start = 0
if segment_end > len(ecg_one_lead): segment_end = len(ecg_one_lead)

# Tiempo y señal en ese segmento
t_segment = t_total[segment_start:segment_end]
ecg_segment = ecg_one_lead[segment_start:segment_end]

# Puntos fiduciales dentro del segmento (para graficar)
fiducial_mask = (m_values >= segment_start) & (m_values < segment_end)
t_fiducials = t_m[fiducial_mask]
s_fiducials = s_values[fiducial_mask]


# Interpolar la línea base (mediana) para todo el rango de t_segment
linebase_interp = np.interp(t_segment, t_fiducials, s_fiducials)

plt.figure(figsize=(12,5))
plt.plot(t_segment, ecg_segment, label='ECG original', linewidth=1.5)
plt.plot(t_segment, linebase_interp, color='orange', linestyle='-', linewidth=2, label='Línea base interpolada (mediana)')
plt.scatter(t_fiducials, s_fiducials, color='red', label='Puntos PQ (nivel isoeléctrico)', zorder=5)
plt.axvline(x=m_values[0]/fs_ecg, color='green', linestyle='--', label='Inicio segmento PQ')

# Marcar los QRS dentro del segmento
for ni in qrs_detections:
    t_ni = ni / fs_ecg
    if segment_start/fs_ecg <= t_ni <= segment_end/fs_ecg:
        plt.axvline(x=t_ni, color='red', linestyle='-', alpha=0.5)

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.title('Segmento PQ: estimación nivel isoeléctrico y línea base (Mediana)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.show()

#################################
#%% Conclusión Mediana
#################################

print("""
Conclusión:
Se logró eliminar de manera efectiva el movimiento de línea de base con dos filtros de mediana
en cascada aplicados sobre la señal completa. El gráfico centrado en el QRS permite observar
que la forma de la onda QRS permanece intacta luego del filtrado, validando la eficacia
del método no lineal.
""")

###############################################
#%% 2. Filtro de Spline Cubico (no lineal) sobre toda la señal
###############################################
N_ecg = len(ecg_one_lead)
t_ecg = np.arange(N_ecg) / fs_ecg  # eje de tiempo en segundos

print(f"Muestras totales: {N_ecg}")

plt.figure(1)
plt.plot(ecg_one_lead)
plt.xlabel('Muestras')
plt.ylabel('Amplitud (mV)')
plt.title('Señal ECG completa')
plt.grid()

def estimate_baseline_spline(ecg_signal, qrs_positions, fs=1000, n0_ms=100, window_ms=20):
    """
    Estima la línea de base de una señal ECG usando spline cúbica,
    a partir de puntos en el segmento PQ (aprox. n0_ms antes del QRS).
    El valor se promedia en una ventana de window_ms alrededor del punto.
    
    Parámetros:
    - ecg_signal: señal ECG 1D
    - qrs_positions: índices donde se detectan los QRS (enteros)
    - fs: frecuencia de muestreo (Hz)
    - n0_ms: tiempo en ms para retroceder desde QRS (segmento PQ)
    - window_ms: ventana de promedio en ms
    
    Retorna:
    - baseline: línea de base interpolada (vector del mismo largo que ecg_signal)
    - baseline_times: tiempos de los puntos fiduciales usados para la spline
    - baseline_values: valores promedio en esos puntos
    """
    
    n0 = int(n0_ms * fs / 1000)          # conversión ms a muestras
    window = int(window_ms * fs / 1000)  # ventana en muestras
    
    t_ecg = np.arange(len(ecg_signal)) / fs
    baseline_times = []
    baseline_values = []

    for qrs in qrs_positions:
        idx = qrs - n0
        if idx - window//2 < 0 or idx + window//2 >= len(ecg_signal):
            continue  # evitamos salirnos del vector

        window_data = ecg_signal[idx - window//2 : idx + window//2]
        baseline_times.append(t_ecg[idx])
        baseline_values.append(np.mean(window_data))
        
    spline_func = CubicSpline(baseline_times, baseline_values)
    baseline = spline_func(t_ecg)
    return baseline, baseline_times, baseline_values

# Estimamos línea de base usando spline cúbica
baseline_full, base_times, base_vals = estimate_baseline_spline(ecg_one_lead, qrs_detections, fs=fs_ecg, n0_ms=100, window_ms=20)
ecg_clean_full = ecg_one_lead - baseline_full

print("\nSpline cúbico - primeros valores de s, b^ y x^ (filtrado spline):\n")
print("   s (original)     b^ (spline)         x^ (filtrada)")
for i in range(20):  # primeros 20 valores
    print(f"{ecg_one_lead[i]:12.6f}  {baseline_full[i]:18.6f}  {ecg_clean_full[i]:15.6f}")


#%% Segmento para detalle visual
seg_start = 700000
seg_end = 745000

ecg_segment = ecg_one_lead[seg_start:seg_end]
baseline_segment = baseline_full[seg_start:seg_end]
ecg_clean_segment = ecg_clean_full[seg_start:seg_end]
t_segment = t_ecg[seg_start:seg_end]

#%% Graficamos señal completa y línea de base con QRS
plt.figure(figsize=(14,4))
plt.plot(t_ecg, ecg_one_lead, label='ECG original', alpha=0.4)
plt.plot(t_ecg, baseline_full, label='Línea de base (Spline)', color='orange', linewidth=1.2)
plt.scatter(t_ecg[qrs_detections], ecg_one_lead[qrs_detections], color='red', s=10, label='QRS detectados')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.title('Señal ECG completa con línea de base estimada (Spline) y latidos')
plt.legend()
plt.grid(True)

# Graficamos detalle del segmento con puntos usados para spline
plt.figure(figsize=(12,4))
plt.plot(t_segment, ecg_segment, label='ECG original', alpha=0.5)
plt.plot(t_segment, baseline_segment, label='Línea de base (Spline)', color='orange', linewidth=2)

plt.scatter([t for t in base_times if seg_start/fs_ecg <= t <= seg_end/fs_ecg],
            [v for t, v in zip(base_times, base_vals) if seg_start/fs_ecg <= t <= seg_end/fs_ecg],
            color='red', label='Puntos fiduciales')

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.title('Estimación de línea de base por interpolación spline (segmento)')
plt.legend()
plt.grid(True)

#%% ECG limpio (segmento)
plt.figure(figsize=(12,4))
plt.plot(t_segment, ecg_clean_segment, label='ECG limpio (Spline)', color='green')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.title('ECG con línea de base eliminada (Spline) - Segmento')
plt.legend()
plt.grid(True)

plt.show()

#%% SEGMENTO PQ Spline

# Parámetros para segmento PQ
n0 = 80  # muestras (80 ms antes del QRS)
window_size = 20  # muestras (±10 ms)

# Extraer puntos S={(m_i, s(m_i))}
m_values = []
s_values = []

for ni in qrs_detections:
    start_idx = ni - n0 - window_size // 2
    end_idx = ni - n0 + window_size // 2
    
    if start_idx < 0 or end_idx >= len(ecg_one_lead):
        continue
    
    m_i = ni - n0
    s_i = np.mean(ecg_one_lead[start_idx:end_idx])
    
    m_values.append(m_i)
    s_values.append(s_i)

m_values = np.array(m_values)
s_values = np.array(s_values)

# Vector tiempo para toda la señal
N = len(ecg_one_lead)
t = np.arange(N) / fs_ecg  # en segundos

# Tiempo para los puntos m_i
t_m = m_values / fs_ecg

# Interpolación spline cúbica de línea base
spline_func = CubicSpline(t_m, s_values)
baseline_estimate = spline_func(t)

# Graficar un segmento para visualizar claramente el PQ y la línea base

# Elegimos un segmento alrededor de un QRS (por ejemplo el primero válido)
first_idx = m_values[0]
segment_start = first_idx - 150  # 150 ms antes del punto PQ
segment_end = first_idx + 150    # 150 ms después

if segment_start < 0:
    segment_start = 0
if segment_end > N:
    segment_end = N

t_segment = t[segment_start:segment_end]
ecg_segment = ecg_one_lead[segment_start:segment_end]
baseline_segment = baseline_estimate[segment_start:segment_end]

# Puntos fiduciales usados para spline en el segmento
fiducial_mask = (m_values >= segment_start) & (m_values < segment_end)
t_fiducials = t_m[fiducial_mask]
s_fiducials = s_values[fiducial_mask]

plt.figure(figsize=(12,5))
plt.plot(t_segment, ecg_segment, label='ECG original')
plt.plot(t_segment, baseline_segment, color='orange', linewidth=2, label='Línea base (Spline)')
plt.scatter(t_fiducials, s_fiducials, color='red', label='Puntos PQ (nivel isoeléctrico)')
plt.axvline(x=m_values[0]/fs_ecg, color='green', linestyle='--', label='Inicio segmento PQ')

# Graficar líneas rojas para todos los latidos (ni) dentro del segmento
for ni in qrs_detections:
    t_ni = ni / fs_ecg
    if segment_start/fs_ecg <= t_ni <= segment_end/fs_ecg:
        plt.axvline(x=t_ni, color='red', linestyle='-', alpha=0.5)

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.title('Segmento PQ: estimación nivel isoeléctrico y línea base (Spline)')
plt.legend()
plt.grid(True)
plt.show()


# Gráfico centrado en un QRS (Spline)
sample_center = int(qrs_detections[0])
window_qrs = 0.3  # segundos
N_qrs = int(window_qrs * fs_ecg)
start_idx = sample_center - N_qrs // 2
end_idx = sample_center + N_qrs // 2

ecg_segment_qrs = ecg_one_lead[start_idx:end_idx]
baseline_qrs = baseline_full[start_idx:end_idx]
ecg_clean_qrs = ecg_clean_full[start_idx:end_idx]
t_qrs = np.arange(-N_qrs//2, N_qrs//2) / fs_ecg

plt.figure(figsize=(10, 5))
plt.plot(t_qrs, ecg_segment_qrs, label='ECG Original', color='blue')
plt.plot(t_qrs, baseline_qrs, label='Línea de base estimada (Spline)', color='orange')
plt.plot(t_qrs, ecg_clean_qrs, label='ECG Filtrada (Spline)', color='green')
plt.title('Segmento centrado en QRS - Original vs Línea de base vs Filtrado (Spline)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Conclusión método spline
print("""
Conclusión:
La estimación de la línea de base mediante spline cúbico también permite eliminar el movimiento lento
preservando la morfología de la señal ECG. Se observa una pequeña anticipación temporal (~80 ms)
en relación a la señal original, debido a la ubicación fija de los puntos de control en el segmento PQ.
Este método es eficaz para remover oscilaciones lentas sin distorsionar significativamente el complejo QRS.
""")



print("""
Conclusión comparativa:
Ambos métodos permiten estimar y remover eficazmente la línea de base, conservando la morfología del QRS.
- El filtro de mediana en cascada es simple de implementar y robusto ante artefactos aislados.
- El spline cúbico, aunque más costoso computacionalmente, genera una línea de base más suave al ajustarse por puntos fisiológicamente significativos (segmento PQ).
En señales con movimiento lento o interferencias suaves, el spline puede ofrecer una mejor aproximación.
""")
#%% Comparativa de gráficos QRS Mediana y Spline
# Gráfico centrado en un QRS (Mediana)
# Elegimos un QRS para centrar la ventana
i = 0
sample_center = int(qrs_detections[i])

window_qrs = 0.3  # segundos (±150 ms)
N_qrs = int(window_qrs * fs_ecg)

start_idx = sample_center - N_qrs // 2
end_idx = sample_center + N_qrs // 2

# Extraemos los segmentos de la señal original, línea base y señal filtrada
ecg_segment_qrs = ecg_one_lead[start_idx:end_idx]
mediana2_qrs = mediana2[start_idx:end_idx]
ecg_filtrada_qrs = ecg_filtrada[start_idx:end_idx]

# Tiempo relativo centrado en QRS
t_qrs = np.arange(-N_qrs//2, N_qrs//2) / fs_ecg



# Gráfico centrado en un QRS (Spline)
baseline_qrs = baseline_full[start_idx:end_idx]
ecg_clean_qrs = ecg_clean_full[start_idx:end_idx]



# -- Estimación línea base con spline para segmento PQ --
plt.figure(figsize=(12, 6))

# Plot de ECG Original
plt.plot(t_qrs, ecg_segment_qrs, label='ECG Original', color='lightblue')

# Línea de base estimada - Mediana
plt.plot(t_qrs, mediana2_qrs, label='Línea de base estimada (Mediana)', color='purple', linestyle='--')

# ECG filtrada - Mediana
plt.plot(t_qrs, ecg_filtrada_qrs, label='ECG Filtrada (Mediana)', color='magenta')

# Línea de base estimada - Spline
plt.plot(t_qrs, baseline_qrs, label='Línea de base estimada (Spline)', color='green', linestyle='--')

# ECG filtrada - Spline
plt.plot(t_qrs, ecg_clean_qrs, label='ECG Filtrada (Spline)', color='grey')

plt.title('Segmento centrado en QRS - Comparación Mediana vs Spline')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))



#%% Comparativa segmento PQ Mediana y Spline


plt.figure(figsize=(12,6))

# ECG original
plt.plot(t_segment, ecg_segment, label='ECG original', color='blue', linewidth=1.5)

# Línea base interpolada (mediana)
plt.plot(t_segment, linebase_interp, color='orange', linestyle='-', linewidth=2, label='Línea base interpolada (Mediana)')
plt.scatter(t_fiducials, s_fiducials, color='red', label='Puntos PQ (Mediana)', zorder=5)

# Línea base spline
plt.plot(t_segment, baseline_segment, color='green', linestyle='--', linewidth=2, label='Línea base (Spline)')
plt.scatter(t_fiducials, s_fiducials, color='lime', label='Puntos PQ (Spline)', marker='x', zorder=5)

# Señal filtrada por mediana (original menos línea base mediana)
ecg_filtrada_segment = ecg_segment - linebase_interp
plt.plot(t_segment, ecg_filtrada_segment, color='magenta', linewidth=1.5, label='ECG filtrada (Mediana)')

# Señal filtrada por spline (original menos línea base spline)
ecg_filtrada_spline_segment = ecg_segment - baseline_segment
plt.plot(t_segment, ecg_filtrada_spline_segment, color='grey', linewidth=1.5, linestyle='--', label='ECG filtrada (Spline)')

# Línea vertical inicio segmento PQ
plt.axvline(x=m_values[0]/fs_ecg, color='purple', linestyle='--', label='Inicio segmento PQ')

# Líneas verticales QRS dentro del segmento
for ni in qrs_detections:
    t_ni = ni / fs_ecg
    if segment_start/fs_ecg <= t_ni <= segment_end/fs_ecg:
        plt.axvline(x=t_ni, color='red', linestyle='-', alpha=0.5)

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.title('Segmento PQ: Comparación señales filtradas por Mediana y Spline')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Punto 3 - Filtro adaptado (matched filter)
# --- 1. Centrar señal y patrón para eliminar offset ---
ecg_centered = ecg_one_lead - np.mean(ecg_one_lead)
pattern_centered = qrs_pattern[::-1] - np.mean(qrs_pattern)

# --- 2. Filtrado por convolución (matched filter) ---
filtered_signal = np.convolve(ecg_centered, pattern_centered, mode='same')

# --- 3. Detección de picos en señal filtrada ---
height_threshold = np.max(filtered_signal) * 0.4
distance_threshold = int(fs_ecg * 0.3)
peaks, _ = find_peaks(filtered_signal, height=height_threshold, distance=distance_threshold)

# --- 4. Comparación detección con referencia ---
tolerance = int(0.05 * fs_ecg)
TP = 0
FP = 0
detected = np.zeros(len(qrs_detections), dtype=bool)

for p in peaks:
    match = False
    for i, qrs in enumerate(qrs_detections):
        if not detected[i] and abs(p - qrs) <= tolerance:
            TP += 1
            detected[i] = True
            match = True
            break
    if not match:
        FP += 1

FN = len(qrs_detections) - TP
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

print(f"Detección por filtro adaptado:")
print(f"  Verdaderos Positivos (TP): {TP}")
print(f"  Falsos Positivos (FP): {FP}")
print(f"  Falsos Negativos (FN): {FN}")
print(f"  Sensibilidad (Recall): {sensitivity*100:.2f}%")
print(f"  Valor Predictivo Positivo (Precision): {precision*100:.2f}%")

# --- 5. Visualización resultados detección ---
plt.figure(figsize=(12, 4))
plt.plot(ecg_one_lead, label='ECG original')
plt.plot(filtered_signal / np.max(filtered_signal), label='Filtro adaptado (normalizado)')
plt.plot(peaks, ecg_one_lead[peaks], 'rx', label='Detecciones')
plt.plot(qrs_detections, ecg_one_lead[qrs_detections], 'go', label='QRS referencia')
plt.legend()
plt.title("Detección de latidos con filtro adaptado")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

#---Segmento del filtro adaptado 
# Defino el segmento
seg_start = 0
seg_end = 745000
ecg_segment = ecg_one_lead[seg_start:seg_end]


filtered_segment = filtered_signal[seg_start:seg_end] / np.max(filtered_signal)

# Ahora, los picos y QRS que caen dentro del segmento, reindexados
peaks_segment_idx = peaks[(peaks >= seg_start) & (peaks < seg_end)] - seg_start
qrs_segment_idx = qrs_detections[(qrs_detections >= seg_start) & (qrs_detections < seg_end)] - seg_start

# Graficás
plt.figure(figsize=(12, 4))
plt.plot(ecg_segment, label='ECG original')
plt.plot(filtered_segment, label='Filtro adaptado (normalizado)')
plt.plot(peaks_segment_idx, ecg_segment[peaks_segment_idx], 'rx', label='Detecciones')
plt.plot(qrs_segment_idx, ecg_segment[qrs_segment_idx], 'go', label='QRS referencia')
plt.legend()
plt.title("Detección de latidos con filtro adaptado (segmento)")
plt.xlabel("Muestras (segmento)")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()


# --- 7. Cálculo de error medio entre señal y filtro adaptado en segmento (opcional) ---
mean_error = np.mean(filtered_segment - ecg_segment)
print(f"Error medio (filtered_segment - ecg_segment): {mean_error:.5f}")

# Zoom en un segmento para visualización clara
start = int(30 * fs_ecg)
end = int(35 * fs_ecg)

plt.figure(figsize=(12, 4))
plt.plot(ecg_one_lead[start:end], label='ECG original')
plt.plot(filtered_signal[start:end] / np.max(filtered_signal), label='Filtro adaptado (normalizado)')
plt.plot(peaks[(peaks >= start) & (peaks < end)] - start, 
         ecg_one_lead[peaks[(peaks >= start) & (peaks < end)]], 'rx', label='Detecciones')
plt.legend()
plt.title("Zoom: Detección de latidos con filtro adaptado")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.grid(True)
plt.show()