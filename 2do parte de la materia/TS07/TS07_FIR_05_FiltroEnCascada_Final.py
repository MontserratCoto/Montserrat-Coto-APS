import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla

plt.close('all')

#%% Lectura de ECG
fs_ecg = 1000
nyq_frec = fs_ecg / 2
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()

# Normalizar señal
ecg_one_lead = ecg_one_lead / np.std(ecg_one_lead)
# Variables del archivo
qrs_pattern = mat_struct['qrs_pattern1'].flatten()
heartbeat_normal = mat_struct['heartbeat_pattern1'].flatten()
heartbeat_ventricular = mat_struct['heartbeat_pattern2'].flatten()
qrs_locations = mat_struct['qrs_detections'].flatten()

# Normalizar señal
ecg_one_lead = ecg_one_lead / np.std(ecg_one_lead)
heartbeat_normal = heartbeat_normal / np.std(heartbeat_normal)
heartbeat_ventricular = heartbeat_ventricular / np.std(heartbeat_ventricular)
qrs_pattern = qrs_pattern /np.std(qrs_pattern)


#%% Parámetros del filtro
cant_coef = 781  # suficiente para buenas transiciones
f_c_hp = 0.7
f_c_lp = 35.0

# Normalizadas
f_c_hp_norm = f_c_hp / nyq_frec
f_c_lp_norm = f_c_lp / nyq_frec

# Filtro pasa alto (elimina muy bajas frecuencias)
b_hp = sig.firwin(numtaps=cant_coef, cutoff=f_c_hp_norm, window=('kaiser', 10), pass_zero=False)

# Filtro pasa bajo (elimina altas frecuencias)
b_lp = sig.firwin(numtaps=cant_coef, cutoff=f_c_lp_norm, window=('kaiser', 10), pass_zero=True)

# Convolución: pasabanda = pasaaltos * pasabajos
b_band = np.convolve(b_hp, b_lp)
b_band /= np.sum(b_band)  # normalización para no alterar amplitud

# Aplicación del filtro
ecg_filtrada = sig.filtfilt(b_band, 1, ecg_one_lead)
ecg_filtrada = ecg_filtrada / np.std(ecg_filtrada)

# Para compatibilidad con el código viejo
ecg_filtrado_kaiser = ecg_filtrada.copy()

#%% Visualización de la respuesta en frecuencia
f_band, h_band = sig.freqz(b_band, worN=4096, fs=fs_ecg)

plt.figure()
plt.plot(f_band, 20 * np.log10(np.abs(h_band) + 1e-10), label='Filtro FIR Pasabanda (ventana)')
plt.title('Respuesta en frecuencia del filtro FIR Pasabanda')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.ylim([-80, 5])
plt.grid(True)

# Plantilla
plot_plantilla(
    filter_type='bandpass',
    fpass=np.array([0.7, 35.0]),
    fstop=np.array([0.5, 50.0]),
    ripple=0.5,
    attenuation=40,
    fs=fs_ecg
)

plt.legend()
plt.tight_layout()
plt.show()

#%% Visualización señal original vs filtrada (primeros 2 segundos)
plt.figure()
plt.plot(ecg_one_lead[:2000], label='Original')
plt.plot(ecg_filtrada[:2000], label='Filtrada')
plt.legend()
plt.title("Comparación entre señal original y filtrada")
plt.grid(True)
plt.show()

#%% Visualización de ondas P, QRS y T en un segmento
#%% Selección del latido a visualizar
i = 0  # Índice del latido (0 = primer latido detectado)
sample_center = int(qrs_locations[i])

#%% Parámetros de ventanas para cada onda (en segundos)
p_start_sec = -0.2   # 200 ms antes del QRS
p_end_sec   = -0.1   # 100 ms antes del QRS

qrs_start_sec = -0.04  # 40 ms antes
qrs_end_sec   =  0.04  # 40 ms después

t_start_sec =  0.15  # 150 ms después del QRS
t_end_sec   =  0.35  # 350 ms después del QRS

#%% Conversión a índices de muestra
p_start = sample_center + int(p_start_sec * fs_ecg)
p_end   = sample_center + int(p_end_sec * fs_ecg)

qrs_start = sample_center + int(qrs_start_sec * fs_ecg)
qrs_end   = sample_center + int(qrs_end_sec * fs_ecg)

t_start = sample_center + int(t_start_sec * fs_ecg)
t_end   = sample_center + int(t_end_sec * fs_ecg)

#%% Extraer segmento completo desde inicio P hasta final T
start_segment = p_start
end_segment = t_end

# Validación de límites
if start_segment < 0 or end_segment >= len(ecg_one_lead):
    raise ValueError("El segmento se va fuera del rango de la señal.")

ecg_segment_full = ecg_one_lead[start_segment:end_segment]
ecg_filtrada_full = ecg_filtrado_kaiser[start_segment:end_segment]
t_segment_full = np.arange(start_segment, end_segment) / fs_ecg

#%% Graficar segmento completo con sombreado de ondas
plt.figure(figsize=(12, 5))
plt.plot(t_segment_full, ecg_segment_full, label='ECG Original', color='blue')
plt.plot(t_segment_full, ecg_filtrada_full, label='ECG Filtrada (FIR - Kaiser)', color='green')

# Sombras para ondas
plt.axvspan(p_start / fs_ecg, p_end / fs_ecg, color='pink', alpha=0.3, label='Onda P')
plt.axvspan(qrs_start / fs_ecg, qrs_end / fs_ecg, color='orange', alpha=0.3, label='Complejo QRS')
plt.axvspan(t_start / fs_ecg, t_end / fs_ecg, color='purple', alpha=0.3, label='Onda T')

plt.title('Segmento con ondas P, QRS y T (Filtro FIR - Kaiser)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% Segmento QRS centrado para visualización ampliada
window_qrs_sec = 0.3  # Ventana total: ±150 ms
N_qrs = int(window_qrs_sec * fs_ecg)

# Validación de bordes
start_qrs = sample_center - N_qrs // 2
end_qrs = sample_center + N_qrs // 2
if start_qrs < 0 or end_qrs >= len(ecg_one_lead):
    raise ValueError("El segmento QRS se va fuera del rango de la señal.")

ecg_segment_qrs = ecg_one_lead[start_qrs:end_qrs]
ecg_segment_qrs_filt = ecg_filtrado_kaiser[start_qrs:end_qrs]
t_qrs = np.arange(-N_qrs//2, N_qrs//2) / fs_ecg

plt.figure(figsize=(10,5))
plt.plot(t_qrs, ecg_segment_qrs, label='QRS Original')
plt.plot(t_qrs, ecg_segment_qrs_filt, label='QRS Filtrado (FIR - Kaiser)', alpha=0.7)
plt.title('Segmento centrado en QRS - Original vs Filtrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
