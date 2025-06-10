# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:56:25 2025

@author: monse
"""


import numpy as np
import scipy.io as sio
import scipy.signal as sig
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
from scipy.signal import sosfreqz, group_delay
from pytc2.sistemas_lineales import bodePlot, plot_plantilla
from scipy.signal import sosfreqz

plt.close('all')

#%% Lectura de ECG

fs_ecg = 1000  # Frecuencia de muestreo en Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
nyq_frec = fs_ecg / 2  # Frecuencia de Nyquist

# Variables del archivo
qrs_pattern = mat_struct['qrs_pattern1'].flatten()
heartbeat_normal = mat_struct['heartbeat_pattern1'].flatten()
heartbeat_ventricular = mat_struct['heartbeat_pattern2'].flatten()
qrs_locations = mat_struct['qrs_detections'].flatten()


ecg_one_lead = ecg_one_lead / np.std(ecg_one_lead)
heartbeat_normal = heartbeat_normal / np.std(heartbeat_normal)
heartbeat_ventricular = heartbeat_ventricular / np.std(heartbeat_ventricular)
qrs_pattern = qrs_pattern /np.std(qrs_pattern)

#%% Visualización señal completa

plt.figure(figsize=(12, 4))
t_total = np.arange(len(ecg_one_lead)) / fs_ecg
plt.plot(t_total, ecg_one_lead, label='ECG crudo')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal ECG completa')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% Visualización latido típico

i = 0
sample_center = int(qrs_locations[i])
window_size = 0.6  # segundos (±300 ms)
N = int(window_size * fs_ecg)

ecg_segment = ecg_one_lead[sample_center - N // 2 : sample_center + N // 2]
ecg_segment = ecg_segment / np.std(ecg_segment)
t_segment = np.arange(-N//2, N//2) / fs_ecg

# plt.figure(figsize=(10, 4))
# plt.plot(t_segment, ecg_segment, label='Latido típico')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.title(f'Zoom sobre el latido #{i + 1}')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

#%% Visualización latidos promedio y QRS

# plt.figure(figsize=(10, 4))
# plt.plot(np.arange(len(heartbeat_normal)) / fs_ecg, heartbeat_normal, label='Latido normal')
# plt.plot(np.arange(len(heartbeat_ventricular)) / fs_ecg, heartbeat_ventricular, label='Latido ventricular', alpha=0.7)
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.title('Latidos promedio')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6, 3))
# plt.plot(np.arange(len(qrs_pattern)) / fs_ecg, qrs_pattern, label='QRS pattern')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.title('Onda QRS promedio')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()





#%% Diseño del filtro digital (punto a)
#IIR

filter_type = 'bandpass'


# # ----------- Aproximación de máxima planicidad (Butterworth) -----------
# # Descomentar esta línea para usar Butterworth (máxima planicidad)
aprox_name = 'butter'
mi_sos_butter = sig.iirdesign(
    wp=[0.5, 30],
    ws=[0.1, 50],
    gpass=0.5,
    gstop=40,
    ftype=aprox_name,
    output='sos',
    fs=fs_ecg
)
fpass = np.array([0.5, 30])      # Banda de paso
fstop = np.array([0.1, 50])      # Banda de detención
ripple = 0.5                     # Rizado en banda de paso (dB)
attenuation = 40                # Atenuación en banda de detención (dB)


# Diseño del filtro (estructura en SOS)
mi_sos_butter = sig.iirdesign(
    wp=fpass,
    ws=fstop,
    gpass=ripple,
    gstop=attenuation,
    ftype=aprox_name,
    output='sos',
    fs=fs_ecg
)


ecg_filtrada_butter = sig.sosfiltfilt(mi_sos_butter, ecg_segment)
ecg_filtrada_butter = ecg_filtrada_butter / np.std(ecg_filtrada_butter)

#%% Visualización de plantilla y respuesta en frecuencia

f_low = np.linspace(0.01, 0.4, 300)
f_fine = np.linspace(0.4, 0.6, 500)
f_high = np.linspace(0.6, nyq_frec, 700)
f_total = np.concatenate((f_low, f_fine, f_high))

w_rad = f_total / nyq_frec * np.pi
w, hh = sosfreqz(mi_sos_butter, worN=w_rad)

plt.figure()
plt.plot(w / np.pi * nyq_frec, 20 * np.log10(np.abs(hh) + 1e-15),
         label=f'Respuesta del filtro (Butter)')
plt.title(f'Plantilla del filtro digital para ECG (Butter)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(True)
plt.ylim([-80, 5])

plot_plantilla(
    filter_type=filter_type,
    fpass=fpass,
    ripple=ripple,
    fstop=fstop,
    attenuation=attenuation,
    fs=fs_ecg
)

plt.legend()
plt.show()

#%% Aplicación del filtro al segmento


plt.figure(figsize=(10, 5))
plt.plot(t_segment, ecg_segment, label='ECG Original', color='blue')
plt.plot(t_segment, ecg_filtrada_butter,
         label=f'ECG Filtrada (Butter)', color='green')
plt.title(f'ECG Original vs Filtrada (Butter)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% Aplicación del filtro justo sobre el QRS

# Centrado en el mismo latido de antes
i = 0
sample_center = int(qrs_locations[i])

# Nueva ventana más ajustada: ±150 ms
window_qrs = 0.3  # segundos
N_qrs = int(window_qrs * fs_ecg)

# Extracción del segmento centrado en el QRS
ecg_segment_qrs = ecg_one_lead[sample_center - N_qrs // 2 : sample_center + N_qrs // 2]
t_qrs = np.arange(-N_qrs//2, N_qrs//2) / fs_ecg

# Filtrado
ecg_filtrada_butter_qrs = sig.sosfiltfilt(mi_sos_butter, ecg_segment_qrs)

# Gráfico
plt.figure(figsize=(10, 5))
plt.plot(t_qrs, ecg_segment_qrs, label='ECG Original', color='blue')
plt.plot(t_qrs, ecg_filtrada_butter_qrs, label=f'ECG Filtrada (Butter)', color='green')
plt.title(f'Segmento centrado en QRS - Original vs Filtrada (Butter)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%% Visualización segmento con ondas P, QRS y T

# Parámetros para ventanas de las ondas (en segundos)
p_start_sec = -0.2  # 200 ms antes del QRS (inicio P)
p_end_sec = -0.1    # 100 ms antes del QRS (fin P)

qrs_start_sec = -0.04  # 40 ms antes del QRS (inicio QRS)
qrs_end_sec = 0.04     # 40 ms después del QRS (fin QRS)

t_start_sec = 0.15    # 150 ms después del QRS (inicio T)
t_end_sec = 0.35      # 350 ms después del QRS (fin T)

# Muestras relativas
sample_center = int(qrs_locations[0])
p_start = sample_center + int(p_start_sec * fs_ecg)
p_end = sample_center + int(p_end_sec * fs_ecg)

qrs_start = sample_center + int(qrs_start_sec * fs_ecg)
qrs_end = sample_center + int(qrs_end_sec * fs_ecg)

t_start = sample_center + int(t_start_sec * fs_ecg)
t_end = sample_center + int(t_end_sec * fs_ecg)

# Extraemos un segmento amplio para visualizar
start_segment = p_start
end_segment = t_end

ecg_segment_full = ecg_one_lead[start_segment:end_segment]
t_segment_full = np.arange(start_segment, end_segment) / fs_ecg
ecg_filtrada_butter_full = sig.sosfiltfilt(mi_sos_butter, ecg_segment_full)

plt.figure(figsize=(12, 5))
plt.plot(t_segment_full, ecg_segment_full, label='ECG Original', color='blue')
plt.plot(t_segment_full, ecg_filtrada_butter_full, label='ECG Filtrada', color='green')

# Sombreado de las ondas P, QRS y T
plt.axvspan(p_start / fs_ecg, p_end / fs_ecg, color='pink', alpha=0.3, label='Onda P')
plt.axvspan(qrs_start / fs_ecg, qrs_end / fs_ecg, color='orange', alpha=0.3, label='Complejo QRS')
plt.axvspan(t_start / fs_ecg, t_end / fs_ecg, color='purple', alpha=0.3, label='Onda T')

plt.title('Segmento con ondas P, QRS y T (Butter)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% Filtrado de latidos promedio

# heartbeat_normal_butter_filt = sig.sosfiltfilt(mi_sos_butter, heartbeat_normal)
# heartbeat_ventricular_butter_filt = sig.sosfiltfilt(mi_sos_butter, heartbeat_ventricular)

# t_normal = np.arange(len(heartbeat_normal)) / fs_ecg
# t_vent = np.arange(len(heartbeat_ventricular)) / fs_ecg

# plt.figure(figsize=(12, 5))

# # Latido normal sin filtrar vs filtrado
# plt.subplot(1, 2, 1)
# plt.plot(t_normal, heartbeat_normal, label='Normal sin filtrar')
# plt.plot(t_normal, heartbeat_normal_butter_filt, label='Normal filtrado', alpha=0.7)
# plt.title('Latido Normal: Sin filtrar vs Filtrado (Butter)')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.legend()
# plt.grid(True)

# # Latido ventricular sin filtrar vs filtrado
# plt.subplot(1, 2, 2)
# plt.plot(t_vent, heartbeat_ventricular, label='Ventricular sin filtrar')
# plt.plot(t_vent, heartbeat_ventricular_butter_filt, label='Ventricular filtrado', alpha=0.7)
# plt.title('Latido Ventricular: Sin filtrar vs Filtrado (Butter)')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# ----------- Aproximación de Cauer (Elíptico) -----------
# Descomentar esta línea para usar Elíptico (Cauer, mínima orden)
aprox_name = 'ellip'
mi_sos_ellip = sig.iirdesign(
    wp=[0.5, 30],
    ws=[0.1, 50],
    gpass=0.5,
    gstop=40,
    ftype=aprox_name,
    output='sos',
    fs=fs_ecg
)

fpass = np.array([0.5, 30])      # Banda de paso
fstop = np.array([0.1, 50])      # Banda de detención
ripple = 0.5                     # Rizado en banda de paso (dB)
attenuation = 40                # Atenuación en banda de detención (dB)

# Diseño del filtro (estructura en SOS)
mi_sos_ellip = sig.iirdesign(
    wp=fpass,
    ws=fstop,
    gpass=ripple,
    gstop=attenuation,
    ftype=aprox_name,
    output='sos',
    fs=fs_ecg
)


ecg_filtrada_ellip = sig.sosfiltfilt(mi_sos_ellip, ecg_segment)
ecg_filtrada_ellip = ecg_filtrada_ellip / np.std(ecg_filtrada_ellip)

#%% Visualización de plantilla y respuesta en frecuencia

f_low = np.linspace(0.01, 0.4, 300)
f_fine = np.linspace(0.4, 0.6, 500)
f_high = np.linspace(0.6, nyq_frec, 700)
f_total = np.concatenate((f_low, f_fine, f_high))

w_rad = f_total / nyq_frec * np.pi
w, hh = sosfreqz(mi_sos_ellip, worN=w_rad)

plt.figure()
plt.plot(w / np.pi * nyq_frec, 20 * np.log10(np.abs(hh) + 1e-15),
         label=f'Respuesta del filtro (Cauer)')
plt.title(f'Plantilla del filtro digital para ECG (Cauer)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(True)
plt.ylim([-80, 5])

plot_plantilla(
    filter_type=filter_type,
    fpass=fpass,
    ripple=ripple,
    fstop=fstop,
    attenuation=attenuation,
    fs=fs_ecg
)

plt.legend()
plt.show()

#%% Aplicación del filtro al segmento


plt.figure(figsize=(10, 5))
plt.plot(t_segment, ecg_segment, label='ECG Original', color='blue')
plt.plot(t_segment, ecg_filtrada_ellip,
         label=f'ECG Filtrada (Cauer)', color='green')
plt.title(f'ECG Original vs Filtrada (Cauer)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% Aplicación del filtro justo sobre el QRS

# Centrado en el mismo latido de antes
i = 0
sample_center = int(qrs_locations[i])

# Nueva ventana más ajustada: ±150 ms
window_qrs = 0.3  # segundos
N_qrs = int(window_qrs * fs_ecg)

# Extracción del segmento centrado en el QRS
ecg_segment_qrs = ecg_one_lead[sample_center - N_qrs // 2 : sample_center + N_qrs // 2]
t_qrs = np.arange(-N_qrs//2, N_qrs//2) / fs_ecg

# Filtrado
ecg_filtrada_ellip_qrs = sig.sosfiltfilt(mi_sos_ellip, ecg_segment_qrs)

# Gráfico
plt.figure(figsize=(10, 5))
plt.plot(t_qrs, ecg_segment_qrs, label='ECG Original', color='blue')
plt.plot(t_qrs, ecg_filtrada_ellip_qrs, label=f'ECG Filtrada (Cauer)', color='green')
plt.title(f'Segmento centrado en QRS - Original vs Filtrada (Cauer)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%% Visualización segmento con ondas P, QRS y T

# Parámetros para ventanas de las ondas (en segundos)
p_start_sec = -0.2  # 200 ms antes del QRS (inicio P)
p_end_sec = -0.1    # 100 ms antes del QRS (fin P)

qrs_start_sec = -0.04  # 40 ms antes del QRS (inicio QRS)
qrs_end_sec = 0.04     # 40 ms después del QRS (fin QRS)

t_start_sec = 0.15    # 150 ms después del QRS (inicio T)
t_end_sec = 0.35      # 350 ms después del QRS (fin T)

# Muestras relativas
sample_center = int(qrs_locations[0])
p_start = sample_center + int(p_start_sec * fs_ecg)
p_end = sample_center + int(p_end_sec * fs_ecg)

qrs_start = sample_center + int(qrs_start_sec * fs_ecg)
qrs_end = sample_center + int(qrs_end_sec * fs_ecg)

t_start = sample_center + int(t_start_sec * fs_ecg)
t_end = sample_center + int(t_end_sec * fs_ecg)

# Extraemos un segmento amplio para visualizar
start_segment = p_start
end_segment = t_end

ecg_segment_full = ecg_one_lead[start_segment:end_segment]
t_segment_full = np.arange(start_segment, end_segment) / fs_ecg
ecg_filtrada_ellip_full = sig.sosfiltfilt(mi_sos_ellip, ecg_segment_full)

plt.figure(figsize=(12, 5))
plt.plot(t_segment_full, ecg_segment_full, label='ECG Original', color='blue')
plt.plot(t_segment_full, ecg_filtrada_ellip_full, label='ECG Filtrada', color='green')

# Sombreado de las ondas P, QRS y T
plt.axvspan(p_start / fs_ecg, p_end / fs_ecg, color='pink', alpha=0.3, label='Onda P')
plt.axvspan(qrs_start / fs_ecg, qrs_end / fs_ecg, color='orange', alpha=0.3, label='Complejo QRS')
plt.axvspan(t_start / fs_ecg, t_end / fs_ecg, color='purple', alpha=0.3, label='Onda T')

plt.title('Segmento con ondas P, QRS y T (Cauer)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% Filtrado de latidos promedio

# heartbeat_normal_ellip_filt = sig.sosfiltfilt(mi_sos_ellip, heartbeat_normal)
# heartbeat_ventricular_ellip_filt = sig.sosfiltfilt(mi_sos_ellip, heartbeat_ventricular)

# t_normal = np.arange(len(heartbeat_normal)) / fs_ecg
# t_vent = np.arange(len(heartbeat_ventricular)) / fs_ecg

# plt.figure(figsize=(12, 5))

# # Latido normal sin filtrar vs filtrado
# plt.subplot(1, 2, 1)
# plt.plot(t_normal, heartbeat_normal, label='Normal sin filtrar')
# plt.plot(t_normal, heartbeat_normal_ellip_filt, label='Normal filtrado', alpha=0.7)
# plt.title('Latido Normal: Sin filtrar vs Filtrado (Cauer)')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.legend()
# plt.grid(True)

# # Latido ventricular sin filtrar vs filtrado
# plt.subplot(1, 2, 2)
# plt.plot(t_vent, heartbeat_ventricular, label='Ventricular sin filtrar')
# plt.plot(t_vent, heartbeat_ventricular_ellip_filt, label='Ventricular filtrado', alpha=0.7)
# plt.title('Latido Ventricular: Sin filtrar vs Filtrado (Cauer)')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()


#%% Diseño de filtros con FIR - Ventanas

# Parámetros de la plantilla del filtro 
fpass = np.array([1.3, 35])      # Banda de paso
fs_ecgtop = np.array([0.1, 50])  # Banda de detención
ripple = 0.5                    # Rizado en banda de paso (dB)
attenuation = 40                # Atenuación en banda de detención (dB)

#%% Diseño de filtro con firwin2

cant_coef = 2001    # cantidad de coeficientes (orden + 1), ideal impar

nyq = fs_ecg / 2        # frecuencia de Nyquist

# Definimos los puntos de frecuencia y ganancia
#freq_hz = [0.0, 0.5, 1, 40.0, 50.0, nyq] 
freq_hz =[0.0, 0.1, 1, 35.0, 50.0, nyq]  # en Hz
gain =    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]   # ganancia deseada en cada punto

# Normalizar las frecuencias
freq_norm = [f / nyq for f in freq_hz]

# Diseñar el filtro con firwin2
Windows_fir = sig.firwin2(
    numtaps=cant_coef,
    freq=freq_norm,
    gain=gain,
    window=('kaiser', 6)
)

aprox_name = 'FIR-Kaiser' 

f_low = np.linspace(0.01, 0.4, 300)
f_fine = np.linspace(0.4, 0.6, 500)
f_high = np.linspace(0.6, nyq_frec, 700)
f_total = np.concatenate((f_low, f_fine, f_high))

w_rad = f_total / nyq_frec * np.pi
w, hh = sig.freqz(Windows_fir, worN=w_rad)

plt.figure()
plt.plot(w / np.pi * nyq_frec, 20 * np.log10(np.abs(hh) + 1e-15),
         label=f'Respuesta del filtro ({aprox_name})')
plt.title(f'Plantilla del filtro digital para ECG ({aprox_name})')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(True)
plt.ylim([-80, 5])

plot_plantilla(
    filter_type='bandpass',
    fpass=fpass,
    ripple=ripple,
    fstop=fs_ecgtop,
    attenuation=attenuation,
    fs=fs_ecg
)

plt.legend()
plt.show()

#%% Diseño de FIR con cuadrados mínimos

from pytc2.filtros_digitales import fir_design_ls

# Parámetros

fs_ecgtop = np.array([0.5, 50])    # Banda de detención
fpass = np.array([2, 35])          # Banda de paso
ripple = 2                        # dB (banda de paso)
attenuation = 30                  # dB (banda de detención)

# Frecuencias normalizadas (por Nyquist = fs_ecg/2)
fn = fs_ecg / 2
Be = [
    0.0, fs_ecgtop[0]/fn,           # detención baja
    fpass[0]/fn, fpass[1]/fn,       # paso
    fs_ecgtop[1]/fn, 1.0            # detención alta
]

# Respuesta deseada en cada banda
D = [0, 0, 1, 1, 0, 0]

# Peso relativo (convertido de dB aproximado)
W = [10**(attenuation/20), 1, 10**(attenuation/20)]  # enfatiza la banda de paso

# Estimamos orden (puedes refinar esto)
N = 207 # orden del filtro (ajustable)

# Diseño del filtro
lsq_fir = fir_design_ls(order=N, band_edges=Be, desired=D, weight=W, filter_type='m', grid_density=16)

# Evaluamos FFT
fft_sz = 4096
H = np.fft.fft(lsq_fir, fft_sz)
frecuencias = np.linspace(0, fn, fft_sz//2)

# Graficar
plt.figure(figsize=(10, 5))
plt.plot(frecuencias, 20*np.log10(np.abs(H[:fft_sz//2]) + 1e-8), label='Filtro FIR LS')
plt.title("Respuesta en Frecuencia del Filtro FIR Pasabanda")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.ylim([-80, 5])
plt.grid(True)
plt.legend()
plt.tight_layout()

plot_plantilla(
    filter_type='bandpass',
    fpass=fpass,
    ripple=ripple,
    fstop=fs_ecgtop,
    attenuation=attenuation,
    fs=fs_ecg
)

plt.title(f"Filtro FIR Pasa Banda - Orden {N}")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#%% Aplicación de filtros

# Filtramos la señal con ambos métodos
ecg_filtrado_kaiser = sig.lfilter(Windows_fir, 1, ecg_one_lead)
ecg_filtrado_ls = sig.lfilter(lsq_fir, 1, ecg_one_lead)

t = np.arange(len(ecg_one_lead)) /fs_ecg

# # Graficar señal original y filtradas
# plt.figure(figsize=(15,6))
# plt.plot(t, ecg_one_lead, label='ECG Original', alpha=0.7)
# plt.plot(t, ecg_filtrado_kaiser, label='ECG Filtrado Kaiser')
# plt.plot(t, ecg_filtrado_ls, label='ECG Filtrado LSQ')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.title('Señal ECG original y señales filtradas')
# plt.legend()
# plt.grid(True)
# plt.show()

# Aplicación de filtros sin retardo con filtfilt
ecg_filtrado_kaiser = sig.filtfilt(Windows_fir, [1], ecg_one_lead)
ecg_filtrado_kaiser=ecg_filtrado_kaiser/np.std(ecg_filtrado_kaiser)

ecg_filtrado_ls = sig.filtfilt(lsq_fir, [1], ecg_one_lead)
ecg_filtrado_ls=ecg_filtrado_ls/np.std(ecg_filtrado_ls)

# # Gráfico comparativo
# plt.figure(figsize=(12, 5))
# plt.plot(t, ecg_one_lead, label='ECG Original', color='gray', alpha=0.6)
# plt.plot(t, ecg_filtrado_kaiser, label='Filtrado Kaiser (filtfilt)', color='blue')
# plt.plot(t, ecg_filtrado_ls, label='Filtrado LSQ (filtfilt)', color='green')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.title('ECG Original vs Filtrado (sin retardo)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


#%% Comparativa señales originales vs filtradas: FIR Kaiser
# Visualización segmento con ondas P, QRS y T usando filtro FIR

# Parámetros para ventanas de las ondas (en segundos)
p_start_sec = -0.2  # 200 ms antes del QRS (inicio P)
p_end_sec = -0.1    # 100 ms antes del QRS (fin P)

qrs_start_sec = -0.04  # 40 ms antes del QRS (inicio QRS)
qrs_end_sec = 0.04     # 40 ms después del QRS (fin QRS)

t_start_sec = 0.15    # 150 ms después del QRS (inicio T)
t_end_sec = 0.35      # 350 ms después del QRS (fin T)

# Ubicación del primer QRS
sample_center = int(qrs_locations[0])

# Cálculo de índices en muestras
p_start = sample_center + int(p_start_sec * fs_ecg)
p_end = sample_center + int(p_end_sec * fs_ecg)

qrs_start = sample_center + int(qrs_start_sec * fs_ecg)
qrs_end = sample_center + int(qrs_end_sec * fs_ecg)

t_start = sample_center + int(t_start_sec * fs_ecg)
t_end = sample_center + int(t_end_sec * fs_ecg)

# Extraemos segmento completo para visualizar
start_segment = p_start
end_segment = t_end

ecg_segment_full = ecg_one_lead[start_segment:end_segment]
t_segment_full = np.arange(start_segment, end_segment) / fs_ecg

# Aplicar filtro FIR con lfilter
ecg_filtrada_full = ecg_filtrado_kaiser[start_segment:end_segment]

plt.figure(figsize=(12, 5))
plt.plot(t_segment_full, ecg_segment_full, label='ECG Original', color='blue')
plt.plot(t_segment_full, ecg_filtrada_full, label='ECG Filtrada (FIR)', color='green')

# Sombreado de las ondas P, QRS y T
plt.axvspan(p_start / fs_ecg, p_end / fs_ecg, color='pink', alpha=0.3, label='Onda P')
plt.axvspan(qrs_start / fs_ecg, qrs_end / fs_ecg, color='orange', alpha=0.3, label='Complejo QRS')
plt.axvspan(t_start / fs_ecg, t_end / fs_ecg, color='purple', alpha=0.3, label='Onda T')

plt.title('Segmento con ondas P, QRS y T (Filtro FIR - Ventanas (Kaiser))')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# QRS
# Índice del latido a analizar
i = 0
sample_center = int(qrs_locations[i])

# Ventana centrada ±150 ms
window_qrs = 0.3  # segundos
N_qrs = int(window_qrs * fs_ecg)

# Segmento QRS original (sin filtrar)
ecg_segment_qrs = ecg_one_lead[sample_center - N_qrs // 2 : sample_center + N_qrs // 2]

# Segmento QRS filtrado, extraído de la señal filtrada completa
ecg_segment_qrs_filt = ecg_filtrado_kaiser[sample_center - N_qrs // 2 : sample_center + N_qrs // 2]

# Tiempo para la ventana QRS
t_qrs = np.arange(-N_qrs//2, N_qrs//2) / fs_ecg

# Graficar ambos
plt.figure(figsize=(10,5))
plt.plot(t_qrs, ecg_segment_qrs, label='Segmento QRS Original')
plt.plot(t_qrs, ecg_segment_qrs_filt, label='Segmento QRS Filtrado (Filtro FIR - Ventanas (Kaiser))', alpha=0.7)
plt.title(f'Segmento QRS Filtrado (Filtro FIR - Ventanas (Kaiser))')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()

#%% Comparativa señales originales vs filtradas: FIR LSQ
#%% Visualización segmento con ondas P, QRS y T usando filtro FIR LSQ

# Parámetros para ventanas de las ondas (en segundos)
p_start_sec = -0.2  # 200 ms antes del QRS (inicio P)
p_end_sec = -0.1    # 100 ms antes del QRS (fin P)

qrs_start_sec = -0.04  # 40 ms antes del QRS (inicio QRS)
qrs_end_sec = 0.04     # 40 ms después del QRS (fin QRS)

t_start_sec = 0.15    # 150 ms después del QRS (inicio T)
t_end_sec = 0.35      # 350 ms después del QRS (fin T)

# Ubicación del primer QRS
sample_center = int(qrs_locations[0])

# Cálculo de índices en muestras
p_start = sample_center + int(p_start_sec * fs_ecg)
p_end = sample_center + int(p_end_sec * fs_ecg)

qrs_start = sample_center + int(qrs_start_sec * fs_ecg)
qrs_end = sample_center + int(qrs_end_sec * fs_ecg)

t_start = sample_center + int(t_start_sec * fs_ecg)
t_end = sample_center + int(t_end_sec * fs_ecg)

# Extraemos segmento completo para visualizar
start_segment = p_start
end_segment = t_end

ecg_segment_full = ecg_one_lead[start_segment:end_segment]
t_segment_full = np.arange(start_segment, end_segment) / fs_ecg

# Aplicar filtro FIR con lfilter (ya calculado anteriormente)
ecg_filtrada_full_ls = ecg_filtrado_ls[start_segment:end_segment]

plt.figure(figsize=(12, 5))
plt.plot(t_segment_full, ecg_segment_full, label='ECG Original', color='blue')
plt.plot(t_segment_full, ecg_filtrada_full_ls, label='ECG Filtrada (FIR LSQ)', color='green')

# Sombreado de las ondas P, QRS y T
plt.axvspan(p_start / fs_ecg, p_end / fs_ecg, color='pink', alpha=0.3, label='Onda P')
plt.axvspan(qrs_start / fs_ecg, qrs_end / fs_ecg, color='orange', alpha=0.3, label='Complejo QRS')
plt.axvspan(t_start / fs_ecg, t_end / fs_ecg, color='purple', alpha=0.3, label='Onda T')

plt.title('Segmento con ondas P, QRS y T (Filtro FIR - Cuadrados Mínimos)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% QRS
# Índice del latido a analizar
i = 0
sample_center = int(qrs_locations[i])

# Ventana centrada ±150 ms
window_qrs = 0.3  # segundos
N_qrs = int(window_qrs * fs_ecg)

# Segmento QRS original (sin filtrar)
ecg_segment_qrs = ecg_one_lead[sample_center - N_qrs // 2 : sample_center + N_qrs // 2]

# Segmento QRS filtrado, extraído de la señal filtrada completa (LSQ)
ecg_segment_qrs_filt = ecg_filtrado_ls[sample_center - N_qrs // 2 : sample_center + N_qrs // 2]

# Tiempo para la ventana QRS
t_qrs = np.arange(-N_qrs//2, N_qrs//2) / fs_ecg

# Graficar ambos
plt.figure(figsize=(10,5))
plt.plot(t_qrs, ecg_segment_qrs, label='Segmento QRS Original')
plt.plot(t_qrs, ecg_segment_qrs_filt, label='Segmento QRS Filtrado (Filtro FIR - Cuadrados Mínimos)', alpha=0.7)
plt.title(f'Segmento QRS Filtrado (Filtro FIR - Ventanas (Cuadrados Mínimos))')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()

#%% Análisis de regiones de interés con señales filtradas

cant_muestras = len(ecg_one_lead)

# Duración total en segundos
dur_total = cant_muestras / fs_ecg

# Definir regiones con ruido (en muestras) dentro del rango válido
regs_ruido = (
    [100, 200],   # ejemplo: muestras 100 a 200 (0.1s a 0.2s)
    [300, 400],   # ejemplo: muestras 300 a 400 (0.3s a 0.4s)
)

# Definir regiones sin ruido (en segundos, convertido a muestras)
regs_sin_ruido = (
    np.array([0.1, 0.2]) * fs_ecg,   # 0.1 a 0.2 segundos
    np.array([0.4, 0.5]) * fs_ecg,   # 0.4 a 0.5 segundos
    np.array([0.5, 0.6]) * fs_ecg,   # 0.5 a 0.6 segundos
)

regiones = [("Con Ruido", regs_ruido), ("Sin Ruido", regs_sin_ruido)]

min_len = 50  # mínimo largo de región en muestras para graficar

for tipo_region, lista_regiones in regiones:
    for i, reg in enumerate(lista_regiones):
        reg = np.array(reg, dtype=int)
        start_idx, end_idx = reg[0], reg[1]

        # Verifico si la región está dentro de la señal y tiene un tamaño mínimo
        if start_idx < 0 or end_idx > cant_muestras or (end_idx - start_idx) < min_len:
            print(f"Región inválida para {tipo_region} - Región {i+1}, la salto.")
            continue

        zoom_region = np.arange(start_idx, end_idx)
        t_zoom = t_total[zoom_region]

        plt.figure(figsize=(12, 10))
        plt.suptitle(f'{tipo_region} - Región {i+1}: muestras {start_idx} a {end_idx}', fontsize=14)

        plt.subplot(4, 1, 1)
        plt.plot(t_zoom, ecg_one_lead[zoom_region], label='Original', alpha=0.6)
        plt.plot(t_zoom, ecg_filtrada_butter[zoom_region], label='Butterworth', linewidth=1.2)
        plt.title('Filtro IIR - Butterworth')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(t_zoom, ecg_one_lead[zoom_region], label='Original', alpha=0.6)
        plt.plot(t_zoom, ecg_filtrada_ellip[zoom_region], label='Cauer (Elíptico)', linewidth=1.2)
        plt.title('Filtro IIR - Cauer (Elíptico)')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(t_zoom, ecg_one_lead[zoom_region], label='Original', alpha=0.6)
        plt.plot(t_zoom, ecg_filtrado_kaiser[zoom_region], label='FIR - Ventana Kaiser', linewidth=1.2)
        plt.title('Filtro FIR - Ventana Kaiser')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(t_zoom, ecg_one_lead[zoom_region], label='Original', alpha=0.6)
        plt.plot(t_zoom, ecg_filtrado_ls[zoom_region], label='FIR - Mínimos Cuadrados', linewidth=1.2)
        plt.title('Filtro FIR - Mínimos Cuadrados')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()