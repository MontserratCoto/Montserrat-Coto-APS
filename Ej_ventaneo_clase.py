# -*- coding: utf-8 -*-
"""
@author: monse
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
from scipy.fft import fft, fftshift

#%% Datos de la simulación

# Datos del ADC
B = 8 # bits
Vf = 2 # rango simétrico de +/- Vf Volts
q =  Vf/2**(B-1)# paso de cuantización de q Volts

# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = q**2/12# Watts 
kn = 1. # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn # 


# Defino mis parámetros
vmax = 1 #la amplitud máxima de la senoidal (volts)
dc = 0 #su valor medio (volts)
ff = 0.9 #la frecuencia fundamental de la onda (Hz)
ph = 0 #la fase (radianes)

fs = 1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras
f0= 1  # Frecuencia de la señal en Hz
ts = 1/fs  # tiempo de muestreo
df =fs/N  # resolución espectral 

# Defino la función pedida
def mi_funcion_sen(vmax, dc, ff, ph, N, fs):
   #tt es el vector de instantes de tiempo en los que se toman las muestras de la señal
    tt = np.linspace(0, (N-1)*ts, num=N) #Vector de N posiciones
    # Argumento del seno: 2π*frecuencia*tiempo + fase
    arg=2*np.pi*ff*tt + ph
    #Señal senoidal
    xx = vmax * np.sin(arg) + dc
    return tt, xx  
#tt guarda los tiempos en los que se toman las muestras.
#xx guarda los valores de la señal en esos tiempos.

# Llamo a la función 
tt, xx = mi_funcion_sen(vmax=1, dc=0, ff=250 +0.5, ph=0, N=1000, fs=fs)


#%% Ventaneo
ventana=signal.windows.hann(N,sym=True)
xw=xx*ventana

#%% Normalizo
#xn=xx/np.std(xx)
#xwn=xw/np.std(xx)

analog_sig = xx/np.sqrt(xx) # señal analógica normalizada sin ruido; xn
nn = np.random.normal(0, np.sqrt(pot_ruido_cuant), N) # señal de ruido de analógico
sr = analog_sig + nn# señal analógica de entrada al ADC (con ruido analógico)
srq = np.round(sr/q)*q # señal cuantizada

analog_sig_ventana = xw/np.sqrt(xw) # señal analógica normalizada sin ruido; xn
nn_ventana = np.random.normal(0, np.sqrt(pot_ruido_cuant), N) # señal de ruido de analógico
sr_ventana = analog_sig_ventana + nn_ventana# señal analógica de entrada al ADC (con ruido analógico)
srq_ventana = np.round(sr_ventana/q)*q # señal cuantizada

nq = srq - sr # señal de ruido de cuantización

#%% Experimento: 

# Grilla de sampleo temporal

tt = np.linspace(0, (N-1)*ts, N)  # Vector de tiempos
argg = 2*np.pi*f0*tt # Argumento para la función senoidal


# Grafico
# cierro ventanas anteriores
plt.close('all')

plt.figure(1)
plt.plot(tt, analog_sig )
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.title("Señal Senoidal")
plt.grid()
plt.show()

plt.figure(2)
plt.plot(tt, analog_sig_ventana)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.title("Señal Senoidal con ventana hann")
plt.grid()
plt.show()

plt.figure(3)
plt.plot(tt, analog_sig,color='orange',ls='dotted', label='$ s $ (seno.)')
plt.plot(tt, analog_sig_ventana,color='pink',ls='dotted', label='$ s $ (seno con ventana)')
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.title("Señal Senoidal vs ventana hann")
plt.grid()
plt.show()

# ###########
# # Espectro
# ###########

#grilla de sampleo frecuencial
df =fs/N
ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs/2

plt.figure(4)

plt.plot(tt, srq, lw=1, ls='solid', color='blue', marker='o', markersize=1,markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='$s_{RQ}$ -Señal cuantizada')
plt.plot(tt, sr, lw=1, ls='dashdot', color='green', label='$s_{R}$ - Señal analógica de entrada al ADC (con ruido analógico)')
plt.plot(tt, analog_sig, lw=2, color='orange', ls='dotted', label='$s$-Señal analógica con ruido analógico')

plt.title('Señal muestreada original')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

plt.figure(5)
plt.plot(tt, srq_ventana, lw=1, ls='solid', color='blue', marker='o', markersize=1,markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='$s_{RQ}$ -Señal cuantizada ventana')
plt.plot(tt, sr_ventana, lw=1, ls='dashdot', color='green', label='$s_{R}$ - Señal analógica con ventana de entrada al ADC (con ruido analógico)')
plt.plot(tt, analog_sig_ventana, lw=2, color='orange', ls='dotted', label='$s$-Señal analógica ventana con ruido analógico')

plt.title('Señal muestreada con ventana')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()