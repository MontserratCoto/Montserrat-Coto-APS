# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 17:34:44 2025

@author: monse
"""

#%% módulos y funciones a importar
import matplotlib.pyplot as plt
import numpy as np


#%% a) Generar el siguiente resultado producto de la experimentación. B = 4 bits, kn=1

#%% Datos de la simulación

fs = 1000 # Frecuencia de muestreo en Hz
N = 1000 # Número total de muestras
f0= 1  # Frecuencia de la señal en Hz

# Datos del ADC
B = 4 # Bits del ADC (resolución)
Vf = 2 # Rango de voltaje simétrico +/- Vf Volts
q =  Vf/2**(B-1) # Paso de cuantización en Volts

# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = q**2/12 # Potencia del ruido de cuantización (Watts)
kn = 1. # Factor de escala para el ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn  # Potencia del ruido analógico 

ts = 1/fs  # Período de muestreo (tiempo entre muestras)
df =fs/N  # Resolución espectral

#%% Experimento: 

# Grilla de sampleo temporal

tt = np.linspace(0, (N-1)*ts, N)  # Vector de tiempos
argg = 2*np.pi*f0*tt # Argumento para la función senoidal


xx = np.sqrt(2)*np.sin(argg)  # Señal senoidal
varianza = np.var(xx) # Cálculo de la varianza de la señal

analog_sig = xx/np.sqrt(varianza) # señal analógica normalizada (para que su potencia sea 1 W) sin ruido
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)  # Ruido gaussiano
sr = analog_sig + nn # Señal analógica de entrada al ADC (con ruido analógico)
srq = np.round(sr/q)*q # Señal cuantizada

nq = srq - sr # Ruido de cuantización


#%% Visualización de resultados

# cierro ventanas anteriores
plt.close('all')

# ##################
# # Señal temporal
# ##################

plt.figure(1)

# plt.plot(tt, srq, lw=1, ls='solid', color='blue', marker='o', markersize=1, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='ADC out (diezmada)')
# plt.plot(tt, sr, lw=1, ls='dashdot', color='black', label='$ s $ (analog)')
# plt.plot(tt, analog_sig, lw=2, color='orange', marker='', ls='dotted', label='$ s $ Analog Signal')
plt.plot(tt, srq, lw=1, ls='solid', color='blue', marker='o', markersize=1,markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='$s_{RQ}$ -Señal cuantizada')
plt.plot(tt, sr, lw=1, ls='dashdot', color='green', label='$s_{R}$ - Señal analógica de entrada al ADC (con ruido analógico)')
plt.plot(tt, analog_sig, lw=2, color='orange', ls='dotted', label='$s$-Señal analógica con ruido analógico')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V - $k_n$ = {:3.3f}'.format(B, Vf, q, kn))
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


# ###########
# # Espectro
# ###########

plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr) #Vector de N muestras, num. complejos
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( analog_sig)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# # grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

#Los espectros los consigo quedandome con la mitad del vector del np.fft.fft porque son redundantes
#por la símetria de la fft, o me quedo con el módulo o me quedo con la fase

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$s$-Señal analógica con ruido analógico')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$s_{R}$ - Señal analógica de entrada al ADC (con ruido analógico)' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$s_{RQ}$ -Señal cuantizada' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')


plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V  q = {:3.3f} V- $k_n$ = {:3.3f}'.format(B, Vf, q, kn))
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

# # #############
# # # Histograma
# # #############

plt.figure(3)
bins = 10
plt.hist(nq.flatten(), bins=bins)
#plt.hist(nqf.flatten()/(q/2), bins=2*bins)
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
#plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N*R/bins, N*R/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V- $k_n$ = {:3.3f}'.format(B, Vf, q, kn))

plt.xlabel('Pasos de cuantización (q) [V]')

#%% b) 1. Analizar para una de las siguientes configuraciones 
#%%Kn=1, y vario los B=4, 8 y 16
#%% Datos de la simulación


fs = 1000 # Frecuencia de muestreo en Hz
N = 1000 # Número total de muestras
f0= 1  # Frecuencia de la señal en Hz

# Datos del ADC
B = 4 # Bits del ADC (resolución)
Vf = 2 # Rango de voltaje simétrico +/- Vf Volts
q =  Vf/2**(B-1) # Paso de cuantización en Volts

# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = q**2/12 # Potencia del ruido de cuantización (Watts)
kn = 1. # Factor de escala para el ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn  # Potencia del ruido analógico 

ts = 1/fs  # Período de muestreo (tiempo entre muestras)
df =fs/N  # Resolución espectral

#%% Experimento: 

# Grilla de sampleo temporal

tt = np.linspace(0, (N-1)*ts, N)  # Vector de tiempos
argg = 2*np.pi*f0*tt # Argumento para la función senoidal


xx = np.sqrt(2)*np.sin(argg)  # Señal senoidal
varianza = np.var(xx) # Cálculo de la varianza de la señal

analog_sig = xx/np.sqrt(varianza) # señal analógica normalizada (para que su potencia sea 1 W) sin ruido
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)  # Ruido gaussiano
sr = analog_sig + nn # Señal analógica de entrada al ADC (con ruido analógico)
srq = np.round(sr/q)*q # Señal cuantizada

nq = srq - sr # Ruido de cuantización


#%% Visualización de resultados

# ##################
# # Señal temporal
# ##################




plt.figure(5)

plt.plot(tt, srq, lw=1, ls='solid', color='blue', marker='o', markersize=1,markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='$s_{RQ}$ -Señal cuantizada')
plt.plot(tt, sr, lw=1, ls='dashdot', color='green', label='$s_{R}$ - Señal analógica de entrada al ADC (con ruido analógico)')
plt.plot(tt, analog_sig, lw=2, color='orange', ls='dotted', label='$s$-Señal analógica con ruido analógico')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V - $k_n$ = {:3.3f}'.format(B, Vf, q, kn))
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


# ###########
# # Espectro
# ###########

plt.figure(6)
ft_SR = 1/N*np.fft.fft( sr) #Vector de N muestras, num. complejos
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( analog_sig)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# # grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)


plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$s$-Señal analógica con ruido analógico')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$s_{R}$ - Señal analógica de entrada al ADC (con ruido analógico)' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$s_{RQ}$ -Señal cuantizada' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')


plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V  q = {:3.3f} V- $k_n$ = {:3.3f}'.format(B, Vf, q, kn))
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

# # #############
# # # Histograma
# # #############

plt.figure(7)
bins = 10
plt.hist(nq.flatten(), bins=bins)
#plt.hist(nqf.flatten()/(q/2), bins=2*bins)
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
#plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N*R/bins, N*R/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V- $k_n$ = {:3.3f}'.format(B, Vf, q, kn))

plt.xlabel('Pasos de cuantización (q) [V]')

#%%B=8bits, kn=1
#%% Datos de la simulación

fs = 1000 # Frecuencia de muestreo en Hz
N = 1000 # Número total de muestras
f0= 1  # Frecuencia de la señal en Hz

# Datos del ADC
B = 8 # Bits del ADC (resolución)
Vf = 2 # Rango de voltaje simétrico +/- Vf Volts
q =  Vf/2**(B-1) # Paso de cuantización en Volts

# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = q**2/12 # Potencia del ruido de cuantización (Watts)
kn = 1. # Factor de escala para el ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn  # Potencia del ruido analógico 

ts = 1/fs  # Período de muestreo (tiempo entre muestras)
df =fs/N  # Resolución espectral

#%% Experimento: 

# Grilla de sampleo temporal

tt = np.linspace(0, (N-1)*ts, N)  # Vector de tiempos
argg = 2*np.pi*f0*tt # Argumento para la función senoidal


xx = np.sqrt(2)*np.sin(argg)  # Señal senoidal
varianza = np.var(xx) # Cálculo de la varianza de la señal

analog_sig = xx/np.sqrt(varianza) # señal analógica normalizada (para que su potencia sea 1 W) sin ruido
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)  # Ruido gaussiano
sr = analog_sig + nn # Señal analógica de entrada al ADC (con ruido analógico)
srq = np.round(sr/q)*q # Señal cuantizada

nq = srq - sr # Ruido de cuantización


#%% Visualización de resultados

# ##################
# # Señal temporal
# ##################

plt.figure(8)

plt.plot(tt, srq, lw=1, ls='solid', color='blue', marker='o', markersize=1,markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='$s_{RQ}$ -Señal cuantizada')
plt.plot(tt, sr, lw=1, ls='dashdot', color='green', label='$s_{R}$ - Señal analógica de entrada al ADC (con ruido analógico)')
plt.plot(tt, analog_sig, lw=2, color='orange', ls='dotted', label='$s$-Señal analógica con ruido analógico')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V - $k_n$ = {:3.3f}'.format(B, Vf, q, kn))
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


# ###########
# # Espectro
# ###########

plt.figure(9)
ft_SR = 1/N*np.fft.fft( sr) #Vector de N muestras, num. complejos
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( analog_sig)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# # grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)


plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$s$-Señal analógica con ruido analógico')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$s_{R}$ - Señal analógica de entrada al ADC (con ruido analógico)' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$s_{RQ}$ -Señal cuantizada' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')


plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V  q = {:3.3f} V- $k_n$ = {:3.3f}'.format(B, Vf, q, kn))
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

# # #############
# # # Histograma
# # #############

plt.figure(10)
bins = 10
plt.hist(nq.flatten(), bins=bins)
#plt.hist(nqf.flatten()/(q/2), bins=2*bins)
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
#plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N*R/bins, N*R/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V- $k_n$ = {:3.3f}'.format(B, Vf, q, kn))

plt.xlabel('Pasos de cuantización (q) [V]')

#%%B=16bits, kn=1
#%% Datos de la simulación

fs = 1000 # Frecuencia de muestreo en Hz
N = 1000 # Número total de muestras
f0= 1  # Frecuencia de la señal en Hz

# Datos del ADC
B = 16 # Bits del ADC (resolución)
Vf = 2 # Rango de voltaje simétrico +/- Vf Volts
q =  Vf/2**(B-1) # Paso de cuantización en Volts

# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = q**2/12 # Potencia del ruido de cuantización (Watts)
kn = 1 # Factor de escala para el ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn  # Potencia del ruido analógico 

ts = 1/fs  # Período de muestreo (tiempo entre muestras)
df =fs/N  # Resolución espectral

#%% Experimento: 

# Grilla de sampleo temporal

tt = np.linspace(0, (N-1)*ts, N)  # Vector de tiempos
argg = 2*np.pi*f0*tt # Argumento para la función senoidal


xx = np.sqrt(2)*np.sin(argg)  # Señal senoidal
varianza = np.var(xx) # Cálculo de la varianza de la señal

analog_sig = xx/np.sqrt(varianza) # señal analógica normalizada (para que su potencia sea 1 W) sin ruido
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)  # Ruido gaussiano
sr = analog_sig + nn # Señal analógica de entrada al ADC (con ruido analógico)
srq = np.round(sr/q)*q # Señal cuantizada

nq = srq - sr # Ruido de cuantización


#%% Visualización de resultados

# ##################
# # Señal temporal
# ##################

plt.figure(11)

plt.plot(tt, srq, lw=1, ls='solid', color='blue', marker='o', markersize=1,markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='$s_{RQ}$ -Señal cuantizada')
plt.plot(tt, sr, lw=1, ls='dashdot', color='green', label='$s_{R}$ - Señal analógica de entrada al ADC (con ruido analógico)')
plt.plot(tt, analog_sig, lw=2, color='orange', ls='dotted', label='$s$-Señal analógica con ruido analógico')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V - $k_n$ = {:3.3f}'.format(B, Vf, q, kn))
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


# ###########
# # Espectro
# ###########

plt.figure(12)
ft_SR = 1/N*np.fft.fft( sr) #Vector de N muestras, num. complejos
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( analog_sig)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# # grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)


plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$s$-Señal analógica con ruido analógico')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$s_{R}$ - Señal analógica de entrada al ADC (con ruido analógico)' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$s_{RQ}$ -Señal cuantizada' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')


plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V  q = {:3.3f} V- $k_n$ = {:3.3f}'.format(B, Vf, q, kn))
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

# # #############
# # # Histograma
# # #############

plt.figure(13)
bins = 10
plt.hist(nq.flatten(), bins=bins)
#plt.hist(nqf.flatten()/(q/2), bins=2*bins)
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
#plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N*R/bins, N*R/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V- $k_n$ = {:3.3f}'.format(B, Vf, q, kn))

plt.xlabel('Pasos de cuantización (q) [V]')

#%% b)2. Analizar para una de las siguientes configuraciones 
#%%B=8, y vario los kN=1/10, 1 y 10
#%% Datos de la simulación


fs = 1000 # Frecuencia de muestreo en Hz
N = 1000 # Número total de muestras
f0= 1  # Frecuencia de la señal en Hz

# Datos del ADC
B = 8 # Bits del ADC (resolución)
Vf = 2 # Rango de voltaje simétrico +/- Vf Volts
q =  Vf/2**(B-1) # Paso de cuantización en Volts

# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = q**2/12 # Potencia del ruido de cuantización (Watts)
kn = 10 # Factor de escala para el ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn  # Potencia del ruido analógico 

ts = 1/fs  # Período de muestreo (tiempo entre muestras)
df =fs/N  # Resolución espectral

#%% Experimento: 

# Grilla de sampleo temporal

tt = np.linspace(0, (N-1)*ts, N)  # Vector de tiempos
argg = 2*np.pi*f0*tt # Argumento para la función senoidal


xx = np.sqrt(2)*np.sin(argg)  # Señal senoidal
varianza = np.var(xx) # Cálculo de la varianza de la señal

analog_sig = xx/np.sqrt(varianza) # señal analógica normalizada (para que su potencia sea 1 W) sin ruido
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)  # Ruido gaussiano
sr = analog_sig + nn # Señal analógica de entrada al ADC (con ruido analógico)
srq = np.round(sr/q)*q # Señal cuantizada

nq = srq - sr # Ruido de cuantización


#%% Visualización de resultados

# ##################
# # Señal temporal
# ##################




plt.figure(14)

plt.plot(tt, srq, lw=1, ls='solid', color='blue', marker='o', markersize=1,markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='$s_{RQ}$ -Señal cuantizada')
plt.plot(tt, sr, lw=1, ls='dashdot', color='green', label='$s_{R}$ - Señal analógica de entrada al ADC (con ruido analógico)')
plt.plot(tt, analog_sig, lw=2, color='orange', ls='dotted', label='$s$-Señal analógica con ruido analógico')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V - $k_n$ = {:3.3f}'.format(B, Vf, q, kn))
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


# ###########
# # Espectro
# ###########

plt.figure(15)
ft_SR = 1/N*np.fft.fft( sr) #Vector de N muestras, num. complejos
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( analog_sig)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# # grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)


plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$s$-Señal analógica con ruido analógico')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$s_{R}$ - Señal analógica de entrada al ADC (con ruido analógico)' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$s_{RQ}$ -Señal cuantizada' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')


plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V  q = {:3.3f} V- $k_n$ = {:3.3f}'.format(B, Vf, q, kn))
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

# # #############
# # # Histograma
# # #############

plt.figure(16)
bins = 10
plt.hist(nq.flatten(), bins=bins)
#plt.hist(nqf.flatten()/(q/2), bins=2*bins)
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
#plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N*R/bins, N*R/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V- $k_n$ = {:3.3f}'.format(B, Vf, q, kn))

plt.xlabel('Pasos de cuantización (q) [V]')

