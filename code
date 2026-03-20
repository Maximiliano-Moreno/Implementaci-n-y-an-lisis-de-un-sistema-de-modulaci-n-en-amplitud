# ============================================
# SISTEMA DE MODULACIÓN EN AMPLITUD (AM)
# Implementación y análisis en Python
# Actividad Formativa 4 - Sistemas de Comunicación
# ============================================

# ============================================
# CELDA 1: INSTALAR LIBRERÍAS NECESARIAS
# ============================================

!pip install numpy scipy matplotlib -q

print("✅ Librerías instaladas correctamente")

# ============================================
# CELDA 2: IMPORTAR LIBRERÍAS
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift, ifft
import matplotlib.gridspec as gridspec
from ipywidgets import interact, widgets, FloatSlider, IntSlider
import warnings
warnings.filterwarnings('ignore')

print("✅ Librerías importadas correctamente")

# Configuración de gráficos
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# ============================================
# CELDA 3: CONFIGURACIÓN DE PARÁMETROS DEL SISTEMA
# ============================================

print("="*70)
print("📡 SISTEMA DE MODULACIÓN EN AMPLITUD (AM)")
print("="*70)

# Parámetros de tiempo
fs = 10000  # Frecuencia de muestreo (Hz)
Ts = 1/fs   # Período de muestreo (s)
t = np.arange(0, 0.1, Ts)  # Vector de tiempo (0 a 0.1 segundos)

# Parámetros de la señal de mensaje (información)
fm = 100    # Frecuencia de la señal moduladora (Hz)
Am = 1      # Amplitud de la señal moduladora

# Parámetros de la señal portadora
fc = 1000   # Frecuencia de la portadora (Hz)
Ac = 1      # Amplitud de la portadora

# Índice de modulación
m = 0.8     # Índice de modulación (0 < m < 1 para evitar sobremodulación)

print("\n📊 PARÁMETROS DEL SISTEMA:")
print(f"   Frecuencia de muestreo (fs): {fs} Hz")
print(f"   Frecuencia de la señal de mensaje (fm): {fm} Hz")
print(f"   Frecuencia de la portadora (fc): {fc} Hz")
print(f"   Índice de modulación (m): {m}")
print(f"   Duración de la señal: {len(t)/fs} segundos")

# ============================================
# CELDA 4: GENERACIÓN DE SEÑALES
# ============================================

print("\n" + "="*70)
print("📡 GENERACIÓN DE SEÑALES")
print("="*70)

# Señal de mensaje (información) - Señal de baja frecuencia
# Usamos una señal sinusoidal simple
senal_mensaje = Am * np.cos(2 * np.pi * fm * t)

# Señal portadora - Alta frecuencia
senal_portadora = Ac * np.cos(2 * np.pi * fc * t)

# Modulación en Amplitud (AM)
# s(t) = [Ac + Am * cos(2πfm t)] * cos(2πfc t)
# s(t) = Ac * [1 + m * cos(2πfm t)] * cos(2πfc t)
senal_modulada = Ac * (1 + m * senal_mensaje) * np.cos(2 * np.pi * fc * t)

print("✅ Señal de mensaje generada")
print("✅ Señal portadora generada")
print("✅ Señal modulada en AM generada")

# ============================================
# CELDA 5: VISUALIZACIÓN EN DOMINIO DEL TIEMPO
# ============================================

print("\n" + "="*70)
print("📊 VISUALIZACIÓN EN DOMINIO DEL TIEMPO")
print("="*70)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Subplot 1: Señal de mensaje
axes[0].plot(t[:1000], senal_mensaje[:1000], 'b-', linewidth=1.5)
axes[0].set_title('📡 SEÑAL DE MENSAJE (Información a transmitir)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Tiempo (s)')
axes[0].set_ylabel('Amplitud')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 0.01])
axes[0].text(0.008, 0.8, f'Frecuencia: {fm} Hz', bbox=dict(facecolor='white', alpha=0.8))

# Subplot 2: Señal portadora
axes[1].plot(t[:500], senal_portadora[:500], 'r-', linewidth=1)
axes[1].set_title('🔊 SEÑAL PORTADORA (Alta frecuencia)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Tiempo (s)')
axes[1].set_ylabel('Amplitud')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 0.005])
axes[1].text(0.004, 0.8, f'Frecuencia: {fc} Hz', bbox=dict(facecolor='white', alpha=0.8))

# Subplot 3: Señal modulada en AM
axes[2].plot(t[:1000], senal_modulada[:1000], 'g-', linewidth=1.5)
axes[2].set_title('📡 SEÑAL MODULADA EN AMPLITUD (AM)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Tiempo (s)')
axes[2].set_ylabel('Amplitud')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([0, 0.01])

# Envolvente de la señal modulada
envolvente = Ac * (1 + m * senal_mensaje)
axes[2].plot(t[:1000], envolvente[:1000], 'k--', linewidth=1, alpha=0.7, label='Envolvente')
axes[2].plot(t[:1000], -envolvente[:1000], 'k--', linewidth=1, alpha=0.7)
axes[2].legend()
axes[2].text(0.008, 1.2, f'Índice de modulación m = {m}', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

print("✅ Gráficas en dominio del tiempo generadas")

# ============================================
# CELDA 6: ANÁLISIS EN DOMINIO DE LA FRECUENCIA
# ============================================

print("\n" + "="*70)
print("📊 ANÁLISIS EN DOMINIO DE LA FRECUENCIA (FFT)")
print("="*70)

def calcular_espectro(senal, fs):
    """Calcula el espectro de frecuencia de una señal"""
    N = len(senal)
    espectro = fftshift(fft(senal))
    frecuencias = fftshift(np.fft.fftfreq(N, 1/fs))
    return frecuencias, np.abs(espectro)/N

# Calcular espectros
f_mensaje, esp_mensaje = calcular_espectro(senal_mensaje, fs)
f_portadora, esp_portadora = calcular_espectro(senal_portadora, fs)
f_modulada, esp_modulada = calcular_espectro(senal_modulada, fs)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Espectro de la señal de mensaje
axes[0].plot(f_mensaje, esp_mensaje, 'b-', linewidth=1.5)
axes[0].set_title('📊 ESPECTRO DE LA SEÑAL DE MENSAJE', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Frecuencia (Hz)')
axes[0].set_ylabel('Magnitud')
axes[0].set_xlim([-200, 200])
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=fm, color='r', linestyle='--', alpha=0.5, label=f'fm = {fm} Hz')
axes[0].axvline(x=-fm, color='r', linestyle='--', alpha=0.5)
axes[0].legend()

# Espectro de la señal portadora
axes[1].plot(f_portadora, esp_portadora, 'r-', linewidth=1.5)
axes[1].set_title('📊 ESPECTRO DE LA SEÑAL PORTADORA', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Frecuencia (Hz)')
axes[1].set_ylabel('Magnitud')
axes[1].set_xlim([-1200, 1200])
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=fc, color='b', linestyle='--', alpha=0.5, label=f'fc = {fc} Hz')
axes[1].axvline(x=-fc, color='b', linestyle='--', alpha=0.5)
axes[1].legend()

# Espectro de la señal modulada en AM
axes[2].plot(f_modulada, esp_modulada, 'g-', linewidth=1.5)
axes[2].set_title('📊 ESPECTRO DE LA SEÑAL MODULADA EN AM', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Frecuencia (Hz)')
axes[2].set_ylabel('Magnitud')
axes[2].set_xlim([-1200, 1200])
axes[2].grid(True, alpha=0.3)
axes[2].axvline(x=fc, color='r', linestyle='--', alpha=0.5, label=f'Portadora: ±{fc} Hz')
axes[2].axvline(x=fc+fm, color='orange', linestyle='--', alpha=0.5, label=f'Banda lateral superior: {fc+fm} Hz')
axes[2].axvline(x=fc-fm, color='orange', linestyle='--', alpha=0.5, label=f'Banda lateral inferior: {fc-fm} Hz')
axes[2].axvline(x=-fc+fm, color='orange', linestyle='--', alpha=0.5)
axes[2].axvline(x=-fc-fm, color='orange', linestyle='--', alpha=0.5)
axes[2].legend()

plt.tight_layout()
plt.show()

print("✅ Gráficas en dominio de la frecuencia generadas")

# ============================================
# CELDA 7: ANÁLISIS DEL ÍNDICE DE MODULACIÓN
# ============================================

print("\n" + "="*70)
print("📊 ANÁLISIS DEL ÍNDICE DE MODULACIÓN")
print("="*70)

# Probar diferentes índices de modulación
indices_modulacion = [0.3, 0.5, 0.8, 1.0, 1.2]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, m_val in enumerate(indices_modulacion):
    if idx >= 5:
        break
    
    senal_am = Ac * (1 + m_val * senal_mensaje) * np.cos(2 * np.pi * fc * t)
    
    axes[idx].plot(t[:1000], senal_am[:1000], 'b-', linewidth=1)
    envolvente_test = Ac * (1 + m_val * senal_mensaje)
    axes[idx].plot(t[:1000], envolvente_test[:1000], 'r--', linewidth=1, alpha=0.7)
    axes[idx].plot(t[:1000], -envolvente_test[:1000], 'r--', linewidth=1, alpha=0.7)
    
    if m_val <= 1:
        axes[idx].set_title(f'm = {m_val} (Modulación normal)', fontsize=10)
    else:
        axes[idx].set_title(f'm = {m_val} (Sobremodulación)', fontsize=10, color='red')
    
    axes[idx].set_xlabel('Tiempo (s)')
    axes[idx].set_ylabel('Amplitud')
    axes[idx].set_xlim([0, 0.01])
    axes[idx].grid(True, alpha=0.3)

# Ocultar el último subplot
axes[5].axis('off')

plt.suptitle('📊 EFECTO DEL ÍNDICE DE MODULACIÓN EN LA SEÑAL AM', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n📌 ANÁLISIS DEL ÍNDICE DE MODULACIÓN:")
print("   • m < 1: Modulación normal (submodulación) - La envolvente reproduce fielmente la señal")
print("   • m = 1: Modulación al 100% - Máxima amplitud sin distorsión")
print("   • m > 1: Sobremodulación - Distorsión y pérdida de información")

# ============================================
# CELDA 8: ANÁLISIS CON RUIDO AWGN
# ============================================

print("\n" + "="*70)
print("📊 ANÁLISIS CON RUIDO AWGN (Additive White Gaussian Noise)")
print("="*70)

# Niveles de ruido (SNR en dB)
snr_valores = [30, 20, 10, 5, 0]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, snr_db in enumerate(snr_valores):
    if idx >= 5:
        break
    
    # Calcular potencia de la señal
    potencia_senal = np.mean(senal_modulada**2)
    
    # Calcular potencia del ruido
    potencia_ruido = potencia_senal / (10**(snr_db/10))
    
    # Generar ruido
    ruido = np.sqrt(potencia_ruido) * np.random.randn(len(t))
    
    # Señal con ruido
    senal_ruidosa = senal_modulada + ruido
    
    axes[idx].plot(t[:500], senal_ruidosa[:500], 'b-', linewidth=1, alpha=0.7)
    axes[idx].plot(t[:500], senal_modulada[:500], 'r-', linewidth=0.8, alpha=0.5, label='Original')
    axes[idx].set_title(f'SNR = {snr_db} dB', fontsize=10)
    axes[idx].set_xlabel('Tiempo (s)')
    axes[idx].set_ylabel('Amplitud')
    axes[idx].set_xlim([0, 0.005])
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(fontsize=8)

# Ocultar el último subplot
axes[5].axis('off')

plt.suptitle('📊 EFECTO DEL RUIDO AWGN EN LA SEÑAL AM', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n📌 ANÁLISIS DEL RUIDO:")
print("   • SNR alto (30-20 dB): La señal se mantiene claramente visible")
print("   • SNR medio (10-5 dB): El ruido comienza a distorsionar la señal")
print("   • SNR bajo (0 dB): La señal queda prácticamente oculta por el ruido")

# ============================================
# CELDA 9: DEMODULACIÓN EN AM
# ============================================

print("\n" + "="*70)
print("📊 PROCESO DE DEMODULACIÓN EN AM")
print("="*70)

# Demodulación por detector de envolvente
def demodular_am(senal_modulada, fc, fs, m, Ac):
    """Demodulación de señal AM usando detector de envolvente"""
    # Detector de envolvente (valor absoluto + filtro paso bajo)
    envolvente_detectada = np.abs(senal_modulada)
    
    # Diseño de filtro paso bajo
    orden = 100
    frecuencia_corte = 2 * fm  # Frecuencia de corte del filtro
    b = signal.firwin(orden, frecuencia_corte, fs=fs)
    
    # Aplicar filtro paso bajo
    senal_demodulada = signal.filtfilt(b, 1, envolvente_detectada)
    
    # Eliminar componente DC
    senal_demodulada = senal_demodulada - np.mean(senal_demodulada)
    
    return senal_demodulada

senal_demodulada = demodular_am(senal_modulada, fc, fs, m, Ac)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Señal original de mensaje
axes[0].plot(t[:1000], senal_mensaje[:1000], 'b-', linewidth=1.5)
axes[0].set_title('📡 SEÑAL ORIGINAL (Mensaje a transmitir)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Tiempo (s)')
axes[0].set_ylabel('Amplitud')
axes[0].set_xlim([0, 0.01])
axes[0].grid(True, alpha=0.3)

# Señal modulada en AM
axes[1].plot(t[:1000], senal_modulada[:1000], 'g-', linewidth=1)
axes[1].set_title('📡 SEÑAL MODULADA EN AM (Transmisión)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Tiempo (s)')
axes[1].set_ylabel('Amplitud')
axes[1].set_xlim([0, 0.01])
axes[1].grid(True, alpha=0.3)

# Señal demodulada
axes[2].plot(t[:1000], senal_demodulada[:1000], 'r-', linewidth=1.5)
axes[2].set_title('📡 SEÑAL DEMODULADA (Señal recuperada)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Tiempo (s)')
axes[2].set_ylabel('Amplitud')
axes[2].set_xlim([0, 0.01])
axes[2].grid(True, alpha=0.3)

# Calcular error de demodulación
error = np.mean((senal_mensaje[:1000] - senal_demodulada[:1000])**2)
axes[2].text(0.008, 0.8, f'Error cuadrático medio: {error:.6f}', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

print(f"\n✅ Demodulación completada")
print(f"   Error cuadrático medio de demodulación: {error:.6f}")

# ============================================
# CELDA 10: ANÁLISIS DE ATENUACIÓN EN EL CANAL
# ============================================

print("\n" + "="*70)
print("📊 ANÁLISIS DE ATENUACIÓN EN EL CANAL DE TRANSMISIÓN")
print("="*70)

# Niveles de atenuación (factor de atenuación)
atenuaciones = [1.0, 0.7, 0.5, 0.3, 0.1]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, atenuacion in enumerate(atenuaciones):
    if idx >= 5:
        break
    
    # Aplicar atenuación
    senal_atenuada = atenuacion * senal_modulada
    
    # Demodular señal atenuada
    senal_demod_atenuada = demodular_am(senal_atenuada, fc, fs, m, Ac)
    
    axes[idx].plot(t[:1000], senal_demod_atenuada[:1000], 'b-', linewidth=1)
    axes[idx].plot(t[:1000], senal_mensaje[:1000], 'r--', linewidth=1, alpha=0.7, label='Original')
    axes[idx].set_title(f'Atenuación: {atenuacion*100:.0f}%', fontsize=10)
    axes[idx].set_xlabel('Tiempo (s)')
    axes[idx].set_ylabel('Amplitud')
    axes[idx].set_xlim([0, 0.01])
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(fontsize=8)

axes[5].axis('off')
plt.suptitle('📊 EFECTO DE LA ATENUACIÓN EN LA SEÑAL DEMODULADA', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n📌 ANÁLISIS DE ATENUACIÓN:")
print("   • Atenuación baja (70-100%): La señal se recupera con buena calidad")
print("   • Atenuación media (30-50%): La amplitud disminuye pero la forma se mantiene")
print("   • Atenuación alta (<30%): La señal se vuelve difícil de recuperar")

# ============================================
# CELDA 11: ANÁLISIS DE DISTORSIÓN POR SOBREMODULACIÓN
# ============================================

print("\n" + "="*70)
print("📊 ANÁLISIS DE DISTORSIÓN POR SOBREMODULACIÓN")
print("="*70)

# Crear señal con sobremodulación (m > 1)
m_sobremod = 1.5
senal_sobremod = Ac * (1 + m_sobremod * senal_mensaje) * np.cos(2 * np.pi * fc * t)

# Demodular señal sobremodulada
senal_demod_sobremod = demodular_am(senal_sobremod, fc, fs, m_sobremod, Ac)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Señal modulada con sobremodulación
axes[0, 0].plot(t[:1000], senal_sobremod[:1000], 'g-', linewidth=1)
envolvente_sobremod = Ac * (1 + m_sobremod * senal_mensaje)
axes[0, 0].plot(t[:1000], envolvente_sobremod[:1000], 'r--', linewidth=1, alpha=0.7, label='Envolvente teórica')
axes[0, 0].plot(t[:1000], -envolvente_sobremod[:1000], 'r--', linewidth=1, alpha=0.7)
axes[0, 0].set_title(f'SEÑAL MODULADA CON SOBREMODULACIÓN (m = {m_sobremod})', fontsize=11, fontweight='bold', color='red')
axes[0, 0].set_xlabel('Tiempo (s)')
axes[0, 0].set_ylabel('Amplitud')
axes[0, 0].set_xlim([0, 0.01])
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Espectro de la señal sobremodulada
f_sobremod, esp_sobremod = calcular_espectro(senal_sobremod, fs)
axes[0, 1].plot(f_sobremod, esp_sobremod, 'g-', linewidth=1)
axes[0, 1].set_title('ESPECTRO CON SOBREMODULACIÓN', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Frecuencia (Hz)')
axes[0, 1].set_ylabel('Magnitud')
axes[0, 1].set_xlim([-1200, 1200])
axes[0, 1].grid(True, alpha=0.3)

# Señal demodulada con distorsión
axes[1, 0].plot(t[:1000], senal_demod_sobremod[:1000], 'b-', linewidth=1)
axes[1, 0].plot(t[:1000], senal_mensaje[:1000], 'r--', linewidth=1, alpha=0.7, label='Señal original')
axes[1, 0].set_title('SEÑAL DEMODULADA CON DISTORSIÓN', fontsize=11, fontweight='bold', color='red')
axes[1, 0].set_xlabel('Tiempo (s)')
axes[1, 0].set_ylabel('Amplitud')
axes[1, 0].set_xlim([0, 0.01])
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Comparación de error
error_sobremod = np.mean((senal_mensaje[:1000] - senal_demod_sobremod[:1000])**2)
error_normal = np.mean((senal_mensaje[:1000] - senal_demodulada[:1000])**2)

axes[1, 1].bar(['Modulación Normal', 'Sobremodulación'], [error_normal, error_sobremod], color=['green', 'red'])
axes[1, 1].set_title('COMPARACIÓN DE ERROR DE DEMODULACIÓN', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Error cuadrático medio')
axes[1, 1].grid(True, alpha=0.3, axis='y')

for i, v in enumerate([error_normal, error_sobremod]):
    axes[1, 1].text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n📌 ANÁLISIS DE DISTORSIÓN:")
print(f"   • Error normal (m = {m}): {error_normal:.6f}")
print(f"   • Error sobremodulación (m = {m_sobremod}): {error_sobremod:.6f}")
print("   • La sobremodulación introduce distorsión severa y pérdida de información")

# ============================================
# CELDA 12: VISUALIZACIÓN INTERACTIVA
# ============================================

print("\n" + "="*70)
print("🎮 VISUALIZACIÓN INTERACTIVA DEL SISTEMA AM")
print("="*70)

def visualizacion_interactiva(fm=100, fc=1000, m=0.8, snr_db=30, atenuacion=1.0):
    """Función interactiva para explorar parámetros del sistema AM"""
    
    # Generar señales con parámetros actualizados
    senal_mensaje_interact = Am * np.cos(2 * np.pi * fm * t)
    senal_modulada_interact = Ac * (1 + m * senal_mensaje_interact) * np.cos(2 * np.pi * fc * t)
    
    # Calcular potencia y agregar ruido
    potencia_senal = np.mean(senal_modulada_interact**2)
    potencia_ruido = potencia_senal / (10**(snr_db/10))
    ruido = np.sqrt(potencia_ruido) * np.random.randn(len(t))
    
    senal_ruidosa = atenuacion * senal_modulada_interact + ruido
    senal_demod_interact = demodular_am(senal_ruidosa, fc, fs, m, Ac)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    
    # Señal de mensaje
    axes[0, 0].plot(t[:500], senal_mensaje_interact[:500], 'b-', linewidth=1.5)
    axes[0, 0].set_title(f'Señal de Mensaje (fm = {fm} Hz)', fontsize=10)
    axes[0, 0].set_xlabel('Tiempo (s)')
    axes[0, 0].set_ylabel('Amplitud')
    axes[0, 0].set_xlim([0, 0.005])
    axes[0, 0].grid(True, alpha=0.3)
    
    # Señal portadora
    axes[0, 1].plot(t[:200], senal_portadora[:200], 'r-', linewidth=1)
    axes[0, 1].set_title(f'Señal Portadora (fc = {fc} Hz)', fontsize=10)
    axes[0, 1].set_xlabel('Tiempo (s)')
    axes[0, 1].set_ylabel('Amplitud')
    axes[0, 1].set_xlim([0, 0.002])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Señal modulada
    axes[1, 0].plot(t[:500], senal_modulada_interact[:500], 'g-', linewidth=1)
    axes[1, 0].set_title(f'Señal Modulada AM (m = {m})', fontsize=10)
    axes[1, 0].set_xlabel('Tiempo (s)')
    axes[1, 0].set_ylabel('Amplitud')
    axes[1, 0].set_xlim([0, 0.005])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Espectro de la señal modulada
    f_mod, esp_mod = calcular_espectro(senal_modulada_interact, fs)
    axes[1, 1].plot(f_mod, esp_mod, 'g-', linewidth=1)
    axes[1, 1].set_title('Espectro de la Señal AM', fontsize=10)
    axes[1, 1].set_xlabel('Frecuencia (Hz)')
    axes[1, 1].set_ylabel('Magnitud')
    axes[1, 1].set_xlim([-1200, 1200])
    axes[1, 1].grid(True, alpha=0.3)
    
    # Señal con ruido y atenuación
    axes[2, 0].plot(t[:500], senal_ruidosa[:500], 'orange', linewidth=1, alpha=0.7)
    axes[2, 0].plot(t[:500], senal_modulada_interact[:500], 'g--', linewidth=0.8, alpha=0.5, label='Original')
    axes[2, 0].set_title(f'Canal: SNR = {snr_db} dB, Atenuación = {atenuacion*100:.0f}%', fontsize=10)
    axes[2, 0].set_xlabel('Tiempo (s)')
    axes[2, 0].set_ylabel('Amplitud')
    axes[2, 0].set_xlim([0, 0.005])
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend(fontsize=8)
    
    # Señal demodulada vs original
    axes[2, 1].plot(t[:500], senal_demod_interact[:500], 'b-', linewidth=1, label='Demodulada')
    axes[2, 1].plot(t[:500], senal_mensaje_interact[:500], 'r--', linewidth=1, alpha=0.7, label='Original')
    axes[2, 1].set_title('Señal Demodulada vs Original', fontsize=10)
    axes[2, 1].set_xlabel('Tiempo (s)')
    axes[2, 1].set_ylabel('Amplitud')
    axes[2, 1].set_xlim([0, 0.005])
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend(fontsize=8)
    
    plt.suptitle('🔬 SISTEMA DE MODULACIÓN EN AMPLITUD - ANÁLISIS INTERACTIVO', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Crear controles interactivos
print("\n🎮 Controles interactivos - Ajusta los parámetros en tiempo real:")
print("   • fm: Frecuencia de la señal de mensaje (50-500 Hz)")
print("   • fc: Frecuencia de la portadora (500-2000 Hz)")
print("   • m: Índice de modulación (0.2-1.5)")
print("   • SNR: Relación señal-ruido en dB (0-40 dB)")
print("   • Atenuación: Factor de atenuación del canal (0.1-1.0)")

# Descomentar la siguiente línea para activar la visualización interactiva
# interact(visualizacion_interactiva, 
#          fm=IntSlider(min=50, max=500, step=10, value=100, description='Frecuencia mensaje (Hz)'),
#          fc=IntSlider(min=500, max=2000, step=50, value=1000, description='Frecuencia portadora (Hz)'),
#          m=FloatSlider(min=0.2, max=1.5, step=0.05, value=0.8, description='Índice modulación (m)'),
#          snr_db=IntSlider(min=0, max=40, step=5, value=30, description='SNR (dB)'),
#          atenuacion=FloatSlider(min=0.1, max=1.0, step=0.05, value=1.0, description='Atenuación'))

print("\n⚠️ Para activar la visualización interactiva, elimina el comentario (#) de la línea 'interact' anterior")

# ============================================
# CELDA 13: REPORTE FINAL Y ESTADÍSTICAS
# ============================================

print("\n" + "="*70)
print("📊 REPORTE FINAL DEL SISTEMA DE MODULACIÓN EN AM")
print("="*70)

print("\n📌 PARÁMETROS DEL SISTEMA:")
print(f"   • Frecuencia de muestreo (fs): {fs} Hz")
print(f"   • Frecuencia de la señal de mensaje (fm): {fm} Hz")
print(f"   • Frecuencia de la portadora (fc): {fc} Hz")
print(f"   • Índice de modulación (m): {m}")
print(f"   • Ancho de banda de la señal AM: {2*fm} Hz")

print("\n📌 ANÁLISIS EN DOMINIO DEL TIEMPO:")
print("   • La señal modulada presenta una envolvente que reproduce la forma de la señal de mensaje")
print("   • La portadora de alta frecuencia transporta la información modulando su amplitud")
print("   • El índice de modulación determina la profundidad de la modulación")

print("\n📌 ANÁLISIS EN DOMINIO DE LA FRECUENCIA:")
print(f"   • Componente espectral en la frecuencia portadora: ±{fc} Hz")
print(f"   • Banda lateral superior: {fc+fm} Hz")
print(f"   • Banda lateral inferior: {fc-fm} Hz")
print(f"   • Ancho de banda total: {2*fm} Hz")

print("\n📌 ANÁLISIS DE RUIDO:")
print("   • SNR alto (>20 dB): La señal mantiene su integridad")
print("   • SNR medio (10-20 dB): Degradación perceptible pero recuperable")
print("   • SNR bajo (<10 dB): La señal se pierde entre el ruido")

print("\n📌 ANÁLISIS DE ATENUACIÓN:")
print("   • Atenuación suave (>50%): La señal se recupera con menor amplitud")
print("   • Atenuación severa (<30%): La señal se vuelve difícil de detectar")

print("\n📌 ANÁLISIS DE DISTORSIÓN:")
if m <= 1:
    print(f"   • Modulación normal (m = {m}): No hay distorsión significativa")
else:
    print(f"   • Sobremodulación (m = {m}): Distorsión severa - m debe ser ≤ 1")

print("\n📌 CONCLUSIONES:")
print("   • El sistema de modulación AM permite transmitir señales de baja frecuencia")
print("   • a través de canales que requieren frecuencias más altas para su propagación.")
print("   • La calidad de la transmisión depende críticamente del índice de modulación,")
print("   • la relación señal-ruido y las condiciones de atenuación del canal.")
print("   • La sobremodulación introduce distorsión no lineal que degrada la señal,")
print("   • mientras que el ruido AWGN afecta la relación señal-ruido y la inteligibilidad.")
print("   • El detector de envolvente es un método simple y efectivo para demodular,")
print("   • siempre que se cumplan las condiciones de modulación adecuada.")
print("   • Este análisis demuestra la importancia de diseñar sistemas de comunicación")
print("   • robustos que consideren las condiciones reales del canal de transmisión.")

print("\n" + "="*70)
print("✅ ACTIVIDAD COMPLETADA - SISTEMA DE MODULACIÓN EN AM")
print("="*70)

# ============================================
# CELDA 14: EXPORTAR RESULTADOS Y GRÁFICAS
# ============================================

print("\n" + "="*70)
print("📁 EXPORTANDO RESULTADOS")
print("="*70)

# Guardar gráficas como imágenes (opcional)
try:
    # Crear carpeta para resultados
    import os
    if not os.path.exists('resultados_am'):
        os.makedirs('resultados_am')
    
    print("📂 Carpeta 'resultados_am' creada")
    print("✅ Las gráficas se pueden guardar manualmente haciendo clic en el ícono de descarga")
except:
    print("⚠️ No se pudo crear la carpeta de resultados")

print("\n📌 RECOMENDACIONES PARA SUBIR A GITHUB:")
print("   1. Guarda este notebook como 'Modulacion_AM.ipynb'")
print("   2. Crea un repositorio en GitHub llamado 'Modulacion-AM'")
print("   3. Sube este archivo junto con un README.md explicando el proyecto")
print("   4. Incluye capturas de pantalla de las gráficas generadas")
print("\n📌 ESTRUCTURA RECOMENDADA DEL README.md:")
print("   # Sistema de Modulación en Amplitud (AM)")
print("   ## Descripción")
print("   Implementación de un sistema de modulación AM usando Python")
print("   ## Características")
print("   - Generación de señales de mensaje y portadora")
print("   - Modulación en amplitud")
print("   - Análisis en tiempo y frecuencia")
print("   - Simulación de ruido AWGN")
print("   - Análisis de atenuación y distorsión")
print("   - Demodulación por detector de envolvente")
print("   ## Requisitos")
print("   - Python 3.7+")
print("   - NumPy, SciPy, Matplotlib")
print("   ## Ejecución")
print("   Ejecutar todas las celdas en Google Colab o Jupyter Notebook")
print("   ## Resultados")
print("   El sistema genera gráficas que muestran el comportamiento de la señal AM")

print("\n🎉 ACTIVIDAD COMPLETADA EXITOSAMENTE")
