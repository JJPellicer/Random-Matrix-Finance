import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
import scipy.stats as stats

# --------- Cargar y preparar datos (igual que en tu canvas) ---------
archivo = "C:/Users/Propietario/Desktop/TFG Juan/Random-Matrix-Finance-main/Datos/sp500.csv"
df = pd.read_csv(archivo, delimiter=',')

# Adaptación de formato
df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d.%m.%Y")
df["Último"] = df["Último"].str.replace(".", "", regex=False)
df["Último"] = df["Último"].str.replace(",", ".", regex=False).astype(float)

# Ordenar por fecha
df = df.sort_values(by="Fecha").reset_index(drop=True)

# Cálculo de rendimientos logarítmicos
df["Rendimiento"] = np.log(df["Último"] / df["Último"].shift(1))

# Fechas, precios y rendimientos
fechas = df["Fecha"].values
precios = df["Último"].values
rendimientos = df["Rendimiento"].values

# --------- Parámetros de animación ---------
paso = 5               # actualizar cada 5 días
ventana = 50           # usar solo los últimos 50 rendimientos para el histograma

# --------- Animación ---------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Evolución de precios
linea_precios, = ax1.plot([], [], lw=2)
ax1.set_xlim(fechas[0], fechas[-1])
ax1.set_ylim(np.min(precios)*0.95, np.max(precios)*1.05)
ax1.set_title('Evolución del S&P 500')
ax1.set_ylabel('Precio')
ax1.grid(True)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# Histograma de rendimientos
bins = np.linspace(np.nanmin(rendimientos), np.nanmax(rendimientos), 30)
bar_container = ax2.hist([], bins=bins, color='skyblue', edgecolor='black')[2]
ax2.set_xlim(np.nanmin(rendimientos), np.nanmax(rendimientos))
ax2.set_ylim(0, 10)  # Ajustar mejor el eje Y
ax2.set_title(f'Histograma de Rendimientos (últimos {ventana} días)')
ax2.set_xlabel('Rendimiento Logarítmico')
ax2.grid(True)

# Función de inicialización
def init():
    linea_precios.set_data([], [])
    for rect in bar_container:
        rect.set_height(0)
    return linea_precios, *bar_container

# Función de animación
def animate(i):
    idx = i * paso
    if idx == 0 or idx >= len(fechas):
        return init()

    # Actualizar evolución del precio
    linea_precios.set_data(fechas[:idx], precios[:idx])

    # Actualizar histograma de rendimientos en ventana móvil
    inicio = max(1, idx - ventana)
    datos_hist = rendimientos[inicio:idx]

    n, _ = np.histogram(datos_hist, bins=bins)
    for count, rect in zip(n, bar_container):
        rect.set_height(count)

    return linea_precios, *bar_container

# Crear animación
frames_total = len(fechas) // paso
ani = FuncAnimation(fig, animate, frames=frames_total, init_func=init, blit=True, interval=100)

plt.tight_layout()
plt.show()
