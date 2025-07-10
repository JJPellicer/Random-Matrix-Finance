import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats

# Cargar datos
archivo = "C:/Users/Juan/Documents/GitHub/Random-Matrix-Finance/Datos/sp500.csv"
df = pd.read_csv(archivo, delimiter=',')

# Convertir la columna de fecha al formato adecuado
df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d.%m.%Y")

# Convertir la columna "Último" a float (corrigiendo formato)
df["Último"] = df["Último"].str.replace(".", "", regex=False)  # Eliminar separadores de miles
df["Último"] = df["Último"].str.replace(",", ".", regex=False).astype(float)  # Reemplazar coma decimal y convertir a float

# Ordenar por fecha
df = df.sort_values(by="Fecha").reset_index(drop=True)

# Calcular rendimientos logarítmicos
df["Rendimiento"] = np.log(df["Último"] / df["Último"].shift(1))

# Calcular diferencia de rendimientos Delta R para un intervalo Δt
df["Delta_R"] = df["Rendimiento"].diff()

# Calcular promedios
promedio_rendimiento = df["Rendimiento"].mean()
promedio_delta_r = df["Delta_R"].mean()
print(f"Promedio del rendimiento logarítmico: {promedio_rendimiento:.6f}")
print(f"Promedio de Delta R: {promedio_delta_r:.6f}")

# Calcular la función de distribución del precio estimado
df["R_acumulado"] = df["Rendimiento"].cumsum()
df["Precio_Estimado"] = df["Último"].iloc[0] * np.exp(df["R_acumulado"])

# Graficar los rendimientos logarítmicos
plt.figure(figsize=(12, 6))
plt.plot(df["Fecha"], df["Rendimiento"], label="Rendimiento Logarítmico", marker='o', linestyle='-')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
plt.xticks(rotation=45)
plt.xlabel("Fecha")
plt.ylabel("Rendimiento Logarítmico")
plt.title("Evolución del Rendimiento Logarítmico del S&P 500")
plt.legend()
plt.grid(False)
plt.show()

# Graficar la variación de rendimientos (Delta R)
plt.figure(figsize=(12, 6))
plt.plot(df["Fecha"], df["Delta_R"], label="Delta R", marker='o', linestyle='-', color='r')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
plt.xticks(rotation=45)
plt.xlabel("Fecha")
plt.ylabel("Delta R")
plt.title("Variación del Rendimiento Logarítmico del S&P 500")
plt.legend()
plt.grid(False)
plt.show()

# Graficar la función de distribución del precio estimado
plt.figure(figsize=(12, 6))
plt.plot(df["Fecha"], df["Precio_Estimado"], label="Precio Estimado", marker='o', linestyle='-', color='g')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
plt.xticks(rotation=45)
plt.xlabel("Fecha")
plt.ylabel("Precio Estimado")
plt.title("Función de Distribución del Precio del S&P 500")
plt.legend()
plt.grid(False)
plt.show()

#-------------------------------------------------
# Análisis de la Distribución de los Rendimientos
#-------------------------------------------------

# Histograma de los rendimientos
plt.figure(figsize=(12, 6))
plt.hist(df["Rendimiento"].dropna(), bins=50, density=True, alpha=0.6, color='b', label="Histograma")

# Ajuste a una distribución normal
mu, sigma = stats.norm.fit(df["Rendimiento"].dropna())
x = np.linspace(df["Rendimiento"].min(), df["Rendimiento"].max(), 100)
pdf = stats.norm.pdf(x, mu, sigma)
plt.plot(x, pdf, 'r', label=f"Normal(μ={mu:.6f}, σ={sigma:.6f})")

plt.xlabel("Rendimiento Logarítmico")
plt.ylabel("Densidad de Probabilidad")
plt.title("Distribución de los Rendimientos del S&P 500")
plt.legend()
plt.grid(False)
plt.show()

# Q-Q Plot para comparar con la distribución normal
plt.figure(figsize=(12, 6))
stats.probplot(df["Rendimiento"].dropna(), dist="norm", plot=plt)
plt.title("Q-Q Plot de los Rendimientos del S&P 500")
plt.grid(False)
plt.show()
