import pandas as pd
import csv
import io

# Función para leer y limpiar un archivo dowjones-like
def leer_dowjones(path):
    with open(path, 'r', encoding='ISO-8859-1') as f:
        contenido = f.read().replace('""', '"')

    buffer = io.StringIO(contenido)
    lector = csv.reader(buffer, delimiter=',', quotechar='"')
    filas = list(lector)

    if len(filas) == 1:
        fila_unica = filas[0]
        datos = [fila_unica[i:i+7] for i in range(0, len(fila_unica), 7)]
    else:
        datos = [fila for fila in filas if len(fila) >= 7]

    # Cabecera esperada
    columnas = ["Fecha", "Último", "Apertura", "Máximo", "Mínimo", "Vol.", "% var."]

    # Filtrar filas basura y quitar primera si está malformada
    datos = [fila for fila in datos if not any("Fecha" in str(campo) for campo in fila)]
    datos = datos[1:]  # saltar la primera si es ruido

    df = pd.DataFrame([fila[:7] for fila in datos], columns=columnas)
    return df

# Leer ambos archivos
ruta1 = "D:/Usuario/3/TFG/Datos/gold1.csv"
ruta2 = "D:/Usuario/3/TFG/Datos/gold2.csv"

# Leer el primero como base, incluyendo su cabecera literal
with open(ruta1, 'r') as f:
    lineas = f.readlines()

# Extraer cabecera literal del primer archivo
cabecera_literal = lineas[0].strip()

# Leer ambos como DataFrame para combinar
df1 = leer_dowjones(ruta1)
df2 = leer_dowjones(ruta2)

# Concatenar
df_total = pd.concat([df1, df2], ignore_index=True)

# Convertir fechas y ordenar
df_total["Fecha"] = pd.to_datetime(df_total["Fecha"], format="%d.%m.%Y", errors="coerce")
df_total = df_total.dropna(subset=["Fecha"]).sort_values("Fecha", ascending=False).reset_index(drop=True)

# Formatear fecha
df_total["Fecha"] = df_total["Fecha"].dt.strftime("%d.%m.%Y")

# Exportar usando cabecera original (sin encoding forzado)
output_path = "D:/Usuario/3/TFG/Datos/gold.csv"
with open(output_path, 'w', newline='') as f:
    f.write(cabecera_literal + '\n')
    df_total.to_csv(f, index=False, header=False)

print("Archivos combinados, ordenados y exportados con cabecera original a:", output_path)
print(df_total.head())
