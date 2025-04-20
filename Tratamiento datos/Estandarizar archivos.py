import pandas as pd
import csv
import io

# Ruta original en tu equipo
archivo = "D:/Usuario/3/TFG/Datos/bovespa.csv"

# Lectura personalizada con manejo de comillas dobles "" dentro de campos
with open(archivo, 'r', encoding='ISO-8859-1') as f:
    contenido = f.read().replace('""', '"')

# Convertir a buffer para csv.reader
buffer = io.StringIO(contenido)
lector = csv.reader(buffer, delimiter=',', quotechar='"')
filas = list(lector)

# Aplanar todo si hay solo una fila larga
if len(filas) == 1:
    fila_unica = filas[0]
    datos = [fila_unica[i:i+7] for i in range(0, len(fila_unica), 7)]
else:
    datos = [fila for fila in filas if len(fila) >= 7]

# Cabecera manual
columnas = ["Fecha", "Último", "Apertura", "Máximo", "Mínimo", "Vol.", "% var."]

# Filtrar cualquier fila basura con "Fecha" entre los valores
datos = [fila for fila in datos if not any("Fecha" in str(campo) for campo in fila)]

# Eliminar posible primera fila desplazada o corrupta
datos = datos[1:]

# Crear DataFrame
df = pd.DataFrame([fila[:7] for fila in datos], columns=columnas)

# Añadir ".00" al final de valores que no tienen decimales
columnas_numericas = ["Último", "Apertura", "Máximo", "Mínimo"]
for col in columnas_numericas:
    df[col] = df[col].apply(lambda x: x if "," in x else x + ",00")

# Mostrar primeras filas
print("Datos limpios (formato con ,00 agregado si faltaba decimal):")
print(df.head())

# Exportar a nuevo archivo limpio con codificación UTF-8 compatible
output_path = "D:/Usuario/3/TFG/Datos/bovespa_corregido.csv"
df.to_csv(output_path, index=False, sep=',', encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
print(f"Archivo exportado como: {output_path}")
