import pandas as pd
import requests
import time
import io

# Diccionario con los códigos exactos de las métricas
metricas = {
    'Points_Set': '2.2', 
    'Won_Attacks_Set': '2.7',
    'Won_Serves_Set': '2.8',
    'Exc_Receptions_Set': '2.10',
    'Exc_Blocks_Set': '2.9'
}

# Temporadas a extraer
temporadas = [2021, 2022, 2023, 2024, 2025]

# Lista para guardar todos los dataframes
todos_los_datos = []

# Cabeceras para simular un navegador real (Google Chrome en Windows) y evitar el Error 403
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8"
}

for nombre_metrica, id_classifica in metricas.items():
    for temporada in temporadas:
        url = f"https://www.legavolley.it/rendimento/?Tipo=2&Classifica={id_classifica}&AnnoInizio={temporada}&AnnoFine={temporada}&Serie=1&Fase=1&Giornata=0&Pos=40&lang=en"
        
        print(f"Extrayendo: {nombre_metrica} - Temporada {temporada}/{temporada+1}...")
        
        try:
            # 1. Hacemos la petición HTTP con nuestro "disfraz"
            respuesta = requests.get(url, headers=headers)
            
            # Comprobamos si la web nos ha devuelto un error de todos modos
            respuesta.raise_for_status() 
            
            # 2. Le pasamos el texto HTML a pandas (usando io.StringIO para evitar advertencias de pandas)
            tablas = pd.read_html(io.StringIO(respuesta.text))
            
            # Por lo general, la tabla que nos interesa es la primera (índice 0)
            if tablas:
                df = tablas[0] 
                
                # Añadimos columnas para identificar a qué estadística y año pertenece cada fila
                df['Metrica'] = nombre_metrica
                df['Temporada'] = f"{temporada}/{temporada+1}"
                
                todos_los_datos.append(df)
            else:
                print(f"No se detectaron tablas en la web para {nombre_metrica} - {temporada}.")
            
            # Pausa de 1.5 segundos entre peticiones para ser educados con el servidor
            time.sleep(1.5)
            
        except requests.exceptions.HTTPError as errh:
            print(f"Error HTTP en {nombre_metrica} ({temporada}): {errh}")
        except Exception as e:
            print(f"Error general extrayendo {nombre_metrica} de la temporada {temporada}: {e}")

# 3. Unir todo y exportar
if todos_los_datos:
    # Juntamos todas las tablas en una sola
    df_final = pd.concat(todos_los_datos, ignore_index=True)
    
    # Exportamos a Excel
    nombre_archivo = "Estadisticas_LegaVolley_Top40.xlsx"
    df_final.to_excel(nombre_archivo, index=False)
    print(f"\n¡Proceso terminado con éxito! Todos los datos se han guardado en '{nombre_archivo}'.")
else:
    print("\nNo se pudo extraer ningún dato. Revisa tu conexión o si la estructura de la web ha cambiado.")