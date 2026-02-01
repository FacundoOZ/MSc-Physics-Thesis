
# Editar

#============================================================================================================================================
# Tesis de Licenciatura | 
#============================================================================================================================================

import os

# Módulos Propios:

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# recortar_archivo_MAG: función para recortar 1 único archivo
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def promediar_archivo_temporal_KNN(
    directorio: str,                                                                   # Carpeta donde se encuentra el archivo a recortar.
    promedio: int = 40,                                                                # 
    año: str = '2014'                                                                  # Tipo de coordenadas del archivo a recortar.
) -> None:
  """
  Doc
  """
  archivo_KNN: str   = f'tiempos_BS_{año}.txt'
  ruta_KNN: str      = os.path.join(directorio, 'KNN', 'predicción', archivo_KNN)
  ruta_final: str    = os.path.join(directorio, 'KNN', 'predicción', 'post_procesamiento')
  os.makedirs(ruta_final, exist_ok=True)
  archivo_final: str = os.path.join(ruta_final,archivo_KNN.replace('.txt','_promedio.txt'))
  if os.path.exists(ruta_KNN):
    print(f"El archivo '{os.path.basename(archivo_final)}' ya ha sido promediado.")
    return
  


"""  try:
    data = pd.read_csv(ruta_i, sep=r'\s+', header=None, skiprows=160, engine='python') # Extraigo los datos de ruta_i, y omito 160 líneas
    datos_recortados = data.iloc[:, columnas]                                          # En 'datos_recortados' extraigo las columnas deseadas
    datos_recortados.to_csv(recortado, sep=' ', index=False, header=False)             # Convierto ruta final (con nombre) a CSV y sep. TAB
    #print(f'El archivo "{os.path.basename(archivo)}" se ha recortado.')               # Omitir esto para usar todo el paquete.
  except FileNotFoundError:                                                            # Si el archivo de origen no está,
    print(f'El archivo "{os.path.basename(archivo)}" no se ha encontrado.')            # devuelve error.
  except Exception as e:                                                               # Si hay algún otro tipo de error,
    print('El archivo', os.path.basename(archivo), '->', e)                            # creo un aviso.
"""

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————