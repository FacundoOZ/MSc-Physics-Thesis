
# EDITAR

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para graficar magnitudes físicas medidas por SWEA: https://pds-ppi.igpp.ucla.edu/mission/MAVEN/maven/SWEA
#============================================================================================================================================

import numpy             as np
import matplotlib.pyplot as p
import matplotlib.colors as colors
import matplotlib.dates  as mdates # Permite realizar gráficos en formatos de fecha 'DD/MM/YYYY', 'HH:MM:SS', etc.
import cdflib                      # para poder leer archivos .cdf, Common Data Frame (NASA)
from datetime import datetime
from cdflib   import cdfepoch
from bs4      import BeautifulSoup

# Módulos Propios:
from base_de_datos.conversiones import tiempo_UTC_en_segundos

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# graficador_distribución_angular: grafica la distribución angular del paso de electrones del instrumento SWEA (Solar Wind Electron Analizer)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def graficador_distribución_angular(
    directorio: str,
    archivo: str,
    tiempo_inicial: str, tiempo_final: str,
    mínimo: float = 1e5,
    promedio: bool = False
) -> None:
  """
  La función graficador_distribución_angular permite graficar la distribución angular (survey o archive) del paso de electrones en unidades de diferencial de flujo de energía.

  Procedimiento:
    1. Ir al Link: https://pds-ppi.igpp.ucla.edu/collection/urn:nasa:pds:maven.swea.calibrated:data.svy_pad (Survey) (15 Hz) (mediciones cada 4s)
                ó  https://pds-ppi.igpp.ucla.edu/collection/urn:nasa:pds:maven.swea.calibrated:data.arc_pad (Archive) (30 Hz) (mediciones cada 2s)
    Survey: Tiene baja resolución temporal.
    Archive: Tiene alta resolución temporal.
    2. Seleccionar:
      - Start Time: Fecha de inicio.
      - Stop Time: Fecha de final.
    3. Aparecerá un único archivo (a lo sumo otro del día siguiente). Hacer click.
    4. Seleccionar el 2° ícono: Download product and data files.
    5. Extraer el archivo .zip
    6. El archivo deseado es el .cdf que será de la forma:
      'mvn_swe_l2_svypad_20141225_v05_r01.cdf'
    ó  'mvn_swe_l2_arcpad_20141225_v05_r01.cdf'
    7. Colocar en la carpeta '1.Códigos/MAVEN/SWEA'
  """
  cdf          = cdflib.CDF(directorio + 'swea/' + archivo)                      # Abrir el archivo
  energía      = cdf.varget('energy')                                            # Cargo bins de energía de tamaño: (E,)
  dt           = cdfepoch.to_datetime(cdf.varget('epoch'))                       # Tiempos de tamaño: (T,)
  flujo        = cdf.varget('diff_en_fluxes')                                    # Matriz de flujo de tamaño: (T, P, E)
  t0_seg       = tiempo_UTC_en_segundos(tiempo_inicial)                          #
  tf_seg       = tiempo_UTC_en_segundos(tiempo_final)                            #
  dt_seg       = (                                                               #
    (dt-np.array(dt,dtype='datetime64[D]')).astype('timedelta64[s]').astype(int) #
  )
  condicion    = (dt_seg >= t0_seg) & (dt_seg <= tf_seg)                         # Condición de filtrado de datos
  t_filtro     = dt[condicion]                                                   #
  flujo_filtro = np.mean(flujo[condicion], axis=1)                               # Promedio por distribución de ángulo de incl: eje=1 dim=(T,E)
  mapa_flujo   = flujo_filtro.T                                                  # Transpongo (dim = (E, T))
  t_array      = mdates.date2num(t_filtro)                                       #
  if promedio:                                                                   #
    promediar(mapa_flujo)                                                        # Promediado de los elementos x con sus vecinos
  fig, ax = p.subplots()                                                         # Mapa de calor en escala logarítmica
  pcm     = ax.pcolormesh(                                                       #
    t_array, energía, mapa_flujo,                                                #
    norm=colors.LogNorm(vmin=mínimo), cmap='viridis', shading='auto'             #
  )
  ax.set_yscale('log')                                                           # La escala y será logarítmica
  ax.invert_yaxis()                                                              # Invierto el eje y
  ax.set_ylim(energía[-1], energía[0])                                           # Invierto el orden
  ax.xaxis_date()                                                                # Eje x en formato de tiempos
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))                    #
  ax.set_xlabel('Tiempo UTC (HH:MM:SS)')                                         #
  ax.set_ylabel(r'$E_{\text{e}}$ [eV]')                                          #
  ax.set_title('Instrumento SWEA')                                               #
  cbar = fig.colorbar(pcm, ax=ax, label='Flujo de Energía (DEF)')                # Barra de colores del mapa de calor
  cbar.ax.text(                                                                  #
    -1.5, 0.5, r'$log_{10}(DEF)$ [m$^{-2}$sr$^{-1}$s$^{-1}$]',                   #
    va='center', ha='center', rotation=90, transform=cbar.ax.transAxes           #
  )
  fig.tight_layout()                                                             #
  p.show()                                                                       # (block=False) permite usar la terminal mientras veo el plot

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def promediar(mapa) -> None:
  """
  Documentación
  """
  def valido(estado):                                                 #
      return (not np.isnan(estado)) or (np.round(estado) != 0.0)      #
  def invalido(estado):                                               #
      return np.isnan(estado) or (np.round(estado) == 0.0)            #
  filas, columnas = mapa.shape                                        #
  for i in range(filas):                                              #
    for j in range(columnas-3):                                       #
      x0,x1,x2,x3 = mapa[i][j],mapa[i][j+1],mapa[i][j+2],mapa[i][j+3] #
      if valido(x0) and invalido(x1) and valido(x2):                  #
        mapa[i][j+1] = (x0+x2)/2                                      # Si los vecinos por izq y der de j+1 (inválido) son !=0, tomo promedio
      if valido(x0) and invalido(x1) and invalido(x2) and valido(x3): #
        mapa[i][j+1] = (x0+x3)/2                                      # Si los vecinos izq y der de j+1/j+2 (inválidos) son !=0, tomo promedio
        mapa[i][j+2] = (x0+x3)/2                                      #
      if valido(x0) and invalido(x1):                                 #
        mapa[i][j+1] = x0                                             # Si hay muchos inválidos, extrapolo j a j+1
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# graficador_omni_direccional: grafica la distribución omni-direccional del paso de electrones del instrumento SWEA
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def graficador_omni_direccional(directorio: str) -> None:
  """
  Documentación
  """
  with open(directorio + 'SWEA/' + file, 'r') as file:                      # Load and parse the XML
    soup = BeautifulSoup(file, 'xml')
  rows = soup.find_all('TR')
  timestamps = []
  columns = []
  for row in rows[3550:4000]:
    cells = row.find_all('TD')
    if len(cells) >= 6: # X value (time from index 2)
      try:
        timestamp = float(cells[2].text.strip())
      except ValueError:
        continue  # skip if timestamp is missing or invalid

      raw_array = cells[5].text.strip('[]') # Y array (from index 5)
      if raw_array:
        try:
          y_values = list(map(float, raw_array.split()))
          timestamps.append(timestamp)
          columns.append(y_values)
        except ValueError:
          continue  # skip malformed entries
  data = np.array(columns).T  # shape: (y_points, x_points)                 # Convert to 2D array and transpose to get shape (len(y), len(x))
  times = [datetime.utcfromtimestamp(ts) for ts in timestamps]
  extent = [mdates.date2num(times[0]), mdates.date2num(times[-1]), 1, data.shape[0]]
  #extent = [min(timestamps), max(timestamps), 0, data.shape[0]]            # x = time, y = index (Plot the heatmap)
  p.imshow(data, aspect='auto', extent=extent, cmap='viridis')
  #p.yscale('log')
  p.gca().xaxis_date()
  p.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
  p.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
  #p.gcf().autofmt_xdate()
  p.xlabel('tiempo (UNIX)')
  p.ylabel('$E_e$ [eV]')
  p.title('Mapa de calor 2D')
  p.tight_layout()
  p.show()

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————