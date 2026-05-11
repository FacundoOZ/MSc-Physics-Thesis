
# COMENTAR

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para graficar magnitudes físicas medidas por SWEA: https://pds-ppi.igpp.ucla.edu/mission/MAVEN/maven/SWEA
#============================================================================================================================================

import os
import numpy             as np
import matplotlib.pyplot as p
import matplotlib.colors as colors
import matplotlib.dates  as mdates # Permite realizar gráficos en formatos de fecha 'DD/MM/YYYY', 'HH:MM:SS', etc.
import cdflib                      # para poder leer archivos .cdf, Common Data Frame (NASA)
from datetime import datetime
from cdflib   import cdfepoch
from bs4      import BeautifulSoup

# Módulos Propios:
from plots.estilo_plots         import guardar_figura
from base_de_datos.conversiones import tiempo_UTC_en_segundos

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# graficador_distribución_angular: grafica la distribución angular del paso de electrones del instrumento SWEA (Solar Wind Electron Analizer)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def graficador_distribución_angular(
    directorio: str,                                 # Carpeta base donde se encuentra SWEA/
    archivo: str,                                    # Nombre del archivo CDF
    tiempo_inicial: str, tiempo_final: str,          # Intervalo temporal en formato UTC
    mínimo: float = 1e5,                             # Valor mínimo para escala logarítmica
    promedio: bool = False,                          # Aplicar suavizado espacial
    guardar: bool = False                            #
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
  cdf: cdflib.cdfread.CDF = cdflib.CDF(os.path.join(directorio, 'SWEA', archivo))                  #
  t:     np.ndarray = cdfepoch.to_datetime(cdf.varget('epoch'))                                    #
  t_seg: np.ndarray = (t - np.array(t, dtype='datetime64[D]')).astype('timedelta64[s]').astype(int)#
  t0:    int        = tiempo_UTC_en_segundos(tiempo_inicial)                                       #
  tf:    int        = tiempo_UTC_en_segundos(tiempo_final)                                         #
  energía:        np.ndarray = cdf.varget('energy')                                                #
  flujo:          np.ndarray = cdf.varget('diff_en_fluxes')                                        #
  máscara:        np.ndarray = (t_seg >= t0) & (t_seg <= tf)                                       #
  t_filtrado:     np.ndarray = t[máscara]                                                          #
  flujo_filtrado: np.ndarray = flujo[máscara]                                                      #
  flujo_prom:     np.ndarray = np.mean(flujo_filtrado, axis=1)                                     #
  mapa_flujo:     np.ndarray = flujo_prom.T                                                        #
  if promedio:                                                                                     #
    promediar(mapa_flujo)                                                                          #
  fig, ax = p.subplots()                                                                           #
  pcm = ax.pcolormesh(mdates.date2num(t_filtrado), energía, mapa_flujo,                            #
                      norm=colors.LogNorm(vmin=mínimo), cmap='viridis', shading='auto')            #
  ax.set_yscale('log')                                                                             #
  ax.invert_yaxis()                                                                                #
  ax.set_ylim(energía[-1], energía[0])                                                             #
  ax.xaxis_date()                                                                                  #
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))                                      #
  ax.set_xlabel('Tiempo UTC (HH:MM:SS)')                                                           #
  ax.set_ylabel(r'$E_{\text{e}}$ [eV]')                                                            #
  ax.set_title('Mediciones del día 25/12/2014 (Survey)')                                           #
  ax.grid(which='major', alpha=.2,  linestyle='-')                                                 #
  ax.grid(which='minor', alpha=.15, linestyle=':')                                                 #
  cbar = fig.colorbar(pcm, ax=ax)                                                                  #
  cbar.set_label('Flujo Diferencial de Energía (DEF)', fontsize=10)                                #
  cbar.ax.text(-1.5, 0.5, r'$log_{10}(DEF)$ [m$^{-2}$sr$^{-1}$s$^{-1}$]',                          #
               va='center', ha='center', rotation=90, transform=cbar.ax.transAxes)                 #
  fig.tight_layout()                                                                               #
  if guardar:                                                                                      #
    guardar_figura()                                                                               #
  p.show()                                                                                         #

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def promediar(mapa) -> None:
  """
  Documentación
  """
  def valido(estado):                                                 #
    return (not np.isnan(estado)) or (np.round(estado) != 0.0)        #
  def invalido(estado):                                               #
    return np.isnan(estado) or (np.round(estado) == 0.0)              #
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
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————