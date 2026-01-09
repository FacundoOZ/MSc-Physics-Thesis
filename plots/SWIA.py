
# EDITAR

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para graficar magnitudes físicas medidas por SWIA: https://pds-ppi.igpp.ucla.edu/mission/MAVEN/maven/SWIA
#============================================================================================================================================

import numpy             as np
import matplotlib.pyplot as p
import matplotlib.colors as colors
import matplotlib.dates  as mdates # Permite realizar gráficos en formatos de fecha 'DD/MM/YYYY', 'HH:MM:SS', etc.
import cdflib                      # para poder leer archivos .cdf, Common Data Frame (NASA)

from cdflib import cdfepoch

from base_de_datos.conversiones import tiempo_UTC_en_segundos
from plots.SWEA                 import promediar

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# graficador_distribución_angular: grafica la distribución angular del paso de electrones del instrumento SWIA (Solar Wind Ion Analizer)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def graficador_distribución_angular(
    archivo: str,
    tiempo_inicial: str, tiempo_final: str,
    mínimo: float = 1e5,
    promedio: bool = False
) -> None:
  """
  Docstring para graficador_distribución_angular
  
  :param archivo: Descripción
  :type archivo: str
  :param tiempo_inicial: Descripción
  :type tiempo_inicial: str
  :param tiempo_final: Descripción
  :type tiempo_final: str
  :param mínimo: Descripción
  :type mínimo: float
  :param promedio: Descripción
  :type promedio: bool
  """
  cdf            = cdflib.CDF(archivo) # Abrir el archivo
  energía        = cdf.varget('energy')                       # Cargo bins de energía de tamaño: (E,)
  dt             = cdfepoch.to_datetime(cdf.varget('epoch'))  # Tiempos de tamaño: (T,)
  flujo          = cdf.varget('diff_en_fluxes')               # Matriz de flujo de tamaño: (T, P, E)
  t0_seg, tf_seg = tiempo_UTC_en_segundos(tiempo_inicial), tiempo_UTC_en_segundos(tiempo_final)
  dt_seg         = ((dt - np.array(dt, dtype='datetime64[D]')).astype('timedelta64[s]').astype(int))
  condicion      = (dt_seg >= t0_seg) & (dt_seg <= tf_seg)    # Condición de filtrado de datos
  t_filtro       = dt[condicion]
  flujo_filtro   = np.mean(flujo[condicion], axis=1)          # Promedio por distribución de ángulo de eje inclinación (eje=1) (la dimensión = (T, E))
  mapa_flujo     = flujo_filtro.T                             # Transpongo (dimensión = (E, T))
  t_array        = mdates.date2num(t_filtro)
  if promedio:
    promediar(mapa_flujo)                                     # Promediado de los elementos x con sus vecinos
  fig, ax = p.subplots(figsize=(12,3))                        # Mapa de calor en escala logarítmica
  pcm = ax.pcolormesh(t_array, energía, mapa_flujo, norm=colors.LogNorm(vmin=mínimo), cmap='viridis', shading='auto')
  ax.set_yscale('log')                                        # La escala y será logarítmica
  ax.invert_yaxis()                                           # Invierto el eje y
  ax.set_ylim(energía[-1], energía[0])                        # Invierto el orden
  ax.xaxis_date()                                             # Eje x en formato de tiempos
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
  ax.set_xlabel('Tiempo UTC (HH:MM:SS)')
  ax.set_ylabel(r'$E_{\text{e}}$ [eV]')
  ax.set_title('Instrumento SWEA')
  cbar = fig.colorbar(pcm, ax=ax, label='Flujo de Energía (DEF)')   # Barra de colores del mapa de calor
  cbar.ax.text(-1.5, 0.5, r'$log_{10}(DEF)$ [m$^{-2}$sr$^{-1}$s$^{-1}$]', va='center', ha='center', rotation=90, transform=cbar.ax.transAxes)
  fig.tight_layout()
  p.show() # (block=False) permite usar la terminal mientras veo el plot

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————