
#============================================================================================================================================
# Tesis de Licenciatura | Archivo principal para correr los programas
#============================================================================================================================================
import base_de_datos.conversiones          as convert # Conversiones entre magnitudes
import base_de_datos.descarga              as data    # Descarga los datos
import base_de_datos.recorte               as edit    # Recorta los datos
import base_de_datos.unión                 as merge   # Une los datos
import base_de_datos.promedio              as avg     # Promedia datos
import machine_learning.clasificador_KNN   as KNN     # Algoritmo KNN binario supervisado (K-Nearest Neighbors)
import machine_learning.métricas           as metric  #
import machine_learning.validación_cruzada as CV      # Evaluación del modelo KNN con validación cruzada y métricas Recall, Precision y F1.
import ajustes.bow_shock                   as fit     #
import plots.MAG                           as MAG     # Funciones para graficar 2D y 3D
import plots.SWEA                          as SWEA    # Funciones para graficar 2D y 3D
import plots.SWIA                          as SWIA    # Funciones para graficar 2D y 3D
import plots.animación_3D                  as ani     # Animación 3D de la trayectoria de MAVEN
import plots.estilo_plots                             # Estilo de gráficos

ruta: str = 'C:/Users/facuo/Documents/Tesis/MAG/'

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Evolución temporal de x_0 en el último ciclo solar (obtenido por k-NN Eclipse Optimizado)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
"""import matplotlib.pyplot as p
import numpy as np
from plots.estilo_plots import guardar_figura

p.figure()
x     = ['2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024','2025']
y     = [np.nan,0.55547660,0.60006849,0.46524696,0.52887940,0.48749612,0.19490869,0.46060805,0.56612123,0.39803001,0.60197869,0.56655043]
y_err = [np.nan,0.14569734,0.25445534,0.30989747,0.19435214,0.21452367,0.23086488,0.15603615,0.23940949,0.21028380,0.22596452,0.20107036]

p.errorbar(x[:6], y[:6], yerr=y_err[:6], fmt='o-', marker='o', capsize=4, color='green', label='Órbita Science')
p.errorbar(x[5:], y[5:], yerr=y_err[5:], fmt='o-', marker='o', capsize=4, color='blue', label='Órbita Science & Relay')
p.xticks(x)
p.title('Parámetro $x_0$ ajustado de los BS $k$-NN (Eclipse Optimizado) respecto del último ciclo solar')
p.ylabel(r'Parámetro $x_0$')
p.xlabel('Año')
p.grid(which='major', alpha=.2,  linestyle='-')                                 # Uso parámetros estándar de ejes principales,
p.grid(which='minor', alpha=.15, linestyle=':')                                 # y secundarios suaves para todos los gráficos.
p.legend()
guardar_figura()
p.show()"""
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# APRENDIZAJE
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
"""knn = KNN.entrenar(
  directorio         = ruta,
  años_entrenamiento = ['2014','2015','2016','2017','2018','2019'], # BS Fruchtman
  hemisferio_N       = False,                                       # Uso los 3766 BS
  K                  = 3,
  variables          = ['B','R','Bx','By','Bz','Xss','Yss','Zss'],
  promedio           = 15,                                          # en segundos
  ventana            = 120,                                         # en segundos
  ventanas_NBS       = [-2,2,4,6],                                  # las ventanas_BS son = [0]
)
knn.save(directorio=ruta, nombre_archivo=f'Eclipse_FINAL.pkl')"""

"""años: list[str] = ['2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024','2025']
for año in años:
  KNN.clasificar(
    directorio    = ruta,
    knn           = KNN.Clasificador_KNN_Binario.load(directorio=ruta, nombre_archivo=f'Eclipse_FINAL.pkl'),
    predecir_años = [año],
    nombre_modelo = 'Eclipse_FINAL' # Carpeta donde se guardarán los archivos nuevos
  )
  avg.promediar_archivo_temporal_KNN(directorio=ruta, modelo='Eclipse_FINAL', año=año, promedio=600)"""

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# VALIDACIÓN CRUZADA "A MANO"
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
"""años: list[str] = ['2014','2015','2016','2017','2018','2019']
for año in años:
  knn = KNN.entrenar(# IMPORTANTE => entreno con TODOS (los 3766 BS Fruchtman del hemisferio NORTE Y SUR)
    directorio         = ruta,
    años_entrenamiento = [x for x in años if x!=año],
    hemisferio_N       = False,
    K                  = 12,
    variables          = ['B','R','Bx','By','Bz','Xss','Yss','Zss'],
    promedio           = 5,  # en segundos
    ventana            = 60, # en segundos
    ventanas_NBS       = [-3,-2,2,3],# entero para la posición de ventanas a utilizar como supervisión no-BS (las ventanas BS son [0]).
  )
  knn.save(directorio=ruta, nombre_archivo=f'Eclipse_k12{año}.pkl')
  KNN.clasificar(
    directorio    = ruta,
    knn           = KNN.Clasificador_KNN_Binario.load(directorio=ruta, nombre_archivo=f'Eclipse_k12{año}.pkl'),
    predecir_años = [año],
    nombre_modelo = 'Eclipse_k12' # Carpeta donde se guardarán los archivos nuevos
  )
  avg.promediar_archivo_temporal_KNN(directorio=ruta, modelo='Eclipse_k12', año=año, promedio=600)
metric.calcular_métricas_KNN_con_Fruchtman(# estoy evaluando con los 3766 BS Fruchtman recortados por HEMISFERIO NORTE (mil y pico).
  directorio         = ruta,
  años               = años,
  modelo_KNN         = 'Eclipse_k12',
  post_procesamiento = True,
  hemisferio_N       = True,
  tolerancia         = 120
)"""

#—————————————————————
# Gráficos de Métricas
#—————————————————————
"""metric.graficador_parámetros_KNN(
  directorio         = ruta,
  parámetro          = 'ventanas_NBS', # 'promedio' | 'ventanas_NBS' | 'K' | 'tolerancia'
  post_procesamiento = True,
  métricas           = ['F1'],
  errores            = False,
  guardar            = False
)"""

#———————————————————————————————————————————————————————————————————————————————————————
# TESTING & CROSS-VALIDATION (PELIGROSO MUCHA RAM)
#———————————————————————————————————————————————————————————————————————————————————————
#KNN.diagnosticar_knn(knn=KNN.Clasificador_KNN_Binario.load(directorio=ruta, nombre_archivo='knn_1.pkl'), directorio=ruta, año_test='2020')

"""CV.ejecutar_validación_cruzada(
  directorio         = ruta,
  años_entrenamiento = ['2014','2015','2016','2017','2018','2019'], # con BS de Fruchtman
  K                  = 1,   # vecinos
  variables          = ['B','R','Bx','By','Bz','Xss','Yss','Zss'], # features
  promedio           = 5,   # en segundos
  ventana            = 60, # en segundos
  ventanas_NBS       = [2], # posición de ventanas_NBS respecto a ventanas_BS
  tolerancia         = 120 # en segundos
)"""

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# AJUSTES
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
"""fit.graficador_ajustes(
  directorio       = ruta,
# Elementos que tendrá el plot:
  objetos          = ['Marte','Vignes','Fruchtman'],  # ['Marte','Vignes','Fruchtman','mín','máx','región','KNN','propios']
# Mediciones de BS detectados por Fruchtman:
  años_Fruchtman   = ['2014'], # ['2014',...,'2019']
  ajuste_Fruchtman = True,
  hemisferio_N     = True, # Si es igual a False, grafica todos los bow shocks detectados por Fruchtman. Si no, grafica solo los del norte.
# Trayectoria Cilíndrica de MAVEN:
  #trayectoria      = True,
  #recorte          = 'recorte_Vignes',      # 'datos_recortados_merge' | 'hemisferio_N' | 'recorte_Vignes'
  #tiempo_inicial   = '01/01/2017-00:00:00', # 'DD/MM/YYYY-HH:MM:SS'
  #tiempo_final     = '31/03/2017-23:59:00', # 'DD/MM/YYYY-HH:MM:SS'
  #promedio         = 1,                     # en segundos
# Mediciones de BS detectadas por el KNN
  #modelo             = 'Eclipse_FINAL',
  #post_procesamiento = True,
  #años_KNN           = ['2016'], # ['2014',...,'2025']
  #ajuste_KNN         = True,
# Mediciones de BS catalogadas por mi + Fruchtman:
  #ajuste_prop = True
# Guardar figura en formato .PDF:
  guardar = True
)"""

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# GRAFICOS Y ANIMACIONES
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#for hora in ['00:57:00','02:01:30','06:50:00','10:06:40','11:26:15','14:41:40','15:50:30']:
#  print(convert.fecha_UTC_a_dia_decimal(fecha_UTC='16/11/2014-' + hora)) # Devuelve la fecha UTC en día decimal en formato float

"""MAG.graficador(
  directorio     = ruta + 'datos_recortados_merge',# 'datos_recortados_merge' | 'recorte_Vignes' | 'hemisferio_N' | 'hemisferio_ND'
# Intervalo de tiempo deseado:
  tiempo_inicial = '26/1/2018-06:00:00',
  tiempo_final   = '26/1/2018-07:10:00',
  promedio       = 1, # Suavizado de los datos (reducción de ruido/fluctuaciones).
# Sistema de Referencia: 'ss' ó 'pc'
  coord          = 'ss',
# Magnitudes a graficar:
  B    = True,
  #B_x  = True,
  #B_y  = True,
  #B_z  = True,
  #normalización = True,
  #x_pc = True, # Coordenadas Planeto-Céntricas (PC) (centradas en Marte).
  #y_pc = True,
  #z_pc = True,
  #x_ss = True, # Coordenadas Sun-State (SS) ó Mars Solar Orbit (MSO).
  #y_ss = True,
  #z_ss = True,
  #cil  = True, # Usar solamente con coord='ss' y trayectoria=True.
  #R    = True, # Distancia de MAVEN a Marte.
# Curvas paramétricas 2D y 3D:
  #trayectoria   = True,
# Interpolación:
  scatter       = True,
  tamaño_puntos = 1,
# Mediciones BS detectadas por Fruchtman y/o por el KNN:
  bow_shocks         = ['KNN','Fruchtman'], # ó ['KNN','Fruchtman','propios']
  modelo_KNN         = 'Eclipse_FINAL',
  post_procesamiento = True,
# Guardar figura en formato .PDF:
  guardar = True
)"""

"""ani.trayectoria_3D_MAVEN_MAG(
  directorio     = ruta + 'datos_recortados_merge',
  tiempo_inicial = '30/11/2014-00:00:00',
  tiempo_final   = '30/11/2014-23:59:00',
  promedio       = 1,
  paso           = 600,
  coord          = 'pc'
)"""

"""SWEA.graficador_distribución_angular(
  directorio     = 'C:/Users/facuo/Documents/Tesis/',
  archivo        = 'mvn_swe_l2_svypad_20141225_v05_r01.cdf',
  tiempo_inicial = '09:40:00',
  tiempo_final   = '10:05:00',
  promedio       = False,
  guardar        = True
)"""

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# UNIÓN
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#merge.unir_archivo_MAG(directorio=ruta, archivo_pc='mvn_mag_l2_2022219pc1s_20220807_v01_r01_recortado.sts')
#merge.unir_paquete_MAG(directorio=ruta, año='2025')
#merge.unir_datos_fruchtman_MAG(directorio=ruta, año='2014')

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# RECORTE
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#edit.recortar_archivo_MAG(directorio=ruta, archivo='mvn_mag_l2_2014359ss_20141225_v01_r01.sts', coord='ss')
#edit.recortar_hemisferios_MAG(directorio=ruta, archivo='mvn_mag_l2_2014284merge1s_20141011_v01_r01_recortado.sts', hemisferio='norte')
#edit.recortar_paquete_MAG(directorio=ruta+'base_de_datos_pc', año='2025', coord='pc') # Recibe el año en que deseo cortar los datos
#edit.recortar_hemisferios_paquete_MAG(directorio=ruta, año='2025', hemisferio='norte') # o bien: hemisferio='norte_diurno'
#edit.recortar_datos_fruchtman_MAG(directorio=ruta+'fruchtman', archivo='Catálogo_Fruchtman.txt', año=2019)
"""edit.recortar_Vignes_MAG(directorio=ruta, archivo='mvn_mag_l2_2015274merge1s_20151001_v01_r01_recortado_hemisferio_N.sts',
                         región=edit.preparar_región_Vignes())"""
#edit.recortar_Vignes_paquete_MAG(directorio=ruta, año='2025')

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# DESCARGA
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#data.descargar_archivo_MAG(directorio=ruta, dia='25', mes='12', año='2014', coord='ss')
#data.descargar_paquete_MAG(directorio=ruta+'base_de_datos_pc', fecha_inicio='1/8/2025', fecha_final='30/11/2025', coord='pc')

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# CONVERSIÓN: día decimal <==> fecha UTC:
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#print(convert.dia_decimal_a_fecha_UTC(dia_decimal=1.29468, año=2018)) # Devuelve el día decimal en formato string 'AÑO-MES-DÍA HH:MM:SS'

#for fecha in ['05:28:00','07:20:00','09:55:30','11:50:00','15:00:00','19:12:30','23:48:55']:
#  print(convert.fecha_UTC_a_dia_decimal(fecha_UTC='29/12/2014-' + fecha)) # Devuelve la fecha UTC en día decimal en formato float

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————