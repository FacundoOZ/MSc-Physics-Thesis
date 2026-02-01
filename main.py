
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo principal para correr los programas
#============================================================================================================================================

import base_de_datos.descarga     as data       # Descarga los datos
import base_de_datos.recorte      as edit       # Recorta los datos
import base_de_datos.unión        as merge      # Une los datos
import base_de_datos.conversiones as convert    # Conversiones entre magnitudes
import plots.MAG          as MAG  # Funciones para graficar 2D y 3D
import plots.SWEA         as SWEA # Funciones para graficar 2D y 3D
import plots.SWIA         as SWIA # Funciones para graficar 2D y 3D
import plots.animación_3D as ani  # Animación 3D de la trayectoria de MAVEN
import plots.estilo_plots
import ajustes.bow_shock as fit
import machine_learning.clasificador_KNN   as KNN # Algoritmo KNN binario supervisado (K-Nearest Neighbors)
#import machine_learning.validación_cruzada as CV  # Evaluación del modelo KNN con validación cruzada y métricas Recall, Precision y F1.

ruta: str = 'C:/Users/facuo/Documents/Tesis/MAG/'
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# APRENDIZAJE
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
knn = KNN.entrenar(
  directorio         = ruta,
  años_entrenamiento = ['2014','2015','2016','2017','2018'],
  K                  = 1,
  variables          = ['B','R','Bx','By','Bz','Xss','Yss','Zss'],
  promedio           = 5,
  ventana            = 60,
  ventanas_NBS       = [2],
)
knn.save(directorio=ruta, nombre_archivo='KNN_SALVATION_para2019.pkl')

KNN.clasificar(
  directorio         = ruta,
  knn                = KNN.Clasificador_KNN_Binario.load(directorio=ruta, nombre_archivo='KNN_SALVATION_para2019.pkl'),
  predecir_años      = ['2019']
)

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
  objetos          = ['Marte','Vignes','Fruchtman','mín','máx'],  # ['Marte','Vignes','Fruchtman','mín','máx','región','KNN']
# Mediciones de BS detectados por Fruchtman:
  años_Fruchtman   = ['2014','2015','2016','2017','2018','2019'], # ['2014',...,'2019']
  ajuste_Fruchtman = True,
# Trayectoria Cilíndrica de MAVEN:
  trayectoria      = False,
  recorte          = 'recorte_Vignes', # 'datos_recortados_merge' | 'hemisferio_N' | 'recorte_Vignes'
  tiempo_inicial   = '01/01/2015-00:00:00', # 'DD/MM/YYYY-HH:MM:SS'
  tiempo_final     = '30/01/2015-23:59:00', # 'DD/MM/YYYY-HH:MM:SS'
  promedio         = 5,                     # en segundos
# Mediciones de BS detectadas por el KNN
  años_KNN         = ['2014'], # ['2014',...,'2025']
  ajuste_KNN       = False
)"""

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# GRAFICOS Y ANIMACIONES
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
"""MAG.graficador(
  directorio     = ruta + 'datos_recortados_merge',# ó 'recorte_Vignes' | 'hemisferio_N' | 'hemisferio_ND'
# Intervalo de tiempo deseado:
  tiempo_inicial = '01/11/2015-00:00:00',
  tiempo_final   = '10/11/2015-23:59:00',
  promedio = 30, # Suavizado de los datos (reducción de ruido/fluctuaciones).
# Sistema de Referencia: 'ss' ó 'pc'
  coord          = 'ss',
# Magnitudes a graficar:
  B    = True,
  #B_x  = True,
  #B_y  = True,
  #B_z  = True,
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
  tamaño_puntos = 5,
# Mediciones BS detectadas por Fruchtman y/o por el KNN:
  bow_shocks = ['KNN']
)"""

"""ani.trayectoria_3D_MAVEN_MAG(
  directorio     = ruta + 'datos_recortados_merge',
  tiempo_inicial = '30/11/2014-00:00:00',
  tiempo_final   = '6/12/2014-23:59:00',
  promedio       = 1,
  paso           = 200,
  coord          = 'pc'
)"""

"""SWEA.graficador_distribución_angular(
  directorio     = ruta,
  archivo        = 'mvn_swe_l2_svypad_20141225_v05_r01.cdf',
  tiempo_inicial = '09:40:00',
  tiempo_final   = '10:05:00',
  promedio       = True
)"""

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# UNIÓN
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#merge.unir_archivo_MAG(directorio=ruta, archivo_pc='mvn_mag_l2_2022219pc1s_20220807_v01_r01_recortado.sts')
#merge.unir_paquete_MAG(directorio=ruta, año='2024')
#merge.unir_datos_fruchtman_MAG(directorio=ruta, año='2019')

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# RECORTE
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#edit.recortar_archivo_MAG(directorio=ruta, archivo='mvn_mag_l2_2024229ss1s_20240816_v01_r01.sts', coord='ss')
#edit.recortar_hemisferios_MAG(directorio=ruta, archivo='mvn_mag_l2_2014284merge1s_20141011_v01_r01_recortado.sts', hemisferio='norte')
#edit.recortar_paquete_MAG(directorio=ruta+'base_de_datos_pc', año='2024', coord='ss') # Recibe el año en que deseo cortar los datos
#edit.recortar_hemisferios_paquete_MAG(directorio=ruta, año='2016', hemisferio='norte') # o bien: hemisferio='norte_diurno'
#edit.recortar_datos_fruchtman_MAG(directorio=ruta+'fruchtman', archivo='Catálogo_Fruchtman_ss.txt', año=2014)
"""edit.recortar_Vignes_MAG(directorio=ruta, archivo='mvn_mag_l2_2015274merge1s_20151001_v01_r01_recortado_hemisferio_N.sts',
                         región=edit.preparar_región_Vignes())"""
#edit.recortar_Vignes_paquete_MAG(directorio=ruta, año='2025')

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# DESCARGA
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#data.descargar_archivo_MAG(directorio=ruta, dia='16', mes='8', año='2024', coord='ss')
#data.descargar_paquete_MAG(directorio=ruta+'base_de_datos', fecha_inicio='1/11/2023', fecha_final='31/12/2023', coord='ss')

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# CONVERSIÓN: día decimal <==> fecha UTC:
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#print(convert.dia_decimal_a_fecha_UTC(dia_decimal=1.29468, año=2018)) # Devuelve el día decimal en formato string 'AÑO-MES-DÍA HH:MM:SS'
#print(convert.fecha_UTC_a_dia_decimal(fecha_UTC='3/2/2015-07:04:28')) # Devuelve la fecha UTC en día decimal en formato float

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————