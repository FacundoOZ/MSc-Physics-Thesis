
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
import machine_learning.validación_cruzada as CV  # Evaluación del modelo KNN con validación cruzada y métricas Recall, Precision y F1.

ruta: str = 'C:/Users/facuo/Documents/Tesis/MAG/'

# De los 3766 originales quedaron 1921 (el 51.009028 %) BS de Fruchtman tras el recorte z_pc > 0.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# APRENDIZAJE
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
"""knn = KNN.entrenar(
  directorio=ruta,
  años_entrenamiento=['2014','2015','2016','2017','2018','2019'],
  K=1,
  variables=['B','R','Bx','By','Bz','Xss','Yss','Zss'],
  promedio=5,
  ventana=120,
  ventanas_NBS=[2],
)
knn.save(directorio=ruta, nombre_archivo='knn_1.pkl')"""

#KNN.diagnosticar_knn(knn=KNN.Clasificador_KNN_Binario.load(directorio=ruta, nombre_archivo='knn_1.pkl'), directorio=ruta, año_test='2020')

"""KNN.clasificar(
  directorio=ruta,
  knn=KNN.Clasificador_KNN_Binario.load(directorio=ruta, nombre_archivo='knn_1.pkl'),
  predecir_años=['2018']
)"""

"""CV.ejecutar_validación_cruzada(
  directorio         = ruta,
  años_entrenamiento = ['2014','2015','2016','2017','2018','2019'], # con BS de Fruchtman
  K                  = 1,   # vecinos
  variables          = ['B','R','Bx','By','Bz','Xss','Yss','Zss'], # features
  promedio           = 5,   # en segundos
  ventana            = 120, # en segundos
  ventanas_NBS       = [2], # posición de ventanas_NBS respecto a ventanas_BS
  tolerancia         = 120  # en segundos
)"""

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# AJUSTES
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#fit.graficador_ajustes(directorio=ruta)
#fit.graficador_ajustes(directorio=ruta+'recorte_Vignes', tiempo_inicial='1/1/2015-00:00:00', tiempo_final='30/3/2015-23:59:00')
#fit.ajuste_Fruchtman(directorio=ruta+'fruchtman', año='2015')

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# GRAFICOS Y ANIMACIONES
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
MAG.graficador(
  directorio=ruta+'datos_recortados_merge',# ó 'hemisferio_N' ó 'hemisferio_ND'
# Intervalo de tiempo deseado
  tiempo_inicial='1/1/2018-00:00:00', tiempo_final='1/1/2018-23:59:00', promedio=30,
# Sistema de Referencia: 'ss' ó 'pc'
  coord='ss',
# Magnitudes a graficar:
  B=True,
  #B_x=True, B_y=True, B_z=True,
# Coordenadas Planeto-Céntricas (PC) (centradas en Marte):
  #z_pc=True, x_pc=True, y_pc=True,
# Coordenadas Sun-State (SS) ó Mars Solar Orbit (MSO):
  #x_ss=True, y_ss=True, z_ss=True,
#  cil=True, # Usar solamente con coord='ss' y trayectoria=True
# Distancia de MAVEN a Marte:
  #R=True,
# Curvas paramétricas:
#  trayectoria=True,
# Scatter:
  scatter=True,
  tamaño_puntos=5
)


"""ani.trayectoria_3D_MAVEN_MAG(directorio=ruta+'datos_recortados_merge',
                                tiempo_inicial = '30/11/2014-00:00:00', tiempo_final='6/12/2014-23:59:00', promedio=1,
                                paso=200,
                                coord='pc'
)"""

"""SWEA.graficador_distribución_angular(directorio=ruta,
                                     archivo='mvn_swe_l2_svypad_20141225_v05_r01.cdf',
                                     tiempo_inicial='09:40:00', tiempo_final='10:05:00',
                                     promedio=True
)"""

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# DESCARGA, RECORTE Y UNIÓN
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#data.descargar_archivo_MAG(directorio=ruta, dia='16', mes='8', año='2024', coord='ss')
#data.descargar_paquete_MAG(directorio=ruta+'base_de_datos', fecha_inicio='1/11/2023', fecha_final='31/12/2023', coord='ss') # DD/MM/YYYY
#edit.recortar_archivo_MAG(directorio=ruta, archivo='mvn_mag_l2_2024229ss1s_20240816_v01_r01.sts', coord='ss')
#edit.recortar_paquete_MAG(directorio=ruta+'base_de_datos_pc', año='2024', coord='ss') # Recibe el año en que deseo cortar los datos
#merge.unir_archivo_MAG(directorio=ruta, archivo_pc='mvn_mag_l2_2022219pc1s_20220807_v01_r01_recortado.sts')
#merge.unir_paquete_MAG(directorio=ruta, año='2024')
#edit.recortar_hemisferios_MAG(directorio=ruta, archivo='mvn_mag_l2_2014284merge1s_20141011_v01_r01_recortado.sts', hemisferio='norte')
#edit.recortar_hemisferios_paquete_MAG(directorio=ruta, año='2016', hemisferio='norte') # o bien: hemisferio='norte_diurno'
#edit.recortar_datos_fruchtman_MAG(directorio=ruta+'fruchtman', archivo='Catálogo_Fruchtman_ss.txt', año=2014)
#merge.unir_datos_fruchtman_MAG(directorio=ruta, año='2019')
#edit.recortar_Vignes_MAG(directorio=ruta, archivo='mvn_mag_l2_2015274merge1s_20151001_v01_r01_recortado_hemisferio_N.sts', región=edit.preparar_región_Vignes())
#edit.recortar_Vignes_paquete_MAG(directorio=ruta, año='2025')

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# CONVERSIÓN: día decimal <==> fecha UTC:
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#print(convert.dia_decimal_a_fecha_UTC(dia_decimal=1.0106192420023148, año=2018)) # Devuelve el día decimal en formato string 'AÑO-MES-DÍA HH:MM:SS'
#print(convert.dia_decimal_a_fecha_UTC(dia_decimal=1.0175636284027778, año=2018)) # Devuelve el día decimal en formato string 'AÑO-MES-DÍA HH:MM:SS'
#print(convert.dia_decimal_a_fecha_UTC(dia_decimal=1.1075636023958333, año=2018)) # Devuelve el día decimal en formato string 'AÑO-MES-DÍA HH:MM:SS'
#print(convert.dia_decimal_a_fecha_UTC(dia_decimal=1.183952555, año=2018)) # Devuelve el día decimal en formato string 'AÑO-MES-DÍA HH:MM:SS'
#print(convert.dia_decimal_a_fecha_UTC(dia_decimal=1.2047857120023149, año=2018)) # Devuelve el día decimal en formato string 'AÑO-MES-DÍA HH:MM:SS'
#print(convert.dia_decimal_a_fecha_UTC(dia_decimal=1.2946818720023148, año=2018)) # Devuelve el día decimal en formato string 'AÑO-MES-DÍA HH:MM:SS'
#print(convert.fecha_UTC_a_dia_decimal(fecha_UTC='3/2/2015-07:04:28')) # Devuelve la fecha UTC en día decimal en formato float

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————