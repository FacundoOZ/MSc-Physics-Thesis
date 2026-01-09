
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Estilo general para gráficos
#============================================================================================================================================

import numpy as np
import matplotlib.pyplot as p
from numpy             import pi, cos, sin
from matplotlib.colors import LinearSegmentedColormap

# Parámetros estándar de gráficos:
p.rcParams.update({
  'axes.labelsize': 15,                                                                     # Tamaño de etiquetas de los ejes,
  'xtick.labelsize': 10,                                                                    # Coordenadas eje x,
  'ytick.labelsize': 10,                                                                    # Coordenadas eje y,
  'legend.fontsize': 12,                                                                    # Leyenda
  'axes.prop_cycle': p.cycler('color', ['blue','red','green','orange','purple','yellow']),  # Colores
  'axes.grid': True,                                                                        # Cuadrícula
  'figure.figsize': (12,3),                                                                 # Tamaño de figura
  'xtick.minor.visible': True,                                                              # Sub-ejes en x
  'ytick.minor.visible': True,                                                              # e y
})

shade_m = LinearSegmentedColormap.from_list('shade_m', [(0,0,0), (1.0,0.5,0.0)]) # Color negro (0,0,0) a naranja (1,.5,0) para Marte.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# guardar_figura : 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def guardar_figura() -> None:
  """
  La función guardar_figura no recibe ningún parámetro. Permite guardar la imagen graficada con la mejor resolución posible (formato PDF).
  """
  respuesta = input('Presione ENTER para guardar .PDF (escriba para omitir): ')  # Con input, pregunto al usuario si desea guardar el plot.
  if respuesta == '':                                                            # Si presiona ENTER,
    p.savefig('plot_MAG.pdf', format='pdf', bbox_inches='tight')                 # se guarda en formato .PDF (mejor calidad)
    print(f'Se ha guardado correctamente.')                                      # y devuelve un mensaje.
  else:                                                                          # Si escribe algo y presiona ENTER,
    print('Figura no guardada.')                                                 # no se guarda.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# plot_xy : 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def plot_xy(x: np.ndarray, y: np.ndarray, etiqueta: str = None, scatter: bool = False, tamaño_puntos: int = 2) -> None:
  """
  La función plot_xy realiza el gráfico 2D de X contra Y de dos np.ndarrays pasados por parámetro. Coloca las etiquetas (en formato string)
  correspondientes, y si el parámetro booleano scatter es True, realiza un plot de scatter (por puntos) y permite ajustar el tamaño de
  esos puntos. Si scatter=False, realiza un plot común por interpolación. No devuelve nada. 
  """
  if scatter:                                        # Si scatter=True,
    p.scatter(x, y, s=tamaño_puntos, label=etiqueta) # => realizo gráfico por puntos sin interpolación, con tamaño de puntos y label.
  else:                                              # Si no,
    p.plot(x, y, label=etiqueta)                     # realizo un plot común por interpolación y coloco el label.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# disco_2D : 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def disco_2D(resolución_r: int = 100, resolución_theta: int = 200) -> None:
  """
  Genera las coordenadas (x,y) de un semi-disco unitario mediante coordenadas polares con y>0 (theta pertenece a [0,pi]). Los parámetros
  resolución_r y resolución_theta representan la resolución radial (desde (0,0) hasta el borde) y angular (alrededor del disco).
  """
  ax = p.figure().add_subplot(111)                         # Creo la figura y los ejes.
  r       = np.linspace(0,  1, resolución_r)               # Radio en el intervalo [0, 1].
  theta   = np.linspace(0, pi, resolución_theta)           # Ángulo polar en [0, pi].
  Theta,R = np.meshgrid(theta, r)                          # Malla polar.
  u,v     = R*cos(Theta), R*sin(Theta)                     # Conversión a coordenadas cartesianas x e y.
  ax.pcolormesh(u,v,(u+1)/2, cmap=shade_m, shading='auto') # Creo shade de color entre lado diurno/nocturno x>0/x<0 con función escalar u+1/2
  ax.set_aspect('equal', adjustable='box')                 # Tomo proporciones de los ejes iguales.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# esfera_3D : 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def esfera_3D(resolución: float = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Genera mediante el uso de coordenadas esféricas una tripla de coordenadas (x,y,z) que representa una esfera unitaria. El parámetro
  'resolución' representa la definición con la que se apreciará la esfera (predeterminado en 50). 
  """
  phi          = np.linspace(0, 2*pi, resolución)        # Creo un vector phi en el intervalo [0,2pi] con la resolución pasada por parámetro
  theta        = np.linspace(0,   pi, resolución)        # y un vector theta en el intervalo [0,pi].
  superficie_x = np.outer(cos(phi), sin(theta))          # Conversión a coordenadas cartesianas para la posición en x,
  superficie_y = np.outer(sin(phi), sin(theta))          # para la posición en y,
  superficie_z = np.outer(np.ones_like(phi), cos(theta)) # y para la posición en z mediante coordenadas esféricas.
  return (superficie_x, superficie_y, superficie_z)      # Devuelvo una tripla que representa a la esfera 3D.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————