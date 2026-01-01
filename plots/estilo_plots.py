
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Estilo general para gráficos
#============================================================================================================================================

import matplotlib.pyplot as p

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

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————