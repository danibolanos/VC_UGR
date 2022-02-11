# Alumno : Daniel Bolaños Martínez 76592621E
# Asignatura : Visión por Computador
# Práctica 0 : Introducción a OpenCV

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Ejercicio 1 : Escribir una función que lea el fichero de una imagen y la
# muestre tanto en grises como en color

def leeimagen(filename, flagColor):
  # filename: nombre de la imagen
  # flagColor: 1 para color y 0 para blanco y negro
  if flagColor:
    im = cv2.imread(filename)
  else:
    im = cv2.imread(filename, 0)
  return im

# Ejercicio 2 : Escribir una función que visualice una matriz de números 
# reales cualquiera ya sea monobanda o tribanda. Deberá escalar y 
# normalizar sus valores.
  
def normalizaRGB(im):
  # im: imagen a normalizar y escalar entre 0-255
  # pasamos la matriz a valores flotantes
  im = im.astype(np.float64) 
  
  # si la imagen es en blanco y negro
  if len(im.shape) == 2:
    # calculamos max y min
    vmax = np.max(im)
    vmin = np.min(im)
    # normalizamos la matriz entre 0-1
    if vmax != vmin:
      im = (im - vmin) / (vmax - vmin)
      # escalamos la matriz a los valores 0-255 RGB
      im = im*255.0
    # si maximo == minimo
    else:
      im = im*0.0
      
  # si la imagen es en color
  else: 
    # calculamos max y min
    vmax = np.max(im, (0,1))
    vmin = np.min(im, (0,1))
    # normalizamos la matriz entre 0-1
    if vmax[0]!=vmin[0] and vmax[1]!=vmin[1] and vmax[2]!=vmin[2]:
      im = (im - vmin) / (vmax - vmin)
      # escalamos la matriz a los valores 0-255 RGB
      im = im*255.0
    # si algún maximo == minimo para algún canal
    else:
      im = im*0.0
      
  # cambiamos el tipo de datos de float64 a uint8
  im = im.astype(np.uint8)
  return im
    
    
def pintaI(im):
  # im: imagen a visualizar
  im = normalizaRGB(im)
  # activamos los colores RGB para plt
  imgrgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  # mostramos la imagen con plt.imshow()
  plt.imshow(imgrgb)
  # añadimos un título por defecto
  plt.title("Imagen")
  plt.show()
  return imgrgb
  
# Ejercicio 3 : Escribir una función que visualice varias imágenes 
# a la vez ¿Qué pasa si las imágenes no son todas del mismo tipo: 
# (nivel de gris, color, blanco-negro)? 

# R: Si no son del mismo tipo, nos aparecerá un error que se soluciona
# pasando a color todas las imágenes de la lista. 
# Esto ocurre porque las imágenes a color tienen 2 canales más 
# que las de blanco y negro, es decir, mientras que las imágenes en 
# escala de grises son matrices de enteros (0-255), las imágenes a 
# color son matrices de ternas de enteros (0-255).
  
def concatenaImgs(vim):
  # vim: lista de imagenes a concatenar
  # obtenemos la mayor altura de las imágenes (máximo filas)
  altura = max(im.shape[0] for im in vim)
  for i,im in enumerate(vim):
    # si la imágen está en blanco y negro, pasamos a color
    if len(im.shape) == 2:
      vim[i] = cv2.cvtColor(vim[i], cv2.COLOR_GRAY2BGR)
    # redimensionamos imágenes con número de filas menor que altura
    if im.shape[0] < altura:
      diff = abs(vim[i].shape[0]-vim[i].shape[1])
      vim[i] = cv2.resize(vim[i], (vim[i].shape[1]+diff, altura)) 
  # concatenamos las imágenes de la lista en horizontal
  imglist = cv2.hconcat(vim)
  return imglist
  
def pintaMI(vim):
  # vim: lista de imagenes a concatenar
  imgs = concatenaImgs(vim)
  # pintamos el resultado
  pintaI(imgs)
  
# Ejercicio 4 : Escribir una función que modifique el color en la imagen 
# de cada uno de los elementos de una lista de coordenadas de píxeles.

def modPixels(im, pixels, color):
  # im: imagen a modificar
  # pixels: vector de coordenadas de los pixels a modificar
  # color: nuevo color que obtendrán los pixels especificados 
  for i in pixels:
    # si la imagen es en blanco y negro usamos primer valor de color
    if len(im.shape) == 2:
      im[i] = color[0]
    # si es en color
    else:
      im[i] = color
      
# Ejercicio 5 : Una función que sea capaz de representar varias
# imágenes con sus títulos en una misma ventana.
  
def pintaAll(vim, titles):
  # vim: lista de imágenes
  # titles: lista de títulos de las imágenes
  imgs = concatenaImgs(vim)
  imgs = normalizaRGB(imgs)
  # concatena los títulos de las imágenes
  separador = ' '
  new_name = separador.join(titles)
  # activamos los colores RGB para plt
  imgrgb = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
  # mostramos la imagen por pantalla con plt
  plt.imshow(imgrgb)
  # añadimos el nuevo título
  plt.title(new_name)
  plt.show()
  
    
if __name__ == '__main__':    

  img = './images/orapple.jpg'
  otra_img = './images/dave.jpg'
    
  # Ejercicio 1
  # Leemos las imágenes. 1 color y 0 escala de grises
    
  img_BW = leeimagen(img, 0)
  img_C = leeimagen(img, 1)
  img_O = leeimagen(otra_img, 0)
    
  # Ejercicio 2
  # Dibujamos dos de las imágenes leídas 
  
  pintaI(img_BW)
  pintaI(img_C)
  
  # Nota: normaliza y dibuja una matriz aleatoria (grises y color)
  random_BW = np.random.randn(100, 150)
  random_C = np.random.randn(100, 150, 3)
  random_BW = pintaI(random_BW)
  random_C = pintaI(random_C)
    
  # Ejercicio 3
  # Dibujamos dos imágenes (una en color y otra en grises) concatenadas

  array = [img_C, img_O]
  pintaMI(array)
  
  # Ejercicio 4
  # Modificamos un cuadrante a color negro [0,0,0]
  
  img_mod = np.copy(img_C)
  
  modPixels(img_mod, [(x,y) for x in range(int(img_mod.shape[0]/2)) 
  for y in range(int(img_mod.shape[1]/2))], [0,0,0])
  pintaI(img_mod)
  
  # Ejercicio 5
  # Dibujamos tres imágenes con sus respectivos títulos
  
  pintaAll([img_C, random_BW, img_O], ["Orapple Color", "Aleatorio", "Dave"])
  
  # Nota: también sirve para poner título a una sola imagen
  pintaAll([img_C], ["Orapple"])