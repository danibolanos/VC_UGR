# Asignatura : Visión por Computador
# Práctica 1 : Filtrado y Detección de regiones
# Curso 2019/2020
# Alumno : Daniel Bolaños Martínez 76592621E

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

# Funciones Práctica 0

# Lee una imagen con el flag del color especificado
def leeimagen(filename, flagColor):
  # filename: nombre de la imagen
  # flagColor: 1 para color y 0 para blanco y negro
  if flagColor:
    im = cv2.imread(filename)
  else:
    im = cv2.imread(filename, 0)
  return im

# Normaliza una matriz de flotantes y la escala al rango 0-255
def normalizaRGB(im):
  # im: imagen a normalizar y escalar entre 0-255
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
      
  return im

# Muestra la imagen y le pone un título    
def pintaI(im, title=""):
  # im: imagen a visualizar
  # normalizamos los valores y pasamos a formato uint8
  im = normalizaRGB(im)
  im = im.astype(np.uint8)
  # activamos los colores RGB para plt
  imgrgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  # mostramos la imagen con plt.imshow()
  plt.imshow(imgrgb)
  # añadimos un título por defecto
  plt.title(title)
  plt.xticks([]), plt.yticks([])
  plt.show()
  return imgrgb

# Concatena varias imágenes de forma horizontal
def concatenaImgs(vim):
  # vim: lista de imagenes a concatenar
  # obtenemos la mayor altura de las imágenes (máximo filas)
  hmax = max(im.shape[0] for im in vim)
  for i,im in enumerate(vim):
    # normalizamos los valores y pasamos a formato uint8
    vim[i] = normalizaRGB(vim[i])
    vim[i] = vim[i].astype(np.uint8)
    # si la imágen está en blanco y negro, pasamos a color
    if len(im.shape) == 2:
      vim[i] = cv2.cvtColor(vim[i], cv2.COLOR_GRAY2BGR)
    # si la imagen es más pequeña, añadimos un borde superior
    # de tamaño=(altura-filas)
    if im.shape[0] < hmax:
      vim[i] = cv2.copyMakeBorder(vim[i], top=hmax-im.shape[0], bottom=0, 
      left=0, right=0, borderType=cv2.BORDER_CONSTANT)
  # concatenamos las imágenes de la lista en horizontal
  imglist = cv2.hconcat(vim)
  return imglist

# Muestra varias imágenes y les pone sus títulos correspondientes  
def pintaAll(vim, titles=""):
  # vim: lista de imágenes
  # titles: lista de títulos de las imágenes
  imgs = concatenaImgs(vim)
  # concatena los títulos de las imágenes
  separador = '  |  '
  new_name = separador.join(titles)
  # activamos los colores RGB para plt
  imgrgb = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
  # mostramos la imagen por pantalla con plt
  plt.imshow(imgrgb)
  # añadimos el nuevo título
  plt.title(new_name)
  plt.xticks([]), plt.yticks([])
  plt.show()
  
# Funciones Práctica 1
  
# Ejercicio 1: Implementar las siguientes funciones usando OpenCV.
  
# A. El cálculo de la convolución de una imagen con una máscara 2D.
# Usar una Gaussiana 2D y máscaras 1D.

def funcionGaussiana(im, sigmaX, sigmaY=0, ksizeX=0, ksizeY=0, 
                     border=cv2.BORDER_DEFAULT):
  # im: imagen a visualizar
  # sigmaX: valor sigma en dirección X
  # sigmaY: valor sigma en dirección Y
  # ksizeX: kernel size en la dirección X
  # ksizeY: kernel size en la dirección Y
  # border: especifica el tipo de borde
  # si sigmaY es cero, tomamos el mismo valor que sigmaX
  if sigmaY == 0:
    sigmaY = sigmaX
  # calculamos ksize a partir de sigma con la relación vista en clase
  # tomamos el 95% de la distribución normal de la Gaussiana centrada
  # en el origen
  if ksizeX == 0 or ksizeY == 0:
    ksizeX = 2*math.floor(3*sigmaX)+1
    ksizeY = 2*math.floor(3*sigmaY)+1
  # comprobamos si ksize es impar y si no lo es sumamos 1
  if ksizeX % 2 == 0:
    ksizeX += 1
  if ksizeY % 2 == 0:
    ksizeY += 1
  # pasamos a matriz de float
  im = im.astype(np.float64) 
  # calculamos el kernel de la gaussiana en el eje X
  kerX = cv2.getGaussianKernel(ksize=ksizeX, sigma=sigmaX)
  # transponemos la matriz en la dirección X
  kerX = np.transpose(kerX)
  # calculamos el kernel de la gaussiana en el eje Y
  kerY = cv2.getGaussianKernel(ksize=ksizeY, sigma=sigmaY)
  # No hago el flip por ser  simétricos ambos vectores respecto al centro
  # aplicamos las máscaras 1D en ambas direcciones
  # ddepth = -1 asigna la misma profundidad a la imagen nueva
  imblur = cv2.filter2D(im, ddepth=-1, kernel=kerX, borderType=border)
  imblur = cv2.filter2D(imblur, ddepth=-1, kernel=kerY, borderType=border)
  return imblur

def ejercicio1_A1(nombre, sigma1, sigma2, sigma3, 
                  ksize, borde1, borde2, borde3, borde4, flag=0):
  imagen = leeimagen("./imagenes/"+nombre+".bmp", flag)
  blur1 = funcionGaussiana(imagen, sigma1)
  blur2 = funcionGaussiana(imagen, sigma2)
  
  pintaAll([imagen, blur1, blur2], ["original","sigma="+str(sigma1),
           "sigma="+str(sigma2)])
  
  pintaI(funcionGaussiana(imagen, sigma3, 0, ksize, ksize, borde1), 
         "sigma="+str(sigma3)+" ksize=("+str(ksize)+
         ","+str(ksize)+") Borde CONSTANT")
  pintaI(funcionGaussiana(imagen, sigma3, 0, ksize, ksize, borde2), 
         "sigma="+str(sigma3)+" ksize=("+str(ksize)+
         ","+str(ksize)+") Borde REPLICATE")
  pintaI(funcionGaussiana(imagen, sigma3, 0, ksize, ksize, borde3), 
         "sigma="+str(sigma3)+" ksize=("+str(ksize)+
         ","+str(ksize)+") Borde REFLECT")
  pintaI(funcionGaussiana(imagen, sigma3, 0, ksize, ksize, borde4), 
         "sigma="+str(sigma3)+" ksize=("+str(ksize)+
         ","+str(ksize)+") Borde DEFAULT")
  # comparación con GaussianBlur
  im = leeimagen("./imagenes/bird.bmp", flag)
  gausblur = cv2.GaussianBlur(im, (0,0), sigma2)
  mygaus = funcionGaussiana(im, sigma2)
  pintaAll([mygaus, gausblur], ["Función propia vs GaussianBlur. Sigma="+str(sigma2)])


def convMasks1D(im, dx, dy, ksize, border=cv2.BORDER_DEFAULT):
  # im: imagen a visualizar
  # title: título de la imagen
  # dx: orden derivada de x
  # dy: orden derivada de y
  # ksize: valor de ksize (aperture size) (1,3,5 o 7)
  # border: especifica el tipo de borde
  # pasamos a matriz de float
  im = im.astype(np.float64) 
  # Calculamos los kernels de las derivadas 
  # Normalizamos la máscara con normalize=True
  kerX, kerY = cv2.getDerivKernels(dx=dx, dy=dy, ksize=ksize, normalize=True)
  # trasponemos el kernel en el eje X
  kerX = np.transpose(kerX)
  # hacemos flip del kernel para aplicar convolución el lugar de 
  # correlación
  kerX = np.flip(kerX)
  kerY = np.flip(kerY)
  # Aplicamos el filtro separable ker a la imagen
  # ddepth = -1 asigna la misma profundidad a la imagen nueva
  # aplicamos las máscaras 1D en ambas direcciones
  imconv = cv2.filter2D(im, ddepth=-1, kernel=kerX, 
                        borderType=border)
  imconv = cv2.filter2D(imconv, ddepth=-1, kernel=kerY, 
                        borderType=border)
  return imconv

def ejercicio1_A2(nombre, v1, v2, v3, v4, v5, flag=0):
  imagen = leeimagen("./imagenes/"+nombre+".bmp", flag)
  conv1 = convMasks1D(imagen, v1[0], v1[1], v1[2])
  conv2 = convMasks1D(imagen, v2[0], v2[1], v2[2])
  conv3 = convMasks1D(imagen, v3[0], v3[1], v3[2])
  
  pintaI(conv1, "dx="+str(v1[0])+", dy="+str(v1[1])+", ksize="+str(v1[2]))
  pintaI(conv2, "dx="+str(v2[0])+", dy="+str(v2[1])+", ksize="+str(v2[2]))
  pintaI(conv3, "dx="+str(v3[0])+", dy="+str(v3[1])+", ksize="+str(v3[2]))
  pintaAll([conv1, conv2, conv3], ["Máscaras de convolución"])

  conv4 = convMasks1D(imagen, v4[0], v4[1], v4[2])
  conv5 = convMasks1D(imagen, v5[0], v5[1], v5[2])
  pintaAll([conv4, conv5], ["ksize="+str(v4[2]), "ksize="+str(v5[2])])
  
# B. Usar la función Laplacian para el cálculo de la convolución 2D
# con una máscara normalizada de Laplaciana-de-Gaussiana de tamaño
# variable. Mostrar ejemplos de funcionamiento usando dos tipos de
# bordes y dos valores de sigma: 1 y 3.
 
def laplacianaGaussiana(im, sigma, ksize=1, border=cv2.BORDER_DEFAULT):
  # im: imagen a visualizar
  # sigma: valor especificado de sigma
  # ksize: tamaño del kernel de la Laplaciana, default 1 igual
  # que la función Laplacian de Open CV
  # border: especifica el tipo de borde
  # pasamos a matriz de float
  im = im.astype(np.float64) 
  # Llamamos a Gaussian lo que mejorará la sensibilidad al ruido
  # Aplicará como parámetro el tipo de borde especificado
  imblur = funcionGaussiana(im, sigmaX=sigma, border=border)
  kerX, kerY = cv2.getDerivKernels(dx=2, dy=2, ksize=ksize, normalize=True)
  # trasponemos el kernel en el eje X
  kerX = np.transpose(kerX)
  # hacemos flip del kernel para aplicar convolución el lugar de 
  # correlación
  kerX = np.flip(kerX)
  kerY = np.flip(kerY)
  # ddepth = -1 asigna la misma profundidad a la imagen nueva
  imX = cv2.filter2D(imblur, ddepth=-1, kernel=kerX, 
                     borderType=border)
  imY = cv2.filter2D(imblur, ddepth=-1, kernel=kerY, 
                     borderType=border)
  # devolvemos la laplaciana como suma de ambas imágenes
  return imX+imY

def ejercicio1_B(nombre, sigma1, sigma2, borde1, borde2, flag=0):
  imagen = leeimagen("./imagenes/"+nombre+".bmp", flag)
  lap1 = laplacianaGaussiana(imagen, sigma1)
  lap2 = laplacianaGaussiana(imagen, sigma2)  
  # Fijo ksize=21 para poder visualizar mejor los bordes
  lap3 = laplacianaGaussiana(imagen, sigma1, 21, border=borde1)
  lap4 = laplacianaGaussiana(imagen, sigma1, 21, border=borde2)
  
  pintaAll([lap1, lap2], ["Sin bordes", "sigma="+str(sigma1), "sigma="+str(sigma2)])
  pintaAll([lap3, lap4], ["Con bordes y sigma="+str(sigma1), "REPLICATE", "CONSTANT"])
  
# Ejercicio 2: Implementar funciones para las siguientes tareas.
    
# A. Representación en pirámide Gaussiana de 4 niveles de una imagen.
    
def reducirIm(im):
  # im: imagen a reducir
  filpar = []
  colpar = []
  # calculamos los vectores de índices impares para filas y columnas
  for i in range(im.shape[0]):
    if i % 2 == 1:
      filpar.append(i)
  for j in range(im.shape[1]):
    if j % 2 == 1:
      colpar.append(j)
  # eliminamos todas las filas y columnas impares de la matriz
  im = np.delete(im, filpar, axis=0)
  im = np.delete(im, colpar, axis=1)
  return im
    
def pirGaussiana(im, size, border=cv2.BORDER_DEFAULT):
  # im: imagen a obtener la pirámide gaussiana
  # size: número de niveles de la pirámide Gaussiana
  # border: especifica el tipo de borde
  # pasamos a matriz de float
  im = im.astype(np.float64) 
  # Añado al vector la imagen original
  pirgauss = [im]
  # Repetir el número de niveles de la pirámide
  for i in range(size):
    # Aplicamos un suavizado Gaussiano
    # Fijamos el sigma a 5 y ksize se calcula a partir de sigma
    imMin = funcionGaussiana(pirgauss[i], sigmaX=5, border=border)
    # Reducimos el tamaño de la imagen
    imMin = reducirIm(imMin)
    # Añadimos la imagen al vector y repetimos el proceso con ella
    pirgauss.append(imMin)
  return pirgauss

def ejercicio2_A(nombre, niveles, border=cv2.BORDER_DEFAULT, flag=0):
  imagen = leeimagen("./imagenes/"+nombre+".bmp", flag)
  pirG1 = pirGaussiana(imagen, niveles, border)
  pintaAll(pirG1, ["Piramide Gaussiana "+str(niveles)+" niveles"])

# B. Representación en pirámide Laplaciana de 4 niveles de una imagen.
  
def aumentarIm(im, orig):
  # im: imagen a aumentar
  # en las posiciones impares, replicamos las filas y columnas
  # de la imagen hasta igualar al tamaño original
  for i in range(orig.shape[0]):
    if i % 2 == 1:
      im = np.insert(im, (i), im[i-1,:], axis=0)
  for j in range(orig.shape[1]):
    if j % 2 == 1:
      im = np.insert(im, (j), im[:,j-1], axis=1)
  return im

def pirLaplaciana(im, size, border=cv2.BORDER_DEFAULT):
  # im: imagen a obtener la pirámide laplaciana
  # size: número de niveles de la pirámide Laplaciana
  # border: especifica el tipo de borde
  pirlap = []
  # construimos una pirámide gaussiana de tamaño size+1
  # no paso a float porque pirGausiana ya lo hace
  pirgaus = pirGaussiana(im, size+1, border)
  for i in range(size):
    # aumentamos el tamaño de la imagen de la pirámide i+1
    # adaptándola al tamaño de la imagen en la pos i
    imMax = aumentarIm(pirgaus[i+1], pirgaus[i])
    # multiplicamos por 4 para no reducir la intensidad 
    # de los píxeles en el suavizado Gaussiano
    imMax = funcionGaussiana(imMax, sigmaX=3, border=border)
    diff = pirgaus[i]-imMax
    # incluimos la diferencia en el vector pirlap
    pirlap.append(diff)
  return pirlap

def ejercicio2_B(nombre, niveles, border=cv2.BORDER_DEFAULT, flag=0):
  imagen = leeimagen("./imagenes/"+nombre+".bmp", flag)
  pirL1 = pirLaplaciana(imagen, niveles, border)
  pintaAll(pirL1, ["Piramide Laplaciana "+str(niveles)+" niveles"])

# C. Construir un espacio de escalas Laplaciano para implementar 
# la búsqueda de regiones con el algoritmo especificado.

def neighbors(im, i, j, flag=0):
  # im: imagen de origen
  # i,j: posición del pixel
  # flag: si es 1 incluye el pixel (i,j) en la lista
  # devuelve una lista con los valores de los vecinos
  # devuelve la lista vacía si es un valor del borde
  # fila o columna = 0 o im.shape
  vecinos = []
  indices = [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),(i,j+1),
  (i+1,j-1),(i+1,j),(i+1,j+1)]
  for k in range(len(indices)):
    if indices[k][0]>=0 and indices[k][1]>=0 and indices[k][0]<im.shape[0] and indices[k][1]<im.shape[1]:
      vecinos.append(im[indices[k][0]][indices[k][1]])
    if flag:
      vecinos.append(im[i][j])
  return vecinos

def supresionNoMax(im, pos, vim):
  # im: imagen a aplicar la supresión de máximos
  # pos: posición de im en el vector vim
  # vim: vector de todas las imágenes escaladas
  # hago una copia de la matriz original sobre la que haremos la supresión
  copia = np.copy(im)
  # para cada pixel de la imagen
  for i in range(im.shape[0]):
    for j in range(im.shape[1]):
      # si es un vector de tamaño 1 (1 escala)
      if len(vim)==1:
        maxsup, maxinf = 0.0 , 0.0
      # si tiene 2 o más escalas calculamos los vecinos máximos inf y sup
      else:
        # el primer elemento no tiene escala inferior
        if pos == 0:
          maxinf = 0.0
          maxsup = max(neighbors(vim[pos+1], i, j, 1))
        # el último elemento no tiene escala superior
        if pos == len(vim)-1:
          maxinf = max(neighbors(vim[pos-1], i, j, 1))
          maxsup = 0.0
        # si es un elemento intermedio
        else:
          maxsup = max(neighbors(vim[pos+1], i, j, 1))
          maxinf = max(neighbors(vim[pos-1], i, j, 1))
      # calculamos el máximo de los adyacentes
      maxlocal = max(neighbors(vim[pos], i, j))
      # máximo del cubo
      maxcubo = max(maxlocal, maxinf, maxsup)
      # si el elemento actual no ex máximo del cubo su valor=0.0
      if im[i][j] < maxcubo:
        copia[i][j] = 0.0
      
  return copia

def busqRegiones(im, n, umbral=15, sigma=1):
  # im: imagen a la que aplicar la búsqueda de regiones
  # n: número de escalas
  # umbral: valor mínimo del píxel para dibujar círculo default 15
  # sigma: valor de sigma fijado a 1
  # coeficiente de incremento de la escala
  cte = 1.2
  # constante proporcionalidad de la escala del radio de los círculos
  # la fijamos a 4 para visualizar mejor los círculos
  K = 4
  # pasamos a matriz de float
  im = im.astype(np.float64) 
  vim = []
  vescalas = []
  pixels = []
  # almacenamos todas las escalas en una lista
  for i in range(n):
    imlap = (sigma*sigma)*laplacianaGaussiana(im, sigma)
    # elevamos cada término de la matriz al cuadrado
    im2 = np.power(imlap, 2)
    # guardamos el resultado en una lista de escalas
    vescalas.append(im2)
    # multiplico por el coeficiente de incremento
    sigma = sigma*cte

  # para cada escala hacemos la supresión de no máximos
  # usando la comparación de cada pixel con los vecinos adyacentes
  # y los vecinos de la escala inferior y superior si la tuviesen
  for e in range(len(vescalas)):
    # aplicamos la supresión de no máximos
    imNoMax = supresionNoMax(vescalas[e], e, vescalas)
    # normalizamos la imagen obtenida
    imNorm = normalizaRGB(imNoMax)
    # almacenamos la posición de los píxeles con valor mayor 
    # a umbral en una lista
    for j in range(imNorm.shape[0]):
      for k in range(imNorm.shape[1]):
        if imNorm[j][k] > umbral:
          pixels.append((k,j))
    # pintamos los círculos en color blanco 255
    for p in pixels:
      cv2.circle(imNorm, p, int(sigma*K), 255)
      cv2.circle(im, p, int(sigma*K), 255)
    # almacenamos el resultado
    vim.append([imNorm,im])
    
  return vim

def ejercicio2_C(nombre, escalas, umbral=15, sigma=1):
  imagen = leeimagen("./imagenes/"+nombre+".bmp", 0)
  vim = busqRegiones(imagen, escalas, umbral, sigma)
  for i in range(len(vim)):
    pintaAll([vim[i][0],vim[i][1]], ["Búsqueda de regiones "+str(i+1)+" escala"])
  
# Ejercicio 3: Implementar una función que genere las imágenes de baja
# y alta frecuencia a partir de las parejas de imágenes para generar
# imágenes híbridas.
  
# 1. Escribir una función que muestre las tres imágenes (alta,baja e híbrida).

def imgsHibridas(im1, im2, sigma1, sigma2):
  # im1: imagen a la que aplicar el paso bajo
  # im2: imagen a la que aplicar el paso alto
  # sigma1: parámetro sigma para paso bajo
  # sigma2: parámetro sigma para paso alto
  # En la práctica 0 ya se tuvo en cuenta la normalización de las matrices.
  # pasamos a matriz de float
  im1 = im1.astype(np.float64) 
  im2 = im2.astype(np.float64) 
  # Aplico el filtro frecuencias bajas. Fijo ksize=(0,0)
  im1low = funcionGaussiana(im1, sigmaX=sigma1)
  # Aplico el filtro frecuencias bajas y se lo resto a im2
  # para quedarme con las frecuencias altas
  im2low = funcionGaussiana(im2, sigmaX=sigma2)
  im2high = im2-im2low
  # Monto la imagen híbrida
  hibrida = im2high+im1low
  # creo el vector con las tres imágenes
  vim = [im2high, im1low, hibrida]
  return vim

def ejercicio3_2(nombre1, nombre2, sigma1, sigma2):
  imagen1 = leeimagen("./imagenes/"+nombre1+".bmp", 0)
  imagen2 = leeimagen("./imagenes/"+nombre2+".bmp", 0)
  vim = imgsHibridas(imagen1, imagen2, sigma1, sigma2)
  pintaAll(vim, ["Alta", "Baja", "Híbrida"])
  
def ejercicio3_3(nombre1, nombre2, sigma1, sigma2, niveles):
  imagen1 = leeimagen("./imagenes/"+nombre1+".bmp", 0)
  imagen2 = leeimagen("./imagenes/"+nombre2+".bmp", 0)
  vim = imgsHibridas(imagen1, imagen2, sigma1, sigma2)
  pirG1 = pirGaussiana(vim[2], niveles)
  pintaAll(pirG1, ["Piramide Gaussiana "+str(nombre1)+"-"+str(nombre2)])

# Bonus 1: Implementar con código propio la convolución 2D con 
# cualquier máscara 2D de números reales usando máscaras separables.
    
                ####################################
  
# Bonus 2: Realizar todas las parejas de imágenes híbridas 
# en su formato a color
    
def imgsHibridasColor(im1, im2, sigma1, sigma2):
    # im1: imagen a la que aplicar el paso bajo
    # im2: imagen a la que aplicar el paso alto
    # sigma1: parámetro sigma para paso bajo
    # sigma2: parámetro sigma para paso alto
    # si alguna de las imágenes está en formato blanco
    # y negro, lo pasamos a color
    if len(im1.shape) == 2:
      im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    if len(im2.shape) == 2:
      im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
    # Aplico el filtro frecuencias bajas. Fijo ksize=(0,0)
    im1 = im1.astype(np.float64) 
    im2 = im2.astype(np.float64) 
    im1low = funcionGaussiana(im1, sigmaX=sigma1)
    # Aplico el filtro frecuencias bajas y se lo resto a im2
    # para quedarme con las frecuencias altas
    im2low = funcionGaussiana(im2, sigmaX=sigma2)
    im2high = im2-im2low
    # Monto la imagen híbrida
    hibrida = im2high+im1low
    # creo el vector con las tres imágenes
    vim = [im2high, im1low, hibrida]
    return vim

def bonus2(nombre1, nombre2, sigma1, sigma2):
  imagen1 = leeimagen("./imagenes/"+nombre1+".bmp", 1)
  imagen2 = leeimagen("./imagenes/"+nombre2+".bmp", 1)
  vim = imgsHibridasColor(imagen1, imagen2, sigma1, sigma2)
  pintaAll(vim, ["Alta", "Baja", "Híbrida"])
  pirG1 = pirGaussiana(vim[2], niveles)
  pintaAll(pirG1, ["Piramide Gaussiana Color "+str(nombre1)+"-"+str(nombre2)])
 
# Imágenes obtenidas de otras más grandes e hibridadas
# con las funciones programadas en el ejercicio 3
  
def bonus3(sigma1, sigma2, flag=0):
  imagen1 = leeimagen("./imagenes/watermelon.bmp", flag)
  imagen2 = leeimagen("./imagenes/ball.bmp", flag)
  if flag==0:
    vim = imgsHibridas(imagen1, imagen2, sigma1, sigma2)
  else:
    vim = imgsHibridasColor(imagen1, imagen2, sigma1, sigma2)
  pintaAll(pirGaussiana(vim[2], 4), ["Pirámide Gaussiana Balón-Sandía"])
  pintaAll(vim, ["Alta", "Baja", "Híbrida"])

if __name__ == '__main__':  
   
  print("EJERCICIO 1: \n")
  
  print("Apartado A. Cálculo convolución máscaras 1D y Gaussiana.")
  
  # Calculamos la Gaussiana con sigma 2 y sigma 10 de la imagen leída.
  nombre_imagen = "dog"
  # valores comparación de imágenes
  sigma1 = 2
  sigma2 = 10
  # valores comparación de contornos
  sigma3 = 10
  ksize = 21
  # Tipos de bordes usados (REPLICATE, CONSTANT, REFLECT y DEFAULT)
  borde1 = cv2.BORDER_CONSTANT
  borde2 = cv2.BORDER_REPLICATE
  borde3 = cv2.BORDER_REFLECT
  borde4 = cv2.BORDER_DEFAULT
  ejercicio1_A1(nombre_imagen, sigma1, sigma2, 
                sigma3, ksize, borde1, borde2, borde3, borde4)

  # Probamos diferentes ejemplos de convoluciones fijando el tamaño
  # del kernel a 5 y variando el orden de las derivadas en cada
  # dirección.
  nombre_imagen = "bird"
  # valores comparación de imágenes (dx, dy, ksize) variando dx dy
  # Fijamos ksize=5
  v1 = [1,1,5]
  v2 = [3,0,5]
  v3 = [3,3,5]
  # valores comparación de imágenes (dx, dy, ksize) variando ksize  
  # Fijamos el valor de dx=dy=1
  v4 = [1,1,3]
  v5 = [1,1,9]
  
  ejercicio1_A2(nombre_imagen, v1, v2, v3, v4, v5)

  input("Pulse la tecla ENTER para continuar")

  print("Apartado B. Cálculo convolución Laplaciana-Gaussiana.")
  
  # Calculamos la Laplaciana-Gaussiana con sigma 1 y 3 de la imagen 
  # original y vemos como afectan los bordes a la función fijando
  # sigma=1.
  sigma1 = 1
  sigma2 = 3
  # Tipos de bordes utilizados (REPLICATE y CONSTANT)
  borde1 = cv2.BORDER_REPLICATE
  borde2 = cv2.BORDER_CONSTANT
  ejercicio1_B(nombre_imagen, sigma1, sigma2, borde1, borde2)
  
  input("Pulse la tecla ENTER para continuar")

  print("\nEJERCICIO 2: \n")
  
  print("Apartado A. Pirámide Gaussiana.")
  
  # Dibujamos la pirámide Gaussiana de 4 niveles
  nombre_imagen = "fish"
  niveles = 4
  borde = cv2.BORDER_CONSTANT
  ejercicio2_A(nombre_imagen, niveles)
  ejercicio2_A(nombre_imagen, niveles, borde)
  
  input("Pulse la tecla ENTER para continuar")

  print("Apartado B. Pirámide Laplaciana.")
  
  # Dibujamos la pirámide Laplaciana de 4 niveles
  nombre_imagen = "fish"
  niveles = 4
  borde = cv2.BORDER_CONSTANT
  ejercicio2_B(nombre_imagen, niveles)
  ejercicio2_B(nombre_imagen, niveles, borde)

  input("Pulse la tecla ENTER para continuar")
  
  print("Apartado C. Espacio de búsquedas Laplaciano.")
  
  nombre_imagen = "einstein"
  escalas = 2
  umbral = 50
  # el sigma está fijado a 1 
  ejercicio2_C(nombre_imagen, escalas, umbral)
  
  nombre_imagen = "bird"
  # comparación diferencia de umbrales
  escalas = 1
  umbral = 50
  ejercicio2_C(nombre_imagen, escalas, umbral)
  umbral = 25
  ejercicio2_C(nombre_imagen, escalas, umbral)
  
  input("Pulse la tecla ENTER para continuar")
  
  print("\nEJERCICIO 3. \n")
  
  print("Apartado 1. Imágenes híbridas.")
  
  # Código programado y ejecución del primer ejemplo del Apartado 2.
  nombre_imagen1 = "dog"
  nombre_imagen2 = "cat"
  sigma1 = 8
  sigma2 = 5
  ejercicio3_2(nombre_imagen1, nombre_imagen2, sigma1, sigma2)
  
  # Realizar composición con al menos 3 parejas de imágenes.
  print("Apartado 2. Parejas de imágenes híbridas.")
  
  nombre_imagen1 = "einstein"
  nombre_imagen2 = "marilyn"
  sigma1 = 5
  sigma2 = 3
  ejercicio3_2(nombre_imagen1, nombre_imagen2, sigma1, sigma2)
  
  nombre_imagen1 = "motorcycle"
  nombre_imagen2 = "bicycle"
  sigma1 = 9
  sigma2 = 3
  ejercicio3_2(nombre_imagen1, nombre_imagen2, sigma1, sigma2)
  
  input("Pulse la tecla ENTER para continuar")

  # Crear pirámide Gaussiana e 4 niveles con las imgs. híbridas. 
  print("Apartado 3. Pirámide Gaussiana de imágenes híbridas.")
  
  # Dibujamos una pirámide Gaussiana las tres imágenes híbridas
  niveles = 4
  nombre_imagen1 = "dog"
  nombre_imagen2 = "cat"
  sigma1 = 8
  sigma2 = 5
  ejercicio3_3(nombre_imagen1, nombre_imagen2, sigma1, sigma2, niveles)
  
  nombre_imagen1 = "einstein"
  nombre_imagen2 = "marilyn"
  sigma1 = 5
  sigma2 = 3
  ejercicio3_3(nombre_imagen1, nombre_imagen2, sigma1, sigma2, niveles)
  
  nombre_imagen1 = "motorcycle"
  nombre_imagen2 = "bicycle"
  sigma1 = 9
  sigma2 = 3   
  ejercicio3_3(nombre_imagen1, nombre_imagen2, sigma1, sigma2, niveles)
  
  input("Pulse la tecla ENTER para continuar")

  # BONUS
  
  print("\nBONUS 2: Imágenes híbridas a color.\n")
  
  # Las funciones para pintar ya están diseñadas para que dibujen en 
  # color, por lo que simplemente adaptamos el código para que en el
  # caso de que añadamos una imagen en grises y otra a color, pase
  # ambas a color y proceda a generar su híbrida.
  
  nombre_imagen1 = "dog"
  nombre_imagen2 = "cat"
  sigma1 = 8
  sigma2 = 5
  bonus2(nombre_imagen1, nombre_imagen2, sigma1, sigma2)
  
  nombre_imagen1 = "einstein"
  nombre_imagen2 = "marilyn"
  sigma1 = 5
  sigma2 = 3
  bonus2(nombre_imagen1, nombre_imagen2, sigma1, sigma2)
  
  nombre_imagen1 = "motorcycle"
  nombre_imagen2 = "bicycle"
  sigma1 = 9
  sigma2 = 3
  bonus2(nombre_imagen1, nombre_imagen2, sigma1, sigma2)

  input("Pulse la tecla ENTER para continuar") 

  print("\nBONUS 3: Imagen híbrida personalizada.\n")
  
  # Las imágenes usadas se llaman directamente en el código del bonus3
  
  sigma1 = 10
  sigma2 = 5  
  bonus3(sigma1, sigma2)