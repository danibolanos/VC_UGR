"""
Asignatura : Visión por Computador
Proyecto Final : Implementación Algoritmo Seam-Carving
# Curso 2019/2020
# Alumno : Daniel Bolaños Martínez y José María Borrás Serrano
"""
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from numba import jit
import warnings
import time
import random

# Lee una imagen con el flag del color especificado
def leeimagen(filename, flagColor):
  # filename: nombre de la imagen
  # flagColor: 1 para color y 0 para blanco y negro
  if flagColor:
    im = cv2.imread(filename)
  else:
    im = cv2.imread(filename, 0)
  
  im=im.astype(np.float32)
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
  #cv2.imwrite(title+'.jpg', im)
  return imgrgb

def convMasks1D(im, dx, dy, ksize, border=cv2.BORDER_DEFAULT):
  # im: imagen a visualizar
  # title: título de la imagen
  # dx: orden derivada de x
  # dy: orden derivada de y
  # ksize: valor de ksize (aperture size) (1,3,5 o 7)
  # border: especifica el tipo de borde
  # pasamos a matriz de float
  im = im.astype(np.float32) 
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

#Calcula el mapa de energía de una imagen
def mapa_energia(im, ksize=5):
    # im: imagen de la que obtener el mapa de energía
    # ksize: tamaño del kernel a utilizar en el cálculo de la derivada de la imagen
    
    #pasamos a matriz de float32
    im = im.astype(np.float32) 
    #trabajamos con una copia
    imGris = np.copy(im)
    #Si es una imagen a color, la convertimos a blanco y negro, facilitando el cálculo de la energía
    if len(im.shape)==3: #Para convertir la imagen de color a blanco y negro, facilitando el cálculo de la energía
            imGris=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #el mapa de energia es abs(dI/dx)+abs(dI/dy)
    mapa_energia = abs(convMasks1D(imGris, 1, 0, ksize)) + abs(convMasks1D(imGris, 0, 1, ksize))
    
    return mapa_energia

#Eliminamos un número determinado de columnas de la imagen utilizando Seam Carving
def podar_columnas(im, n):
    # im: imagen a utilizar
    # n: número de columnas a podar
    
    for i in range(n):
        # le quitamos una columna a la imagen
        im = cortar_columna(im)

    return im

# Eliminamos un número determinado de filas de la imagen utilizando Seam Carving
def podar_filas(im, n):
    # im: imagen a utilizar
    # n: número de filas a podar
    
    #rotamos la imagen 90º grados, utilizamos podar_columnas y rotamos la imagen
    # 90º en sentido contrario
    im = np.rot90(im, 1, (0, 1))
    im = podar_columnas(im, n)
    im = np.rot90(im, 3, (0, 1))
    return im

# Eliminamos una columna de la imagen utilizando Seam Carving
def cortar_columna(im): #Va más lento con @jit, así que no lo ponemos
    # im: imagen a utilizar
    
    fil, col = im.shape[0], im.shape[1]
    # Hacemos una imagen (vacía) con el mismo número de filas y una columna menos
    # que la imagen que hemos pasado como argumento
    nueva_im = np.empty((fil,col-1,3))

    # Obtenemos la matriz que contiene la mínima energía acumulada por cada hilo
    M = seam_vertical(im)

    # El mínimo valor de la última fila indica el hilo vertical de mínima energía
    # Hacemos backtracking para encontrar el camino de dicho hilo y 
    # en la imagen nueva guardamos la imagen original sin los pixeles del hilo
    j = np.argmin(M[fil-1, 0:col])
    nueva_im[fil-1] = np.delete(im[fil-1], j, axis=0)
    for i in range(2,fil+1):
        if (j==0):
            j+=np.argmin(M[fil-i, 0:2])
        else:
            j=j-1+np.argmin(M[fil-i, j-1:j+2])
        nueva_im[fil-i] = np.delete(im[fil-i], j, axis=0)
        
    return nueva_im

# Calculamos la matriz que contiene la mínima energía acumulada por cada hilo
@jit
def seam_vertical(im):
    # im: imagen de la que obtener la energía
    fil, col = im.shape[0], im.shape[1]
    
    #Obtenemos el mapa de energía de la imagen [M(i,j)=e(i,j)]
    M = mapa_energia(im)
    
    # M(i,j) = e(i,j) + min(M(i-1,j-1),M(i-1,j),M(i-1,j+1))
    for i in range(1, fil):
       #Cuando j=0
       M[i,0] += np.min(M[i-1, 0:2])
       #Cuando j={1,col-1}
       for j in range(1, col):
           M[i,j] += np.min(M[i-1, j-1:j+2])

    return M

#Primera optimización:
# En lugar de recorrer toda la imagen para calcular los caminos de energía de cada hilo,
# sólo calculamos los hilos en el intervalo de columnas [col_min, col_max] .
# 1. Cada intervalo depende de las columnas por las que pasa el hilo anterior.
# 2. Cada cierto número de hilos recortados recorremos toda la imagen.


def podar_columnas_O1(im, n):
    # im: imagen a utilizar
    # n: número de columnas a podar
    
    for i in range(n):
        # cada diez ciclos, calculamos todos los hilos, es decir,
        # col_min=0, col_max=im.shape[1]
        if(i%10 == 0):
            col_min=0
            col_max=im.shape[1]
        # los siguientes col_min y col_max a utilizar dependen de las columnas
        # por las que haya pasado el hilo que se ha eliminado
        im,  col_min, col_max = cortar_columna_O1(im, col_min, col_max)

    return im

# Primera optimización
# Eliminamos un número determinado de filas de la imagen utilizando Seam Carving
def podar_filas_O1(im, n):
    # im: imagen a utilizar
    # n: número de filas a podar
    
    #rotamos la imagen 90º grados, utilizamos podar_columnas y rotamos la imagen
    # 90º en sentido contrario
    im = np.rot90(im, 1, (0, 1))
    im = podar_columnas_O1(im, n)
    im = np.rot90(im, 3, (0, 1))
    return im

# Primera optimización
# Eliminamos una columna de la imagen utilizando Seam Carving
# Devuelve la imagen quitando el hilo vertical y dos valores:
# - mini: el índice de la columna más a la izquierda por la que pasa el hilo - valor de intervalo
# - maxi: el índice de la columna más a la derecha por la que pasa el hilo + valor de intervalo
def cortar_columna_O1(im, col_min, col_max):
    # im: imagen a utilizar
    # col_min: principio del intervalo en que calcular los hilos
    # col_max: fin del intervalo en que calcular los hilos
    
    # el valor del intervalo que se va a restar/sumar a mini/maxi respectivamente
    intervalo = 50
    
    fil, col = im.shape[0], im.shape[1]
    nueva_im = np.zeros((fil,col-1,3))

    # Obtenemos la matriz que contiene la mínima energía acumulada por cada hilo
    # del intervalo [col_min, col_max]
    # La función seam_vertical que utilizamos es la misma, pero le pasamos 
    # la imagen sólo con las columnas entre [col_min, col_max]
    M = seam_vertical(im[0:fil, col_min:col_max]) 
    col_M = M.shape[1]

    # Hacemos backtracking desde la ultima fila hasta la primera para 
    # encontrar el camino del hilo de mínima energía
    j = np.argmin(M[fil-1, 0:col_M])
    nueva_im[fil-1] = np.delete(im[fil-1], col_min + j, axis=0)
    # mini tendrá el índice de la columna más a la izquierda por la que pasa el hilo
    # maxi tendrá el índice de la columna más a la derecha por la que pasa el hilo
    mini = j
    maxi = j
    #Para las filas desde la penultima hasta la segunda
    for i in range(2,fil+1):
        if (j==0):
            j=np.argmin(M[fil-i, 0:2])
        else:
            j=j-1+np.argmin(M[fil-i, j-1:j+2])  
        mini = min(j, mini)
        maxi = max(j, maxi)
        nueva_im[fil-i] = np.delete(im[fil-i], col_min + j, axis=0)
    
    # mini es el índice de la columna más a la izquierda por la que pasa el hilo 
    # de mínima energía menos intervalo
    # maxi es el índice de la columna más a la derecha por la que pasa el hilo 
    # de mínima energía más intervalo
    # También nos aseguramos de que no se salga de la nueva imagen
    mini = max(col_min + mini - intervalo, 0)
    maxi = min(col_min + maxi + intervalo, col-1)
        
    return nueva_im, mini, maxi

#Segunda optimización

# En lugar de recorrer toda la imagen para cada seam, recorremos solo una parte [col_min, col_max] 
# Así sólo tenemos que hacer todos los hilos la primera vez
# Adicionalmente, en lugar de usar np.delete, utilizamos im[matriz_booleanos].reshape
# Además, hacemos una matriz auxiliar para backtracking de forma que cada elemento
# contiene la posición anterior


def podar_columnas_O2(im, n):
    # im: imagen a utilizar
    # n: número de columnas a podar
    
    # La primera vez hacemos calulamos todos los hilos, luego col_min=0 y col_max=im.shape[1]
    col_min=0
    col_max=im.shape[1]
    for i in range(n):
        # los siguientes col_min y col_max a utilizar dependen de las columnas
        # por las que haya pasado el hilo que se ha eliminado
        im,  col_min, col_max = cortar_columna_O2(im, col_min, col_max)

    return im

# Primera optimización
# Eliminamos un número determinado de filas de la imagen utilizando Seam Carving
def podar_filas_O2(im, n):
    # im: imagen a utilizar
    # n: número de filas a podar
    
    #rotamos la imagen 90º grados, utilizamos podar_columnas y rotamos la imagen
    # 90º en sentido contrario
    im = np.rot90(im, 1, (0, 1))
    im = podar_columnas_O2(im, n)
    im = np.rot90(im, 3, (0, 1))
    return im

# Segunda optimización (La que utiliceramos realmente)
# Eliminamos una columna de la imagen utilizando Seam Carving
# Devuelve la imagen quitando el hilo vertical y dos valores:
# - mini: el índice de la columna más a la izquierda por la que pasa el hilo - valor de intervalo
# - maxi: el índice de la columna más a la derecha por la que pasa el hilo + valor de intervalo
def cortar_columna_O2(im, col_min, col_max):
    # im: imagen a utilizar
    # col_min: principio del intervalo en que calcular los hilos
    # col_max: fin del intervalo en que calcular los hilos
    
    # el valor del intervalo que se va a restar/sumar a mini/maxi respectivamente
    intervalo = 50 
    
    fil, col = im.shape[0], im.shape[1]
    # matriz de booleanos que tendrá True si el pixel se mantiene y False si se elimina
    matriz_bool = np.ones((fil,col,3), dtype=np.bool)

    # Obtenemos la matriz que contiene la mínima energía acumulada por cada hilo
    # del intervalo [col_min, col_max] y la matriz que contiene el backtracking.
    # A la función seam_vertical_O2 que utilizamos le pasamos 
    # la imagen sólo con las columnas entre [col_min, col_max]
    M, backtrack = seam_vertical_O2(im[0:fil, col_min:col_max])

    #Hacemos backtracking desde la ultima fila hasta la primera
    j = np.argmin(M[fil-1, 0:M.shape[1]]) #Estamos en la ultima fila
    mini = j
    maxi = j
    for i in range(1,fil): #Para las filas desde la penultima hasta la segunda
        matriz_bool[fil-i,col_min+j] = (False,False,False) 
        j = backtrack[fil-i, j]
        mini = min(mini, j)
        maxi = max(maxi, j)
    matriz_bool[0,col_min+j] = (False,False,False) #Estamos en la primera fila  
    
    # mini es el índice de la columna más a la izquierda por la que pasa el hilo 
    # de mínima energía menos intervalo
    # maxi es el índice de la columna más a la derecha por la que pasa el hilo 
    # de mínima energía más intervalo
    # También nos aseguramos de que no se salga de la nueva imagen
    mini = max(col_min + mini - intervalo, 0)
    maxi = min(col_min + maxi + intervalo, col - 1)
        
    return im[matriz_bool].reshape((fil,col-1,3)), mini, maxi

# Segunda optimización
# Calculamos la matriz que contiene la mínima energía acumulada para cada hilo
# que esté contenido entre las columnas [col_min, col_max]
# Devolvemos dicha matriz y otra que contiene el backtracking 
# (cada elemento tiene el índice de la columna del elemento anterior del hilo)
@jit
def seam_vertical_O2(im, mapa_e = np.empty((0,0))):
    # im: imagen a utilizar
    # mapa_e: matriz que contiene la energía de cada elemento de la imagen.
    #        Por defecto, es una matriz vacía, así indicamos que no se le ha pasado 
    #       el mapa de energía y por tanto hay que calcularlo.
    
    fil, col = im.shape[0], im.shape[1]
    # matriz para el backtracking
    backtrack = np.empty((fil,col), dtype=np.int)

    # Si no se le pasa el mapa de energía, entonces lo calculamos
    if len(mapa_e)==0:
        M = mapa_energia(im)
    # Si se le pasa el mapa de energía, entonces utilizamos ese
    else:
        M = np.copy(mapa_e)

    # Calculamos M(i,j) = e(i,j) + min(M(i-1,j-1),M(i-1,j),M(i-1,j+1))
    # pero para j perteneciente a [col_min, col_max]
    for i in range(1, fil):
       #Cuando j=0
       indice = np.argmin(M[i-1, 0:2])
       backtrack[i, 0] = indice
       M[i, 0] += M[i-1, indice]
       #Cuando j={0, col-1}
       for j in range(1, col):
           indice = np.argmin(M[i-1, j-1:j+2]) + j - 1
           backtrack[i,j] = indice 
           M[i,j] += M[i-1, indice]

    return M, backtrack

# Añadimos un número determinado de columnas a la imagen utilizando Seam Carving
def aumentar_columnas(im, n, k=3):
    # im: imagen a utilizar
    # n: número de columnas a aumentar
    # k: dividido entre el número de columnas obtenemos el máximo número
    # de costuras que se repiten para aumentar la imagen
    
    col = im.shape[1]
    # número de columnas que tendrá la imagen después de aniadirle las columnas nuevas
    nueva_col = col+n

    # mientras no tengamos el número de columnas que queremos repetiremos el proceso
    # de aniadir un número determinado de columnas
    while(col<nueva_col):
        #columnas nuevas que vamos a aniadir a la vez
        # es el mínimo entre las columnas que quedan por aniadir y un tercio de las columnas
        # que ya tiene la imagen, lo hacemos así para intentar no tomar hilos con energía alta
        col_nuevas = min(col//k,nueva_col-col)
        im = aniadir_columnas(im, col_nuevas)
        col = im.shape[1]

    return im

# Añadimos un número determinado de filas a la imagen utilizando Seam Carving
def aumentar_filas(im, n, k=3):
    # im: imagen a utilizar
    # n: número de filas a aumentar
    # k: dividido entre el número de filas obtenemos el máximo número
    # de costuras que se repiten para aumentar la imagen
    
    #rotamos la imagen y utilizamos aumentar_columnas
    im = np.rot90(im, 1, (0, 1))
    im = aumentar_columnas(im,n,k)
    im = np.rot90(im, 3, (0, 1))
    return im

# Intentamos duplicar un número de hilos determinados, como los hilos no pueden intersecarse
# es posible que no podamos duplicar el número seleccionado. En ese caso, añadimos todos los
# que podemos sin que se intersequen.
@jit
def aniadir_columnas(im,n):
    # im: imagen a utilizar
    # n: número de hilos a duplicar
    
    fil, col = im.shape[0], im.shape[1]
    #nueva_im = np.zeros((fil,col+n,3))
    
    #obtenemos el mapa de energía
    energy_map = mapa_energia(im)

    #guardamos en una matriz los pixeles que forman parte de la costura
    # 1 si forma parte, 0 si no forma parte
    # (ver si la matriz se puede hacer de booleanos)
    mapa_seam = np.zeros((fil,col),dtype=np.uint8)

    #para cada columna a aniadir
    terminar=False
    k=0
    while (k<n and not(terminar)):
        # Copiamos el mapa de energía y hacemos:
        # M(i,j) = e(i,j) + min(M(i-1,j-1),M(i-1,j),M(i-1,j+1))
        M = energy_map.copy()
        for i in range(1, fil):
            #Cuando j=0
            M[i,0] += np.min(M[i-1, 0:2])
            #Cuando j={1,col-1}
            for j in range(1, col):
                    M[i,j] += np.min(M[i-1, j-1:j+2])
            
        # hacemos backtracking para obtener el hilo.
        # Cada elemento del hilo lo ponemos en el mapa de energía con valor infinito
        # para así no repetirlo y que los siguientes hilos sean diferentes.
        # Al mismo tiempo marcamos en una matriz de booleanos los elementos que
        # pertenecen al hilo, para luego duplicarlos
        j = np.argmin(M[fil-1, 0:col])
        # Si la última fila de M sólo tiene el valor infinito, entonces no podemos
        # tomar más hilos sin que se repitan elementos. Así que terminamos de 
        # obtener hilos distintos
        if(M[fil-1][j]==math.inf):
            terminar=True
        else:
            mapa_seam[fil-1][j] = 1
            energy_map[fil-1][j]=math.inf
            for i in range(2,fil+1):
                if (j==0):
                    j+=np.argmin(M[fil-i, 0:2])
                else:
                    j+=np.argmin(M[fil-i, j-1:j+2])-1 
                energy_map[fil-i][j]=math.inf
                mapa_seam[fil-i][j]=1
            k+=1
    
    # nueva imagen con las columnas añadidas
    nueva_im = np.zeros((fil,col+k,3))
    
    # En la nueva imagen vamos poniendo los valores de la imagen original y duplicando los
    # que pertenecen a algún hilo
    for i in range(fil):
        cont=0
        for j in range(col):
            nueva_im[i,j+cont]=im[i,j]
            if(mapa_seam[i,j]==1):
                cont+=1
                nueva_im[i,j+cont]=im[i,j]
                
    return nueva_im
    
# Optimización
# En lugar de calcular todos los hilos cada vez que vayamos a obtener uno nuevo, calculamos
# los hilos en un determinado intervalo
# Además sólo hacemos el cálculo del número de hilos una vez, luego nos encargamos
# de repetir cuidadosamente los elementos de dichos hilos hasta tener el número de columnas deseado
    
# Aumenta la imagen un número de columnas determinado
def aumentar_columnas_O(im, n, k=3):
    # im: imagen a utilizar
    # n: número de columnas a aumentar
    # k: dividido entre el número de columnas obtenemos el máximo número
    # de costuras que se repiten para aumentar la imagen
    
    if n == 0:
        return im
    
    fil, col = im.shape[0], im.shape[1]
    
    # obtenemos una matriz booleana donde los elementos true son los elementos a repetir
    # y num_seam es el número de hilos distintos que hemos podido obtener 
    mapa_seam, num_seam = aniadir_columnas_O(im, min(col//k, n))
    
    #creamos una nueva imagen con las dimensiones deseadas
    nueva_im = np.zeros((fil,col+n,3))
    
    
    # repetir es el número de veces que aniadiremos cada hilo
    repetir = n//num_seam
    # extra es el número de hilos que debemos aniadir una vez adicional para completar
    # el número de columnas
    extra = int(num_seam * (n/num_seam - (n // num_seam) ) )
    
    # En la nueva imagen vamos poniendo los valores de la imagen original y repitiendo los
    # que pertenecen a algún hilo el número de veces necesario
    for i in range(fil):
        cont_seam=0 # llevamos la cuenta de cuantos hilos hemos añadido
        cont_extra=0 # llevamos la cuenta de cuantos hilos hemos añadido una vez extra
        for j in range(col):
            nueva_im[i,j+cont_seam]=im[i,j]
            if (mapa_seam[i,j]==True):
                for k in range(repetir):
                    cont_seam+=1
                    nueva_im[i,j+cont_seam]=im[i,j]
                if (cont_extra < extra):
                    cont_extra+=1
                    cont_seam+=1
                    nueva_im[i,j+cont_seam]=im[i,j]
              
    return nueva_im

#Optimización

# Aumenta la imagen un número de filas determinado    
def aumentar_filas_O(im, n, k=3):
    # im: imagen a utilizar
    # n: número de columnas a aumentar
    # k: dividido entre el número de filas obtenemos el máximo número
    # de costuras que se repiten para aumentar la imagen
    
    # rotamos la imagen y llamamos a aumentar_columnas_O
    im = np.rot90(im, 1, (0, 1))
    im = aumentar_columnas_O(im, n, k)
    im = np.rot90(im, 3, (0, 1))
    return im
    
# Optimización 
    
# Intentamos obtener todos los hilos de mínima energía que podamos sin que se repitan elementos 
# hasta un número determinado. Devuelve una matriz de booleanos donde el elemento es True si pertenece a
# a algún hilo y también devuelve el número de hilos distintos que hemos conseguido.
def aniadir_columnas_O(im,n):
    # es llamada por aumentar_columnas
    # n: número de columnas a aniadir
    fil, col = im.shape[0], im.shape[1]
    
    # número de hilos distintos que ya tenemos
    num_seam=0
    
    #Obtenemos el mapa de energía
    mapa_e = mapa_energia(im)

    #guardamos en una matriz los pixeles que forman parte de la costura
    # 1 si forma parte, 0 si no forma parte
    # (ver si la matriz se puede hacer de booleanos)
    mapa_seam = np.zeros((fil,col),dtype=np.bool)

    # terminar es True si ya no podemos obtener más hilos de mínima energía sin que intersequen
    terminar=False
    
    # el valor del intervalo que se va a restar/sumar a mini/maxi respectivamente
    intervalo = 100
    
    # El intervalo en que calculamos los hilos es [col_min, col_max]
    # La primera vez col_min=0, col_max=col, así calculamos todos los hilos
    col_min = 0
    col_max = col
    
    # Mientras que el número de hilos distintos sea menor que el número que queremos obtener
    # y sigamos pudiendo obtener más hilos sin que intersequen
    while (num_seam < n and not(terminar)):
        
        # Obtenemos la matriz que contiene la mínima energía acumulada por cada hilo
        # del intervalo [col_min, col_max] y la matriz que contiene el backtracking.
        M, backtrack = seam_vertical_O2(im[0:fil, col_min:col_max], mapa_e[0:fil, col_min:col_max])
        
        # Hacemos backtracking para obtener los elementos del hilo de mínima energía
        j = np.argmin(M[fil-1, 0:M.shape[1]])
        
        # Si cada valor de la última fila de M es infinito es porque en el intervalo
        # [col_min, col_max] ya no podemos obtener más hilos que no intersequen
        # Así si el intervalo ha sido toda la imagen, terminamos. Si el intervalo sólo ha sido parte
        # de la imagen intentamos entcontrar un nuevo hilo distinto en toda la imagen
        if(M[fil-1, j] == math.inf):
            if(col_min == 0 and col_max == col):
                terminar=True
            else:
                col_min = 0
                col_max = col
        else:
            mini = j
            maxi = j
            for i in range(1,fil):
                mapa_seam[fil-i, col_min + j] = True
                #aniadir[fil-i].append(col_min+j)
                mapa_e[fil-i, col_min + j]=math.inf
                j = backtrack[fil-i, j]
                mini = min(mini, j)
                maxi = max(maxi, j)
                
            mapa_seam[0, col_min + j] = True
            #aniadir[0].append(col_min+j)
            mapa_e[0, col_min + j]=math.inf
            col_min = max(col_min+mini-intervalo, 0)
            col_max = min(col_min+maxi+intervalo, col)
            num_seam += 1
                
    return mapa_seam, num_seam

# Hacemos resize para dejar la imagen con el número de filas y columnas indicado.
# Para ello utlizaremos el algoritmo de Seam Carving que hemos implementado
def resize_seam(im,n_fil,n_col, k=3):
    # im: imagen a utilizar
    # n_fil: número de filas que queremos que tenga la imagen
    # n_col: número de columnas que queremos que tenga la imagen
    # k: dividido entre el número de columnas obtenemos el máximo número
    # de costuras que se repiten para aumentar la imagen
    fil, col = im.shape[0], im.shape[1]
    im_seam=np.copy(im)
    
    # Teniendo en cuenta de que el número de filas/columnas sea menor/mayor
    # tendremsos que podar/aumentar filas/columnas
    if (col > n_col):
        im_seam=podar_columnas_O2(im_seam, col-n_col)
    else:
        im_seam=aumentar_columnas_O(im_seam, n_col-col, k)
    if (fil > n_fil):
        im_seam=podar_filas_O2(im_seam, fil-n_fil)
    else:
        im_seam=aumentar_filas_O(im_seam, n_fil-fil, k)
        
    return im_seam

# Optimización
# A ser posible, haremos primero el mayor resize de cv2 que nos acerque al número de
# filas/columnas que queremos y sin modificar el ratio de aspecto
#   Ej: queremos triplicar las filas y duplicar las columnas, entonces primero duplicamos
#   las filas y columnas con cv2.resize y luego añadimos las filas que faltan utilizando
#   el algoritmo de Seam Carving
    
# Hacemos resize para dejar la imagen con el número de filas y columnas indicado.
# Para ello utlizaremos el algoritmo de Seam Carving que hemos implementado
def resize_seam_O(im,n_fil, n_col, k=3):
    #Si se puede, utilizamos primero cv2.resize para modificar el tamaño de la imagen 
    # sin cambiar la proporción. Luego utilizamos seam_Carving para las filas/columnas que queden.
    fil, col = im.shape[0], im.shape[1]
    im_seam=np.copy(im)
    
    if(n_fil>fil and n_col>col):
        escala_min = min(n_fil/fil, n_col/col)
        im_seam = cv2.resize(im_seam, (int(col*escala_min), int(fil*escala_min)))
        im_seam=aumentar_columnas_O(im_seam, n_col - int(col*escala_min), k)
        im_seam=aumentar_filas_O(im_seam, n_fil - int(fil*escala_min), k)
    else:
        if(n_fil<fil and n_col<col):
            escala_max = max(n_fil/fil, n_col/col)
            im_seam = cv2.resize(im_seam, (int(col*escala_max), int(fil*escala_max)))
            im_seam=podar_columnas_O2(im_seam, int(col*escala_max) - n_col)
            im_seam=podar_filas_O2(im_seam, int(fil*escala_max) - n_fil)
        else:
            im_seam=resize_seam(im,n_fil,n_col, k)
        
    return im_seam

# Seam_Carving para eliminar objetos
# Se le pasa un array con las posiciones de los pixeles que se desean eliminar y 
# se utiliza Seam Carving dandole un valor negativo de Energía a dichos elementos
def eliminar_objeto_vertical(im,pixels):
    # im: imagen a utilizar
    # pixels: array con las posiciones de los pixeles que se desean eliminar
    
    fil, col = im.shape[0], im.shape[1]
    
    # Hacemos una matriz de booleanos con el mismo número de filas y columnas de
    # la imagen. Cada pixel que haya que eliminar aparece como True en dicha matriz
    mapa_eliminar = np.zeros((fil,col),dtype=np.bool)
    
    for pix in pixels:
        mapa_eliminar[pix[0]][pix[1]]=True
    
    # Llevamos la cuenta del número de pixeles a eliminar    
    num_eliminar = len(pixels)

    # Volvemos a calcular los hilos solo en un intervalo de columnas
    col_min=0
    col_max=col
    # Mientras quede algún pixel a eliminar seguimos utilizando Seam Carving para quitar hilos verticales
    while (num_eliminar > 0):
        # quitamos un hilo vertical y actualizamos todos los valores que se utilizan
        im, mapa_eliminar, eliminados, col_min, col_max = cortar_columna_eliminar(im, col_min, col_max, mapa_eliminar)
        # Si no se ha eliminado ninguno, entonces como intervalo ponemos todas las columnas, así se calculan todos los hilos
        # y el pixel a eliminar estará en alguno de ellos
        if(eliminados == 0):
            col_min=0
            col_max=im.shape[1]
        # Actualizamos el número de pixeles que quedan por eliminar
        num_eliminar = num_eliminar-eliminados

    return im

#Seam_Carving para eliminar objetos
def eliminar_objeto_horizontal(im,pixels):
    # im: imagen a utilizar
    # pixels: array con las posiciones de los pixeles que se desean eliminar
    
    col = im.shape[1]
    
    # Rotamos la imagen y los pixeles que vamos a eliminar consecuentemente
    # Después volvemos a rotar la imagen para dejarla en su posición original
    pixeles_traspuestos = []
    for pix in pixels:
        pixeles_traspuestos.append((col-1-pix[1],pix[0]))
    im = np.rot90(im, 1, (0, 1))
    im = eliminar_objeto_vertical(im, pixeles_traspuestos)
    im = np.rot90(im, 3, (0, 1))
    
    return im

# Seam_Carving para eliminar objetos
# Elimina el hilo vertical de mínima energía de la imagen, como a los pixeles a eliminar
# les asignamos valores negativos grandes el hilo mínimo pasará por dichos pixeles.
@jit
def cortar_columna_eliminar(im, col_min, col_max, mapa_eliminar):
    # im: imagen a utilizar
    # col_min: principio del intervalo en que calcular los hilos
    # col_max: fin del intervalo en que calcular los hilos
    # mapa_eliminar: matriz booleana que indica que elementos se deben eliminar
    
    fil, col = im.shape[0], im.shape[1]
    # LLevamos la cuenta de cuantos pixeles de los que hay que eliminar aparecen
    # en el hilo actual que vamos a quitar de la imagen
    eliminados=0
    
    # el valor del intervalo que se va a restar/sumar a mini/maxi respectivamente
    intervalo = 50 
    # matriz de booleanos que tendrá True si el pixel se mantiene y False si se elimina
    matriz_bool = np.ones((fil,col,3), dtype=np.bool)
    
    # Obtenemos la matriz que contiene la mínima energía acumulada por cada hilo
    # del intervalo [col_min, col_max] y la matriz que contiene el backtracking.
    # A la hora de caluclar la energía de cada hilo tenemos en cuenta los elementos
    # que se quieren eliminar.
    # A la función seam_vertical_O2 que utilizamos le pasamos 
    # la imagen sólo con las columnas entre [col_min, col_max]
    M, backtrack = seam_vertical_eliminar(im[0:fil, col_min:col_max], mapa_eliminar[0:fil, col_min:col_max])

    #Hacemos backtracking desde la ultima fila hasta la primera
    j = np.argmin(M[fil-1, 0:M.shape[1]]) #Estamos en la ultima fila
    mini = j
    maxi = j
    for i in range(1,fil): #Para las filas desde la penultima hasta la segunda
        if(mapa_eliminar[fil-i, col_min+j]==True): # la costura pasa por un pixel a eliminar
            eliminados+=1                          # así que aumentamos en 1 eliminados
        matriz_bool[fil-i,col_min+j] = (False,False,False) 
        j = backtrack[fil-i, j]
        mini = min(mini, j)
        maxi = max(maxi, j)
    if(mapa_eliminar[0, col_min+j]==True): #la costura pasa por un pixel a eliminar
        eliminados+=1                      # así que aumentamos en 1 eliminados
    matriz_bool[0,col_min+j] = (False,False,False) #Estamos en la primera fila  
    
    # mini es el índice de la columna más a la izquierda por la que pasa el hilo 
    # de mínima energía menos intervalo
    # maxi es el índice de la columna más a la derecha por la que pasa el hilo 
    # de mínima energía más intervalo
    # También nos aseguramos de que no se salga de la nueva imagen
    mini = max(col_min + mini - intervalo, 0)
    maxi = min(col_min + maxi + intervalo, col - 1)
    
    # matriz para el reshape de mapa_eliminar, mapa_eliminar es una matriz donde 
    # cada elemento tiene un único valor, en lugar de tres como en la imagen
    matriz_bool2 = matriz_bool[0:fil,0:col,0]
        
    return im[matriz_bool].reshape((fil,col-1,3)), mapa_eliminar[matriz_bool2].reshape((fil,col-1)), eliminados, mini, maxi

#Seam_Carving para eliminar objetos
# Calcula la energía de los hilos de la imagen teniendo en cuenta los pixeles que se 
# quieren eliminar
@jit
def seam_vertical_eliminar(im, mapa_eliminar):
    # im: imagen a utilizar
    # mapa_eliminar: matriz que contiene cuyo elemento es true si tenemos que eliminarlo de la imagen.
    
    fil, col = im.shape[0], im.shape[1]
    # matriz para el backtracking
    backtrack = np.empty((fil,col), dtype=np.int)

    # Obtenemos el mapa de energía
    M = mapa_energia(im)
    
    #A los pixeles maracados para eliminar, le ponemos energía negativa grande
    for i in range(fil):
        for j in range(col):
            if(mapa_eliminar[i,j]==1):
                M[i,j]=-10000

    # Calculamos M(i,j) = e(i,j) + min(M(i-1,j-1),M(i-1,j),M(i-1,j+1))
    # pero para j perteneciente a [col_min, col_max]
    for i in range(1, fil):
       #Cuando j=0
       indice = np.argmin(M[i-1, 0:2])
       backtrack[i, 0] = indice
       M[i, 0] += M[i-1, indice]
       #Cuando j={0, col-1}
       for j in range(1, col):
           indice = np.argmin(M[i-1, j-1:j+2]) + j - 1
           backtrack[i,j] = indice 
           M[i,j] += M[i-1, indice]

    return M, backtrack

# Aumentamos las partes importantes de la imagen manteniendo el mismo tamaño de imagen
# Para ello primero hacemos cv2.resize con una escala que le pasamos y luego 
# utilizamos el algoritmo de Seam Carving para volver al tamaño original.
def amplificacion_seam(im,escala_vertical=2, escala_horizontal=2):
    # im: imagen a utilizar
    # escala_vertical: escala en que se va a aumentar el número de filas con cv2.resize
    #               Por defecto, es el doble
    # escala_horizontal: escala en que se va aumentar el número de columnas con cv2.resize
    #               Por defecto, es el doble
    
    im_seam=np.copy(im)
    fil, col = im.shape[0], im.shape[1]
    
    n_fil = fil
    n_col = col
    
    # Si la escala no es mayor que 1 no hacemos nada, porque entonces no se aumentaría la imagen
    if(escala_horizontal > 1):
        n_col = int(col*escala_horizontal)
    if(escala_vertical > 1):
        n_fil = int(fil*escala_vertical)
    # Aumentamos la imagen con cv2.resize
    im_seam = cv2.resize(im_seam, (n_col, n_fil))

    # Volvemos al tamaño original empleando Seam Carving
    im_seam=podar_columnas_O2(im_seam, n_col-col)
    im_seam=podar_filas_O2(im_seam, n_fil-fil)
        
    return im_seam

# Función para pintar en rojo un conjunto de pixeles de la imagen
def pintar_seam(im,mapa):
    # im: imagen a utilizar
    # mapa: matriz con las mismas filas y columnas que la imagen, 
    #       donde el elemento es 1 si se va a pintar
    
    fil, col=mapa.shape
    #para colorear las costuras en rojo
    pixels = []
    for i in range(fil):
        for j in range(col):
            if(mapa[i,j]==1):
                pixels.append((i,j))
    
    im_lineas = color_seam(im,pixels)
    pintaI(im_lineas)

# Colorea en rojo el seam especificado
def color_seam(im, pixels):
  # im: imagen sobre la que dibujar
  # pixels: array que contiene las tuplas de las pos de los píxeles
  
  imagen = np.copy(im)
  for i in pixels:
    # colorear en rojo
    imagen[i[0], i[1]] = (0, 0, 255)
  
  return np.array(imagen)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    #Cargar las 35 imágenes de la base de datos
    vim = []
    castle = leeimagen("./imagenes/castillo.jpg", 1)
    
    for i in range(1,36):
      vim.append(leeimagen("./imagenes/"+str(i)+".jpg", 1))
      
    print("Prueba Seam-Carving sobre imagen aleatoria de la BD.")  
    #Factor de reducción para la imagen aleatoria sobre las columnas
    factor = 0.5
    indice = random.randint(0, len(vim)-1)
    random_im = vim[indice]
    print("\nImagen aleatoria original:")
    pintaI(random_im)
    print("Imagen resultado:")
    pintaI(podar_columnas_O2(random_im, int(random_im.shape[1]*factor)))
    
    print("Aplicar Seam-Carving sobre todas las imágenes de la BD para reducirlas a la mitad.")
    
    #Dividir a la mitad todas las imágenes de la base de datos
    for i in range(len(vim)):
      im=vim[i]
      n_im=podar_columnas_O2(im, int(im.shape[1]/2))
      pintaI(n_im, "./resultados/resized"+str(i+1))
    
    print("Aplicar las distintas funcionalidades sobre la imagen del castillo.")
    
    #Aplicar todas las funcionalidades al castillo 
    #Reducir imagen a la mitad
    ej_castle=podar_columnas_O2(castle, int(castle.shape[1]/2))
    pintaI(ej_castle, "./resultados/reduce_castle")
    #Aumentar imagen
    ej_castle=resize_seam_O(castle,int(castle.shape[0]*1.5), int(castle.shape[1]*2.0))
    pintaI(ej_castle, "./resultados/aumenta_castle")
    #Ampliar imagen
    pintaI(amplificacion_seam(castle,1.5,1.5),"./resultados/ampliar_castle")
    #Eliminar objeto imagen
    pixels_persona = []
    for i in range(500,600):
        for j in range(70,120):
            pixels_persona.append((i,j))
    pintaI(color_seam(castle, pixels_persona), "./resultados/mask_persona")
    #Elimina con seam vertical
    pintaI(eliminar_objeto_vertical(castle,pixels_persona), "./resultados/elimina_vert")
    #Elimina con seam horizontal
    pintaI(eliminar_objeto_horizontal(castle, pixels_persona),"./resultados/elimina_horiz")
   
    print("Cálculo de los tiempos de las funciones optimizadas sobre la imagen del castillo.")

    #Probar las diferencias de tiempo de las funciones optimizadas    
    #Probando la diferencia de podar_columnas 
    
    inicio = time.time()
    pintaI(podar_columnas(castle, int(castle.shape[1]/2)))
    fin = time.time()
    print("Podar Tiempo: ", fin-inicio)
    #Trabajamos en una parte, cada 10 cicos trabajamos con todo
    inicio = time.time()
    pintaI(podar_columnas_O1(castle, int(castle.shape[1]*0.5))) 
    fin = time.time()
    print("Podar_O1 Tiempo: ", fin-inicio)
    #Trabajamos en una parte, excepto al principio    
    inicio = time.time()
    pintaI(podar_columnas_O2(castle, int(castle.shape[1]*0.5))) 
    fin = time.time()
    print("Podar_O2 Tiempo: ", fin-inicio)
    
    #Probando la diferencia de aumentar _columnas
    inicio = time.time()
    pintaI(aumentar_columnas(castle, int(castle.shape[1]*0.5), 10))
    fin = time.time()
    print("Aumentar Tiempo: ", fin-inicio)
    #----------------------------------------
    inicio = time.time()
    pintaI(aumentar_columnas_O(castle, int(castle.shape[1]*0.5), 10))
    fin = time.time()
    print("Aumentar_O Tiempo: ", fin-inicio)
    
    #Probando la diferencia de resize
    inicio = time.time()
    pintaI(resize_seam(castle,int(castle.shape[0]*1.5), int(castle.shape[1]*2.0), 10))
    fin = time.time()
    print("Resize Tiempo : ", fin-inicio)
    #----------------------------------------
    inicio = time.time()
    pintaI(resize_seam_O(castle,int(castle.shape[0]*1.5), int(castle.shape[1]*2.0), 10))
    fin = time.time()
    print("Resize_O Tiempo : ", fin-inicio)

    
    print("Prueba de distintas funcionalidades con imágenes de la BD.")
    
    #Aumentar imágenes 24 y 14
    im24 = leeimagen("./imagenes/24.jpg", 1)
    im14 = leeimagen("./imagenes/14.jpg", 1)
    ej_24=resize_seam_O(im24,int(im24.shape[0]*1.0), int(im24.shape[1]*1.5))
    pintaI(ej_24, "./resultados/24_aumenta")
    ej_14=resize_seam_O(im14,int(im24.shape[0]*1.5), int(im24.shape[1]*2.0))
    pintaI(ej_14, "./resultados/14_aumenta")

    #Ampliar imágenes 3 y 6
    im13 = leeimagen("./imagenes/13.jpg", 1)
    im26 = leeimagen("./imagenes/26.jpg", 1)
    ej_13=amplificacion_seam(im13,1.5,1.5)
    pintaI(im13)
    pintaI(ej_13, "./resultados/13_amplia")
    ej_26=amplificacion_seam(im26,1.7,1.7)
    pintaI(im26)
    pintaI(ej_26, "./resultados/26_amplia")

    #Eliminar objetos imágenes 1 y 15
    im1 = leeimagen("./imagenes/1.jpg", 1)
    im15 = leeimagen("./imagenes/15.jpg", 1)
    #Eliminación persona central con seam_vertical
    pixels_persona1 = []
    for i in range(320,440):
        for j in range(750,900):
            pixels_persona1.append((i,j))
    pintaI(color_seam(im1, pixels_persona1), "./resultados/1_mask")
    #Elimina con seam vertical
    pintaI(eliminar_objeto_vertical(im1,pixels_persona1), "./resultados/1_elimina_vert")
    #Eliminación persona a la derecha con seam_vertical
    pixels_persona15 = []
    for i in range(350,450):
        for j in range(740,800):
            pixels_persona15.append((i,j))
    pintaI(color_seam(im15, pixels_persona15),"./resultados/15_mask")
    #Elimina con seam vertical
    pintaI(eliminar_objeto_vertical(im15,pixels_persona15),"./resultados/15_elimina_vert")
    