# Asignatura : Visión por Computador
# Práctica 3 : Detección de puntos relevantes y Construcción de panoramas
# Curso 2019/2020
# Alumno : Daniel Bolaños Martínez 76592621E

import numpy as np
import cv2
import random
import math
from matplotlib import pyplot as plt

# Funciones Práctica 0 y 1

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
  #cv2.imwrite('./fotos_memoria/'+title+'.png', im)
  return imgrgb

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

def neighbors(im, i, j, winsize):
  # im: imagen de origen
  # i,j: posición del pixel
  # winsize: tamaño del entorno
  vecinos = []
  indices = []
  # si winsize es 3 los índices empiezan en (i-1,j-1)
  # si winsize es 5 en (i-2, j-2)...
  diff = int(winsize/2)
  # Añade las posiciones de los vecinos en un entorno (winsize X winsize)
  for k in range(0, winsize):
     for m in range(0, winsize):
       if (i-diff+k,j-diff+m)!=(i,j):
         indices.append((i-diff+k,j-diff+m))  
  # devuelve una lista con los valores de los vecinos
  # devuelve la lista vacía si es un valor del borde
  # fila o columna = 0 o im.shape
  for k in range(len(indices)):
    if indices[k][0]>=0 and indices[k][1]>=0 and indices[k][0]<im.shape[0] \
       and indices[k][1]<im.shape[1]:
      vecinos.append(im[indices[k][0]][indices[k][1]])

  return vecinos

def supresionNoMax(im, winsize):
  # im: imagen a aplicar la supresión de máximos
  # threshold: umbral valor a partir del cual se suprime ese píxel
  # winsize: tamaño del entorno de búsqueda del máximo
  # hago una copia de la matriz original sobre la que haremos la supresión
  copia = np.copy(im)
  # para cada pixel de la imagen
  for i in range(im.shape[0]):
    for j in range(im.shape[1]):
      # calculamos el máximo de los adyacentes
      maximo = max(neighbors(im, i, j, winsize))
      # si el elemento actual no es máximo de su entorno su valor=0.0
      if im[i][j] < maximo:
        copia[i][j] = 0.0
      
  return copia

"""
-----------------
## EJERCICIO 1 ##
-----------------
"""

def ejercicio1A(im, blockSize, threshold, winsize, ksizeSobel=3, ksizeDeriv=3, niveles=4):
  # im: imagen en escala de grises a la que se va a calcular los puntos Harris
  # blockSize: tamaño de la ventana de entorno para cornerEigenValsAndVecs
  # ksizeSobel: tamaño de la máscara de Sobel para cornerEigenValsAndVecs
  # ksizeDeriv: tamaño de la máscara para las derivadas
  # threshold: umbral valor a partir del cual se suprime ese píxel
  # winsize: tamaño del entorno de búsqueda del máximo en la supresión
  # niveles: niveles de la pirámide gaussiana
  im = im.astype(np.float32) 
  # Calculo la pirámide Gaussiana de la imagen
  pirIm = [im]
  for i in range(niveles-1):
    pirIm.append(cv2.pyrDown(pirIm[i]))
    
  for n in range(niveles):
    # Calcula los valores y vectores propios de bloques
    # de imágenes para detección de esquinas
    eigenVals = cv2.cornerEigenValsAndVecs(pirIm[n], blockSize=blockSize, ksize=ksizeSobel)
    # Nos quedamos con las matrices que contienen los valores propios
    eigenVals = cv2.split(eigenVals)
    l1 = eigenVals[0]
    l2 = eigenVals[1]
    # Pasamos la matriz a float y hacemos una copia sobre la que 
    # calcularemos los valores de los puntos de Harris
    fim = np.copy(pirIm[n])
    # Calculamos la matriz fim donde el valor de cada píxel se calcula como
    # f_p = (l1_p*l2_p)/(l1_p+l2_p)
    for i in range(fim.shape[0]):
      for j in range(fim.shape[1]):
        if (l1[i][j]+l2[i][j]) == 0.0:
          fim[i][j] = 0.0
        else:
          fim[i][j] = (l1[i][j]*l2[i][j])/(l1[i][j]+l2[i][j])
    # Ponemos a 0.0 los valores de los píxeles de la imagen menores al umbral
    for i in range(fim.shape[0]):
      for j in range(fim.shape[1]):
        if fim[i][j] < threshold:
          fim[i][j] = 0.0
    # Aplicamos la supresión de no máximos teniendo en cuenta el tamaño
    # de entorno del cálculo de los máximos en la nueva imagen
    pirIm[n] = supresionNoMax(fim, winsize)
  # Calculo las derivadas de la imagen original haciendo uso de la
  # función de la P1
  dxIm = convMasks1D(im, dx=1, dy=0, ksize=ksizeDeriv)
  dyIm = convMasks1D(im, dx=0, dy=1, ksize=ksizeDeriv)
  # Calculo el suavizado gaussiano sobre las derivadas haciendo uso
  # de la función de la P1, calcula ksize a partir del sigma
  dxIms = funcionGaussiana(dxIm, sigmaX=4.5)
  dyIms = funcionGaussiana(dyIm, sigmaX=4.5)
  # Calculo la pirámide Gaussiana de la imagen dxIms y dyIms
  pirImX = [dxIms]
  pirImY = [dyIms]
  for i in range(niveles-1):
    pirImX.append(cv2.pyrDown(pirImX[i]))
    pirImY.append(cv2.pyrDown(pirImY[i]))
  
  # Almacenamos un vector de KeyPoints con los valores de:
  # posición del pixel = (x,y)
  # escala = block_size*nivel_pirámide
  # orientación = arctg(sen(theta)/cos(theta))
  keypoints = [] 
  # Para cada píxel mayor que 0 calculamos su posición relativa a la matriz
  # original, su tamaño y su ángulo de orientación en grados
  for n in range(niveles):
    kp = []
    for x in range(pirIm[n].shape[0]):
      for y in range(pirIm[n].shape[1]):
        if pirIm[n][x][y] > 0.0:
          u = math.sqrt(pirImX[n][x][y]**2+pirImY[n][x][y]**2)
          # Comprobamos si |u|=0 en ese caso conseno = seno = 0
          if u == 0.0:
            coseno = 0.0
            seno = 0.0
          else:
            # coseno = u_1 / |u|
            coseno = pirImX[n][x][y]/u
            # seno = u_2 / |u|
            seno = pirImY[n][x][y]/u            
          # Returns value of atan(y/x) in radians. 
          # Returns a numeric value between –pi and pi 
          theta = (360*math.atan2(seno, coseno))/(2*math.pi)
          # Si el ángulo es negativo, lo pasamos a positivo sumando 360º
          if(theta < 0.0):
            theta += 360.0
          kp.append(cv2.KeyPoint(y*2**n, x*2**n, _size=blockSize*(n+1), _angle=theta))
    keypoints.append(kp)
  
  return keypoints

def ejercicio1B(im, imC, blockSize, threshold, winsize):
  # Vector de vectores de keypoints
  v_kp = []
  # Calculo los keypoints para las combinaciones de parámetros especificadas
  if len(blockSize) == len(threshold) == len(winsize):
    for n in range(len(blockSize)):
      v_kp.append(ejercicio1A(im, blockSize[n], threshold[n], winsize[n]))
  # Para cada vector de keypoints dibujo los keypoints en color rojo
  # de todas las escalas y calculo su valor de puntos totales
  for i in range(len(v_kp)):
    total = 0
    imNew = np.copy(imC)
    for j in range(len(v_kp[i])):
      total += len(v_kp[i][j])
      # Dibujo los keypoints para cada octava en rojo
      imNew = cv2.drawKeypoints(imNew, v_kp[i][j], outImage=np.array([]), color=(0, 0, 255), flags=4)
      # Calculo el total de keypoints para la imagen
    pintaI(imNew, "ap1B"+str(i))
    print("Total de KeyPoints calculados = " + str(total))
    
# Genera un color aleatorio
def random_color():
  return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

def ejercicio1C(im, kpoints):
  total = 0
  im_copia = np.copy(im)
  for i in range(len(kpoints)):
    # Selecciono un color aleatorio
    color = random_color()
    total += len(kpoints[i])
    # Limpio la imagen de la octava para dibujar los nuevos puntos
    imOctava = np.copy(im_copia)
    # Pinto en rojo por separado los keypoints para cada escala
    imOctava = cv2.drawKeypoints(imOctava, kpoints[i], outImage=np.array([]), color=(0, 0, 255), flags=4)
    pintaI(imOctava, "Octava"+str(i+1))
    # Muestro los keypoints para cada octava
    print("Número de KeyPoints en la escala "+ str(i+1) + " = " + str(len(kpoints[i])))
    # Dibujo los keypoints para cada octava cada una en un color diferente
    im = cv2.drawKeypoints(im, kpoints[i], outImage=np.array([]), color=color, flags=4)
  pintaI(im, "ap1C")
  # Calculo el total de keypoints para la imagen
  print("Total de KeyPoints calculados = " + str(total))
  
# Devuelve las posiciones de los píxeles refinados para cada escala
def refKeyPts(im, kpoints, winSize=5, zeroZone=-1, stop_criteria= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
  im = im.astype(np.float32)
  new_points = []
  # Calculo la pirámide Gaussiana de la imagen
  pirIm = [im]
  for i in range(len(kpoints)-1):
    pirIm.append(cv2.pyrDown(pirIm[i]))  
  # Almacenamos las coordenadas originales de cada keypoint para cada escala
  for i in range(len(pirIm)):
    v_points = []
    for j in range(len(kpoints[i])):
      v_points.append(kpoints[i][j].pt)
    new_points.append(np.float32(v_points))
  # Calculo el refinamiento de keypoints para los puntos clave 
  # de cada escala con la función cornerSubPix
  for i in range(len(pirIm)):
    cv2.cornerSubPix(pirIm[i], new_points[i], (winSize, winSize), (zeroZone, zeroZone), stop_criteria)    
  # Devuelve las coordenadas de los nuevos puntos para cada escalas en new_points
  return new_points

def ejercicio1D(im, kpoints, n=3):
  # Creamos una copia de la imagen que usaremos para obtener las n muestras
  subim = np.copy(im)
  # Llamamos a la función que refina los píxeles con cornerSubPix
  ref_points = refKeyPts(im, kpoints) 
  # Seleccionaremos aleatoriamente en cada lista 3 puntos y sus equiv. refinados
  kp_select = []
  kp_ref = []
  encontrados = 0
  # Buscamos aleatoriamente entre n*10 keypoints hasta elegir 3 distintos
  for i in range(n*10):
    # Elegimos aleatoriamente n keypoints de la imagen
    octava = random.randint(0, len(kpoints)-1)
    pixel = random.randint(0, len(kpoints[octava])-1)
    (x,y) = ref_points[octava][pixel]
    (x0, y0) = kpoints[octava][pixel].pt
    # Repetimos el bucle hasta que haya diferencias
    # entre los puntos originales y los refinados
    if (x0,y0) != (x,y) and encontrados < n:
      encontrados += 1
      # Cambiamos los valores del ángulo y tamaño para mejorar la visualización
      # en la matriz 10x10 con zoom x5
      kp_ref.append(cv2.KeyPoint(x, y, 1.0, 0.0))
      kp_select.append(cv2.KeyPoint(x0, y0, 1.0, 0.0))
  # Coloreamos los keypoints seleccionados en rojo para los keypoints
  # originales y en verde para los keypoints refinados
  subim = cv2.drawKeypoints(subim, kp_select, outImage=np.array([]), color=(0, 0, 255), flags=4)
  subim = cv2.drawKeypoints(subim, kp_ref, outImage=np.array([]), color=(0, 255, 0), flags=4)
  # Para cada keypoint seleccionado
  for i in range(n):
    # Calculamos su posición en la matriz original
    (x,y) = kp_select[i].pt
    x,y = int(x), int(y)
    # Dibujamos una matriz 10x10 alrededor del keypoint original (en rojo)
    # y aumentamos la imagen con un factor x 5
    pintaI(cv2.resize(subim[y-5:y+5,x-5:x+5], (0,0), fx=5, fy=5), 'Frag'+str(i+1))
  
"""
-----------------
## EJERCICIO 2 ##
-----------------
"""

# Criterio BruteForce+crossCheck
def bruteForceCC(im1, im2, akaze):
  # im1: imagen 1 para la correspondencia
  # im2: imagen 2 para la correspondencia
  # akaze: detector y extractor AKAZE
  # Creamos el matcher usando BFMatcher
  # que implementa un matcher de descriptores por Fuerza Bruta
  # El parámetro crossCheck activo indica que la correspondendia
  # debe darse en ambos sentidos
  bf = cv2.BFMatcher(crossCheck=True)
    
  # La función detectAndCompute() que se encarga de detectar 
  # los keypoints y calcular los descriptores de los mismos
  # para cada imagen. No especificamos el parámetro mask
  kpts1, desc1 = akaze.detectAndCompute(im1, None)
  kpts2, desc2 = akaze.detectAndCompute(im2, None)
    
  # Emparejamos descriptores haciendo uso del matcher
  # que hemos definido anteriormente
  matches = bf.match(desc1, desc2)
  '''
  Dibujamos las 100 correspondencias usando la función drawMatches
  im1	First source image.
  kpts1	Keypoints from the first source image.
  im2	Second source image.
  kpts2	Keypoints from the second source image.
  matches1to2	Matches from the first image to the second one
  which means that kpts1[i] has a corresponding point in kpts2[matches[i]] .
  outImg	Output image.
  flag = 2 -> NOT_DRAW_SINGLE_POINTS (No dibuja keypoints individuales)
  '''
  # Si hay más de 100 matches elegimos 100 aleatoriamente de la muestra
  if len(matches) > 100:
    subset = random.sample(matches, 100)
  else:
    subset = matches
  # Obtenemos una muestra de 100 correspondencias aleatorias
  im3 = cv2.drawMatches(im1, kpts1, im2, kpts2, matches1to2=subset,\
                        outImg=np.array([]), flags=2)

  return ((kpts1, desc1), (kpts2, desc2), matches, im3)

# Criterio Lowe-Average-2NN
def loweAvg2NN(im1, im2, akaze, k=0.7):
  # im1: imagen 1 para la correspondencia
  # im2: imagen 2 para la correspondencia
  # akaze: detector y extractor AKAZE
  # k: constante diferencia distancias
  # Creamos el matcher usando BFMatcher
  # crossCheck=False es necesario para usar knnMatch con k=2 
  # comportamiento predeterminado cuando encuentre los 
  # k vecinos más cercanos para cada descriptor de consulta.
  bf = cv2.BFMatcher(crossCheck=False)
    
  # La función detectAndCompute() que se encarga de detectar 
  # los keypoints y calcular los descriptores de los mismos
  # para cada imagen. No especificamos el parámetro mask
  kpts1, desc1 = akaze.detectAndCompute(im1, None)
  kpts2, desc2 = akaze.detectAndCompute(im2, None)
    
  # Emparejamos descriptores haciendo uso del matcher
  # que hemos definido anteriormente
  matches = bf.knnMatch(desc1, desc2, k=2)
  
  # Seleccionamos las correspondencias correctas
  # Consideramos correctas las que minimicen la
  # distancia respecto a su pareja salvo una constante k
  correct_matches = []
  for a,b in matches:
    # Aplicar ratio k
    if a.distance < k*b.distance:
      correct_matches.append(a)
  '''
  Dibujamos las 100 correspondencias usando la función drawMatches
  im1	First source image.
  kpts1	Keypoints from the first source image.
  im2	Second source image.
  kpts2	Keypoints from the second source image.
  matches1to2	Matches from the first image to the second one
  which means that kpts1[i] has a corresponding point in kpts2[matches[i]] .
  outImg	Output image.
  flag = 2 -> NOT_DRAW_SINGLE_POINTS (No dibuja keypoints individuales)
  '''
  # Si hay más de 100 matches elegimos 100 aleatoriamente de la muestra
  if len(correct_matches) > 100:
    subset = random.sample(correct_matches, 100)
  else:
    subset = correct_matches
  # Obtenemos una muestra de 100 correspondencias aleatorias
  im3 = cv2.drawMatches(im1, kpts1, im2, kpts2, matches1to2=subset,\
                        outImg=np.array([]), flags=2)
  return ((kpts1, desc1), (kpts2, desc2), correct_matches, im3)
    
def ejercicio2(im1, im2, k=[0.7]):
  # Creamos el objeto de la clase que implementa el detector
  # de keypoints y extractor de descriptores AKAZE
  akaze = cv2.AKAZE_create()
  # Genera imágenes con correspondencia BruteForce+crossCheck
  kpdesc1, kpdesc2, matches, im = bruteForceCC(im1, im2, akaze)
  pintaI(im, "BruteForceMatch")
  for i in range(len(k)):
    # Genera imágenes con correspondencia Lowe-Average-2NN
    kpdesc1, kpdesc2, matches, im = loweAvg2NN(im1, im2, akaze, k[i])
    pintaI(im, "2NNMatch"+str(k[i]).replace(".", "_"))
    
"""
-----------------
## EJERCICIO 3 ##
-----------------
"""
# Función obtenida de stack overflow
# https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv/41670793
def autocrop(im):
  # im: imagen a la que queremos realizar el recorte
  # mejorar eficiencia M[0,:,:]==np.zeros()
  im = im.astype(np.float32)
  # Hacemos una copia en blanco y negro de 
  # la imagen si im es a color
  if len(im.shape)==3:
    crop = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  else:
    crop = np.copy(im)
  rows = []
  cols = []
  # Obtenemos un vector de booleanos con True si esa fila/columna
  # contiene algún valor distinto de cero o False si no lo tiene
  rbool = np.any(crop.T!=0, axis=0)
  cbool = np.any(crop.T!=0, axis=1)
  # Agregamos a un vector las posicións de las filas de ceros
  for i in range(len(rbool)):
    if not rbool[i]:
      rows.append(i)
  # Agregamos a un vector las posicións de las columnas de ceros
  for j in range(len(cbool)):
    if not cbool[j]:
      cols.append(j)
  # Eliminamos de la imagen original las filas y columnas de ceros
  # que representaban zonas en negro del canvas sin imagen 
  im = np.delete(im, rows, axis=0)
  im = np.delete(im, cols, axis=1)
  
  return im

# Calculo la homografía que lleva la primera imagen al centro del canvas
def homografia_centro(im, canvas):
  # im: imagen de la cual queremos obtener la homografía para centrarla en el canvas
  # canvas: fondo donde se mostrará el mosaico y que tiene como dimensiones la suma
  #         de las dimensiones del total de imágenes a usar
  # Se obtienen las dimensiones de la imagen central y del canvas
  height_canvas, width_canvas = canvas.shape[0], canvas.shape[1]
  height_im, width_im = im.shape[0], im.shape[1]
  # Se calculan las coordenadas de origen de la imagen central
  x = (width_canvas - width_im)/2
  y = (height_canvas - height_im)/2 
  # Generamos la matriz 3x3 con los valores 
  # de la traslación y la diagonal a unos
  # |1 0 x|
  # |0 1 y|
  # |0 0 1|
  M0 = np.array([[1,0,x], [0,1,y], [0,0,1]])                
  return M0

# Calculamos la homografía entre los puntos de las imágenes
def homografia(im1, im2, k=0.7, eps=1.0):
  # im1: imagen 1 para calcular la homografía
  # im2: imagen 2 para calcular la homografía
  # k: constante k para la función LoweAvg2NN
  # eps: error para la homeografía en la función findHomography
  # Creamos el objeto de la clase que implementa el detector
  # de keypoints y extractor de descriptores AKAZE
  akaze = cv2.AKAZE_create()
  # Se calculan las correspondencias con el criterio LoewAvg2NN
  # que hemos implementado en el apartado anterior
  (kpts1, desc1), (kpts2, desc2), matches, im = loweAvg2NN(im1, im2, akaze, k)
  # Ordenamos respecto al orden de los matches
  src_pts = np.array([kpts1[m.queryIdx].pt for m in matches])
  dst_pts = np.array([kpts2[m.trainIdx].pt for m in matches])
  # Calculamos homografía y la devolvemos
  H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, eps)[0]
  return H

# Calculamos previamente todas las homografías necesarias 
# para trasladar cada imagen al mosaico y luego realiza la traslación 
def mosaico(vim, k=0.7, eps=1.0):
  # vim: vector que contiene las imagenes que componen el mosaico
  # k: constante k para la función LoweAvg2NN
  # eps: error para la homeografía en la función findHomography
  # Calculamos las dimensiones del canvas como la sumatoria de alto y ancho
  # de cada imagen contenida en el vector vim
  height = sum([im.shape[0] for im in vim])
  width = sum([im.shape[1] for im in vim])
  # Creamos una matriz canvas con 3 o 1 canales según sean 
  # a color o a grises las imágenes de vim
  if len(vim[0].shape) == 3:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
  else:
    canvas = np.zeros((height,width), dtype=np.uint8)
  # Calculamos la homografía necesaria para trasladar
  # la primera imagen al centro del canvas
  M0 = homografia_centro(vim[0], canvas)
  # Usamos borde constante para obtener el canvas en negro
  # Creamos una lista con las homografías que vayamos calculando
  # para finalmente aplicarlas todas al final
  homografias = [M0]
  # Calculamos las homeografías restantes para formar el mosaico
  for i in range(len(vim)-1):
    # Obtenemos las homografías para las imágenes dos a dos
    H = homografia(vim[i+1], vim[i], k, eps)
    # Multiplicamos por la homografía del paso anterior para
    # conseguir la composición de las homografías
    M = np.dot(homografias[i], H)
    homografias.append(M)
  # Trasladamos todas las imágenes al canvas utilizando
  # las homografías calculadas y fijando el borde transparente
  for i in range(len(homografias)):
    canvas = cv2.warpPerspective(vim[i], homografias[i], (width, height), \
                                 dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
  # Usamos la función autocrop para eliminar los bordes negros de la imagen
  canvas = autocrop(canvas)
  return canvas

# Calculamos un mosaico para N=2
def ejercicio3(vim, k=0.7, eps=1, name="Mosaico_ej3"):
  # vim: vector que contiene las imagenes que componen el mosaico
  # k: constante k para la función LoweAvg2NN
  # eps: error para la homeografía en la función findHomography
  mosaico3 = mosaico(vim, k, eps)
  pintaI(mosaico3, name)
  
# Calculamos un mosaico para todas las imágenes
def ejercicio4(vim, k=0.7, eps=1, name="Mosaico_ej4"):
  # vim: vector que contiene las imagenes que componen el mosaico
  # k: constante k para la función LoweAvg2NN
  # eps: error para la homeografía en la función findHomography
  mosaico4 = mosaico(vim, k, eps)
  pintaI(mosaico4, name)

if __name__ == '__main__': 
  im1 = leeimagen("imagenes/Yosemite1.jpg", 0)
  im2 = leeimagen("imagenes/Yosemite2.jpg", 0)
  im1C = leeimagen("imagenes/Yosemite1.jpg", 1)
  im2C = leeimagen("imagenes/Yosemite2.jpg", 1)
  print("EJERCICIO 1: \n")
  print("Apartado A. Detectar puntos Harris.")
  # Calculo de los puntos Harris para 4 niveles para im1
  print("Calculando...")
  keypts = ejercicio1A(im1, blockSize=5, threshold=50, winsize=3)
  input("Pulse la tecla ENTER para continuar")
  print("Apartado B. Probar varios parámetros.")
  # Probar con varios parámetros para obtener N > 2000
  # Modificamos el tamaño de blockSize, threshold y winsize
  # Dejaremos fijos los niveles en 4 y los tamaños de la máscara de derivadas y Sobel
  # que son conceptos que ya tratamos en la práctica 1
  ejercicio1B(im1, im1C, blockSize=[7,7], threshold=[5,50], winsize=[5,5])
  input("Pulse la tecla ENTER para continuar")
  print("Apartado C. Dibujar keypoints para cada octava.")
  # Dibujamos los keypoints para cada octava sobre los puntos calculados
  ejercicio1C(im1C, keypts)
  input("Pulse la tecla ENTER para continuar")
  print("Apartado D. Hacer refinamiento con cornerSubPix.")
  # Hacemos el refinamiento con cornerSubPix de 3 keypoints aleatorios 
  ejercicio1D(im1, keypts) 
  input("Pulse la tecla ENTER para continuar")
  print("\nEJERCICIO 2: \n")
  # Dibujamos imágenes con correspondencias BruteForce
  # y LoweAverage 2NN con valores de k = 0.7 y 0.8
  ejercicio2(im1, im2, [0.7, 0.8])
  input("Pulse la tecla ENTER para continuar")
  print("\nEJERCICIO 3: \n")
  # Dibujamos el Mosaico para N = 2 utiliando las imágenes de Yosemite
  # Podemos variar la constante k usada para calcular las correspondencias
  # para LoweAverage2NN y el valor de eps usado en findHomography
  ejercicio3([im1C, im2C])
  input("Pulse la tecla ENTER para continuar")
  print("\nEJERCICIO 4: \n")
  # Dibujamos el Mosaico para todas las imágenes 
  # utilizando las imágenes de los exteriores de la ETSIIT (mosaico00)
  ime2 = leeimagen("imagenes/mosaico002.jpg", 1)
  ime3 = leeimagen("imagenes/mosaico003.jpg", 1)
  ime4 = leeimagen("imagenes/mosaico004.jpg", 1)
  ime5 = leeimagen("imagenes/mosaico005.jpg", 1)
  ime6 = leeimagen("imagenes/mosaico006.jpg", 1)
  ime7 = leeimagen("imagenes/mosaico007.jpg", 1)
  ime8 = leeimagen("imagenes/mosaico008.jpg", 1)
  ime9 = leeimagen("imagenes/mosaico009.jpg", 1)
  ime10 = leeimagen("imagenes/mosaico010.jpg", 1)
  ime11 = leeimagen("imagenes/mosaico011.jpg", 1)
  imy1 = leeimagen("imagenes/yosemite1.jpg", 1)
  imy2 = leeimagen("imagenes/yosemite2.jpg", 1)
  imy3 = leeimagen("imagenes/yosemite3.jpg", 1)
  imy4 = leeimagen("imagenes/yosemite4.jpg", 1)
  imy5 = leeimagen("imagenes/yosemite5.jpg", 1)
  imy6 = leeimagen("imagenes/yosemite6.jpg", 1)
  imy7 = leeimagen("imagenes/yosemite7.jpg", 1)
  # Variamos k si es necesario para que podamos obtener 
  # mejores correspondencias para generar el mosaico
  print("Mosaico ETSIIT")
  ejercicio4([ime2, ime3, ime4, ime5, ime6, ime7, ime8, ime9, ime10, ime11], name="ETSIIT")
  print("Mosaico Yosemite parte 1")
  ejercicio4([imy1, imy2, imy3, imy4], name="Yosemite1")
  print("Mosaico Yosemite parte 2")
  ejercicio4([imy5, imy6, imy7], name="Yosemite2")