import os
import cv2
import json
from WordSegmentation import wordSegmentation, prepareImg
from statistics import mean, stdev
from colorama import init, Fore
import numpy as np

def findOutliers(heights):
  """
    Muestra las alturas de las cajas segmentadas mostrando cuantas desviaciones tipicas se van de la media.
    Los outliers (2 desviaciones típicas más que la media o 1.5 desviaciones típicas menos) se marcan en rojo y con un '#'.
  """
  # Calculamos y mostramos la media y la desviación típica
  our_mean = mean(heights)
  our_stDev = stdev(heights)
  print(f'Mean: {our_mean}\nStandard deviation: {our_stDev}')

  # Para cada altura mostramos cuantas desviaciones típicas se desvía de la media
  for (index, item) in enumerate(heights):
    deviation = (item - our_mean) / our_stDev
    # Usamos el formato nº_imagen:altura:deviacion
    my_str = str(index) + ':' + str(item) + ':' + "{:.4}".format(str(deviation))
    # Si es un outlier lo marcamos en rojo
    if deviation > 2 or deviation < -1.5:
      my_str = Fore.RED + my_str + Fore.WHITE
    print(my_str, ',', end='')
  print() # Al final ponemos un salto de linea

def extendBox(img, wordBox):
  """
    Amplia la altura de as cajas segmentadas para rodear completamente la palabra.
    TODO: conseguir que funcione.
    TODO: La idea a medio implementar se puede mejorar aplicando Sobel solo a los pixels necesarios.
  """
  threshold = 154 # Por ejemplo
  (x, y, w, h) = wordBox

  # Detectamos lo bordes verticales de la imagen
  imgSobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

  # Si estamos en un borde no ampliamos la imagen
  if x == 0 or x+w > len(imgSobel):
    return wordBox

  if np.max(imgSobel[x][y : y+h]) == 255:
    indices = np.where(a == a.max())
    avg = np.average(imgSobel[x-2: x+1][indices[0]-2: indices[0]+2])
    if avg > threshold:
      print('Box augmented')
      return extendBox(imgSobel, (x-1, y, w+1, h))
  else:
    return wordBox



def main():
  """reads images from data/ and outputs the word-segmentation to out/"""

  # read input images from directory
  imgFiles = os.listdir('../text/')
  for (i,f) in enumerate(imgFiles):
    print(f'Segmenting words of sample {f}')
    
    # read image, prepare it by converting it to grayscale
    # No redimensionamos para poder leer textos enteros en vez de lineas
    img = prepareImg(cv2.imread('../text/%s'%f), 50, resize=False)
    
    # execute segmentation with given parameters
    # -kernelSize: size of filter kernel (odd integer)
    # -sigma: standard deviation of Gaussian function used for filter kernel
    # -theta: approximated width/height ratio of words, filter function is distorted by this factor
    #         cuanto más grande sea sigma más provable que una palabras, cuanto más pequeño más falsos positivos
    # -minArea: ignore word candidates smaller than specified area
    #           esta variable pierde sentido en imagenes de cualquier tamaño. TODO: hacer el area relativa
    imgFiltered, imgThres, res = wordSegmentation(img, kernelSize=25, sigma=8, theta=8, minArea=35)
    
    # write output to '../words/inputText/' directory
    if not os.path.exists(f'../words/{f}'):
      os.mkdir(f'../words/{f}')
    
    # Las alturas de las palabras reconocidas
    heights = []

    img_copy = np.copy(img) # Copia sobre la que dibujar los rectángulos


    # iterate over all segmented words
    print(f'Segmented into {len(res)} words')
    for (index, wordBox) in enumerate(res):
      # wordBox = extendBox(img, wordBox) # Extendemos las imágenes
      (x, y, w, h) = wordBox
      wordImg = img[y : y+h, x : x+w]
      # save word
      cv2.imwrite(f'../words/{f}/{index:0>3}.png', wordImg)
      # draw bounding box in summary image
      cv2.rectangle(img_copy,(x,y),(x+w,y+h),0,1)

      heights.append(h)


    # Guardamos las desviacioes de las alturas
    our_mean  = mean(heights)
    our_stDev = stdev(heights)

    wordsDict = {}
    for (index, item) in enumerate(heights):
      deviation = (item - our_mean) / our_stDev
      wordsDict[f'{index:0>3}.png'] = {'height_deviation':deviation}

    # Usamos el formato JSON, que python se encarga de formatear
    print(f'Saving in ../predictions/{f}.json...')
    file = open(f'../predictions/{f}.json', 'w+')
    file.write(json.dumps(wordsDict))
    file.close()

    # Mostramos las alturas y desviaciones de cada palabra
    findOutliers(heights)

    ( _ , imgBinarized ) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgSobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    ( _ , imgSobelBinarized ) = cv2.threshold(imgSobel, 160, 255, cv2.THRESH_BINARY)

    if not os.path.exists(f'../summary/{f}'):
      os.mkdir(f'../summary/{f}')

    # Muestra las distintas imágenes que hemos usado para obtener la segmentación
    cv2.imwrite(f'../summary/{f}/summary.png', img_copy)
    cv2.imwrite(f'../summary/{f}/filtered.png', imgFiltered)
    cv2.imwrite(f'../summary/{f}/binarized.png', imgThres)
    cv2.imwrite(f'../summary/{f}/binarizedOriginal.png', imgBinarized)
    cv2.imwrite(f'../summary/{f}/sobel.png', imgSobel)
    cv2.imwrite(f'../summary/{f}/sobelBinarized.png', imgSobelBinarized)


if __name__ == '__main__':
  init(convert=True)
  main()