import os
import cv2
from WordSegmentation import wordSegmentation, prepareImg
from statistics import mean, stdev
from colorama import init, Fore
import numpy as np

def findOutliers(heights):
  """
    Muestra las alturas de las cajas segmentadas mostrando cuantas desviaciones tipicas se van de la media.
    Los outliers (2 desviaciones típicas más que la media o 1.5 desviaciones típicas menos) se marcan en rojo y con un '#'.
  """
  our_mean = mean(heights)
  our_stDev = stdev(heights)
  print(f'Mean: {our_mean}\nStandard deviation: {our_stDev}')
  for (index, item) in enumerate(heights):
    deviation = (item - our_mean) / our_stDev
    my_str = str(index) + ':' + str(item) + ':' + "{:.4}".format(str(deviation))
    if deviation > 2 or deviation < -1.5:
      my_str = Fore.RED + my_str + '#, ' + Fore.WHITE
    else:
      my_str += ','
    print(my_str, end='')
  print()

def extendBox(sobelBinarized, wordBox):
  """
    Amplia la altura de as cajas segmentadas para rodear completamente la palabra.
  """
  threshold = 220 # Por ejemplo
  (x, y, w, h) = wordBox
  if x == 0 or x+w > len(sobelBinarized):
    return wordBox
  # print('Deciding if augmenting the box')
  if np.max(sobelBinarized[x][y : y+h]) == 255:
    indices = np.where(a == a.max())
    avg = np.average(sobelBinarized[x-2: x+1][indices[0]-2: indices[0]+2])
    if avg > 154:
      print('Box augmented')
      return extendBox(sobelBinarized, (x-1, y, w+1, h))
  else:
    return wordBox



def main():
  """reads images from data/ and outputs the word-segmentation to out/"""

  # read input images from 'in' directory
  imgFiles = os.listdir('../data/')
  for (i,f) in enumerate(imgFiles):
    print('Segmenting words of sample %s'%f)
    
    # read image, prepare it by resizing it to fixed height and converting it to grayscale
    img = prepareImg(cv2.imread('../data/%s'%f), 50, False)
    
    # execute segmentation with given parameters
    # -kernelSize: size of filter kernel (odd integer)
    # -sigma: standard deviation of Gaussian function used for filter kernel
    # -theta: approximated width/height ratio of words, filter function is distorted by this factor
    # - minArea: ignore word candidates smaller than specified area
    imgFiltered, imgThres, res = wordSegmentation(img, kernelSize=25, sigma=8, theta=8, minArea=35)
    
    # write output to 'out/inputFileName' directory
    if not os.path.exists('../out/%s'%f):
      os.mkdir('../out/%s'%f)
    
    heights = []

    img_copy = np.copy(img)

    imgSobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

    # iterate over all segmented words
    print('Segmented into %d words'%len(res))
    for (j, w) in enumerate(res):
      (wordBox, wordImg) = w
      wordBox = extendBox(imgSobel, wordBox)
      (x, y, w, h) = wordBox
      cv2.imwrite('../out/%s/%d.png'%(f, j), wordImg) # save word
      cv2.rectangle(img_copy,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
      heights.append(h)
    findOutliers(heights)

    # print(str(outliers))

    ( _ , imgBinarized ) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ( _ , imgSobelBinarized ) = cv2.threshold(imgSobel, 160, 255, cv2.THRESH_BINARY)



    # output summary image with bounding boxes around words
    cv2.imwrite('../out/%s/summary.png'%f, img_copy)
    cv2.imwrite(f'../out/{f}/filtered.png', imgFiltered)
    cv2.imwrite(f'../out/{f}/binarized.png', imgThres)
    cv2.imwrite(f'../out/{f}/binarizedOriginal.png', imgBinarized)
    cv2.imwrite(f'../out/{f}/sobel.png', imgSobel)
    cv2.imwrite(f'../out/{f}/sobelBinarized.png', imgSobelBinarized)


if __name__ == '__main__':
  init(convert=True)
  main()