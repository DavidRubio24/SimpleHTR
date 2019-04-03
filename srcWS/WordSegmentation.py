import math
import cv2
import numpy as np
import statistics

img = np.array([])

userAnswer = 0

userAnswerCode = {
  'correct':0,
  'none':1,
  'modify':2,
  'add':3,
  'modify&add':4
}

userBox = []


def isOutlier(box, boxes):
  """
    Comprueba si la caja box=(x, y, w, h) sobresale de entre las demas y probablemente no está bien segmentada.
    Esta implementación solo tiene en cuenta cuanto se desvia la altura de la caja de la media de las cajas.
    TODO: mejorar la predicción
  """

  heights = [i[3] for i in boxes]

  meanHeight = statistics.mean(heights)
  stDevHeight = statistics.stdev(heights)

  outlier = box[3] > meanHeight + 2 * stDevHeight  or  box[3] < meanHeight - 1.5 * stDevHeight

  print(f"{'&' if outlier else '@'}", end='')

  return outlier

def getRectangleGUI(event, x, y, flags, parameters):
  """

  """
  global img, userBox, userAnswer

  img_copy = np.copy(img)

  if event == cv2.EVENT_LBUTTONDOWN:
    userAnswer = userAnswerCode['modify']
    if len(userBox) in (0, 2):
      userBox = [(x, y)]
      # Dibujamos un punto donde el usuario a clicado
      cv2.circle(img_copy, userBox[0], 1, (0,0,255), -1)
    else:
      userBox.append((x, y))
      # Dibujamos un rectanguo entre los puntos que ha clicado el usuario
      cv2.rectangle(img_copy, userBox[0], userBox[1], (0, 0, 255), 1)
    cv2.imshow('image', img_copy)
    print(f'Pressed on ({x},{y})')

  if event == cv2.EVENT_RBUTTONDOWN:
    userAnswer = userAnswerCode['add']
    print('Right clicked')

def askUserConfirmation(image, box):
  """
    Pide al usuario que segmente la palabra correctamente.
  """
  global img
  img = image

  (x, y, w, h) = box
  img_copy = np.copy(image)
  cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0,255,0), 1)
  print('Clicka en dos puntos para seleccionar el rectangulo que los une. Click derecho para deshacer')
  cv2.imshow('image', img_copy)
  cv2.setMouseCallback('image', on_mouse = getRectangleGUI)
  cv2.waitKey(0)
  if userAnswer == userAnswerCode['modify']:
    box = [0, 0, 0, 0]
    box[0] = min(userBox[:][0])
    box[1] = min(userBox[:][1])
    box[2] = abs(userBox[0][0] - userBox[1][0])
    box[3] = abs(userBox[0][1] - userBox[1][1])
    boxes = [box]
  elif userAnswer == userAnswerCode['none']:
    boxes = []
  elif userAnswer == userAnswerCode['correct']:
    boxes = [box]
  elif userAnswer == userAnswerCode['add']:
    newBox = [0, 0, 0, 0]
    newBox[0] = min(userBox[:][0])
    newBox[1] = min(userBox[:][1])
    newBox[2] = abs(userBox[0][0] - userBox[1][0])
    newBox[3] = abs(userBox[0][1] - userBox[1][1])
    boxes = [box, newBox]

  return boxes

def wordSegmentation(img, kernelSize=25, sigma=11, theta=6, minArea=0):
  """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf
  
  Args:
    img: grayscale uint8 image of the text-line to be segmented.
    kernelSize: size of filter kernel, must be an odd integer.
    sigma: standard deviation of Gaussian function used for filter kernel.
    theta: approximated width/height ratio of words, filter function is distorted by this factor.
    minArea: ignore word candidates smaller than specified area.
    
  Returns:
    List of tuples. Each tuple contains the bounding box and the image of the segmented word.
  """

  # apply filter kernel
  kernel = createKernel(kernelSize, sigma, theta)
  imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
  # cv2.imshow('image', imgFiltered)
  # cv2.waitKey(0)
  (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  imgThres = 255 - imgThres

  # find connected components. OpenCV: return type differs between OpenCV2 and 3
  if cv2.__version__.startswith('3.'):
    (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  else:
    (   components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  # append components to result
  res = []
  for c in components:
    # skip small word candidates
    if cv2.contourArea(c) < minArea:
      continue
    # append bounding box and image of word to result list
    currBox = cv2.boundingRect(c) # returns (x, y, w, h)
    res.append(currBox)
  boxes = set()
  for (index, box) in enumerate(res):
    if isOutlier(box, res):
      boxes.update(askUserConfirmation(img, box))
    else:
      boxes.add(box)


  # return list of words, ordenadas de izquierda a derecha y de arriba a abajo más o menos
  return imgFiltered, imgThres, sorted(list(boxes), key=lambda entry: 10 * (entry[1] + entry[3]//2) + (entry[0] + entry[2]//2))
  # return imgFiltered, imgThres, res


def prepareImg(img, height, resize=True):
  """convert given image to grayscale image (if needed) and resize to desired height"""
  assert img.ndim in (2, 3)
  if img.ndim == 3:
    # Si la imagen está en RGB se convierte a escala de grises
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  if resize:
    h = img.shape[0]
    factor = height / h
    img = cv2.resize(img, dsize=None, fx=factor, fy=factor)
  return img


def createKernel(kernelSize, sigma, theta):
  """create anisotropic filter kernel according to given parameters"""
  assert kernelSize % 2 # must be odd size
  halfSize = kernelSize // 2
  
  kernel = np.zeros([kernelSize, kernelSize])
  sigmaX = sigma
  sigmaY = sigma * theta
  
  for i in range(kernelSize):
    for j in range(kernelSize):
      x = i - halfSize
      y = j - halfSize
      
      expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
      xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
      yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
      
      kernel[i, j] = (xTerm + yTerm) * expTerm

  kernel = kernel / np.sum(kernel)
  return kernel
