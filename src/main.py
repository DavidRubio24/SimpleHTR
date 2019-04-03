from __future__ import division
from __future__ import print_function

import os
import argparse
import cv2
import editdistance
import time
import json
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess

max_upper_deviation = 2
max_lower_deviation = -1.5

class FilePaths:
  "filenames and paths to data"
  fnCharList = '../model/charList.txt'
  fnAccuracy = '../model/accuracy.txt'
  fnTrain = '../data/'
  fnInfer = '../data/test.png'
  fnInfers = ['../data/test.png',
              '../data/test5.png',
              '../data/test1.png',
              '../data/test2.png',
              '../data/test3.png',
              '../data/test4.png']
  fnCorpus = '../data/corpus.txt'


def train(model, loader):
  "train NN"
  epoch = 0 # number of training epochs since start
  bestCharErrorRate = float('inf') # best valdiation character error rate
  noImprovementSince = 0 # number of epochs no improvement of character error rate occured
  earlyStopping = 5 # stop training after this number of epochs without improvement
  while True:
    epoch += 1
    print('Epoch:', epoch)

    # train
    print('Train NN')
    loader.trainSet()
    while loader.hasNext():
      iterInfo = loader.getIteratorInfo()
      batch = loader.getNext()
      loss = model.trainBatch(batch)
      print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

    # validate
    charErrorRate = validate(model, loader)
    
    # if best validation accuracy so far, save model parameters
    if charErrorRate < bestCharErrorRate:
      print('Character error rate improved, save model')
      bestCharErrorRate = charErrorRate
      noImprovementSince = 0
      model.save()
      open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
    else:
      print('Character error rate not improved')
      noImprovementSince += 1

    # stop training if no more improvement in the last x epochs
    if noImprovementSince >= earlyStopping:
      print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
      break


def validate(model, loader):
  "validate NN"
  print('Validate NN')
  loader.validationSet()
  numCharErr = 0
  numCharTotal = 0
  numWordOK = 0
  numWordTotal = 0
  while loader.hasNext():
    iterInfo = loader.getIteratorInfo()
    print('Batch:', iterInfo[0],'/', iterInfo[1])
    batch = loader.getNext()
    (recognized, _) = model.inferBatch(batch)
    
    print('Ground truth -> Recognized') 
    for i in range(len(recognized)):
      numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
      numWordTotal += 1
      dist = editdistance.eval(recognized[i], batch.gtTexts[i])
      numCharErr += dist
      numCharTotal += len(batch.gtTexts[i])
      print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
  
  # print validation result
  charErrorRate = numCharErr / numCharTotal
  wordAccuracy = numWordOK / numWordTotal
  print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
  return charErrorRate


def infer(model, fnImg):
  "recognize text in image provided by file path"
  img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
  batch = Batch(None, [img])
  (recognized, probability) = model.inferBatch(batch, True)
  return recognized[0], probability[0]
  # print('Recognized:', '"' + recognized[0] + '"')
  # print('Probability:', probability[0])



def timeElapsed(pointDescriptor):
  global startTime
  currentTime = time.time()
  print('$$$$$$$$$$$$$$$$$   ', pointDescriptor, ': ' , str(currentTime - startTime), ' seconds   $$$$$$$$$$$$$$$$$$$$')
  startTime = currentTime


def main():
  "main function"
  # optional command line args


  parser = argparse.ArgumentParser()
  parser.add_argument('--train', help='train the NN', action='store_true')
  parser.add_argument('--validate', help='validate the NN', action='store_true')
  parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
  parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
  args = parser.parse_args()

  timeElapsed('Arguments parsed')

  decoderType = DecoderType.BestPath
  if args.beamsearch:
    decoderType = DecoderType.BeamSearch
  elif args.wordbeamsearch:
    decoderType = DecoderType.WordBeamSearch

  timeElapsed('DecoderType decided')

  # train or validate on IAM dataset  
  if args.train or args.validate:
    # load training data, create TF model
    loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

    # save characters of model for inference mode
    open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
    
    # save words contained in dataset into file
    open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

    # execute training or validation
    if args.train:
      model = Model(loader.charList, decoderType)
      train(model, loader)
    elif args.validate:
      model = Model(loader.charList, decoderType, mustRestore=True)
      validate(model, loader)

  # infer text on test image
  else:
    print(open(FilePaths.fnAccuracy).read())

    timeElapsed('Before loading model')

    model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)

    timeElapsed('Loading model took')

    # infer(model, FilePaths.fnInfer)

    prediction, probability = infer(model, '../words/binarizedOriginal.png/29.png')

    print(f'prediction: {prediction}\nProbability: {probability}')

    timeElapsed('Predicting took')

    textForders = os.listdir('../words/')
    for folder in textForders:
      print(f'We are in ../words/{folder}')
      wordImgs = os.listdir(f'../words/{folder}')

      with open(f'../predictions/{folder}.json') as file:
        textDict = json.load(file)
      for wordImg in wordImgs:
        print(f'We are recognizing ../words/{folder}/{wordImg:<15}')
        height_deviation = textDict[f'{wordImg}']['height_deviation']
        if height_deviation < max_upper_deviation and height_deviation > max_lower_deviation:
          recognized, probability = infer(model, f'../words/{folder}/{wordImg}')
          textDict[f'{wordImg}']['prediction']  = recognized
          textDict[f'{wordImg}']['probability'] = probability.item()
          timeElapsed(f'Predicting ../words/{folder}/{wordImg:<15} took')
        else:
          print('#### Too much height deviation.')
        # TODO: escribir una linea en '../predictions/{folder}.txt' indicando {wordImg}, recognized, probability
      file = open(f'../predictions/{folder}.json', 'w+')
      file.write(json.dumps(textDict))
      file.close()

    # for (i, filePath) in enumerate(FilePaths.fnInfers * 3):

    #   infer(model, filePath)

    #   timeElapsed(f'Time elapsed for prediction {i}')



if __name__ == '__main__':
  startTime = time.time()
  main()

