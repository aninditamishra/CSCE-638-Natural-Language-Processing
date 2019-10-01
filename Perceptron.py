import sys
import getopt
import os
import math
import operator
from random import shuffle

class Perceptron:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Perceptron initialization"""
    self.numFolds = 10
    self.vocabSize = 0
    self.weights = {}
    self.weightsAvg = {}
    self.X = []
    self.y = []
    self.bias = 0
    self.biasA = 0

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Perceptron classifier 

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    doc = {}
    for word in words:
      if word in doc.keys():
        doc[word]+=1
      else:
        doc[word]=1

    res = 0
    for key,val in doc.items():
      if key in self.weights.keys():
        res+= float(val*self.weights[key])

    if res>0:
      return 'pos'
    else:
      return 'neg'
  

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Perceptron class.
     * Returns nothing
    """
    temp = {}
    for word in words:
      self.vocabSize+=1
      if word not in self.weights.keys():
        self.weights[word]=0
      if word not in temp.keys():
        temp[word]=1
      else:
        temp[word]+=1

    self.X.append(temp)
    if klass =='pos':
      self.y.append(1)
    else:
      self.y.append(-1)

    pass
  
  def dotProduct(self,x):
    prod = 0  
    for key,val in x.items():
      if key in self.weights.keys():
        prod+= self.weights[key] * val
    return prod

  def train(self, split, iterations):
      """
      * TODO 
      * iterates through data examples
      * TODO 
      * use weight averages instead of final iteration weights
      """
      for example in split.train:
          words = example.words
          self.addExample(example.klass, words)

      N = len(self.X)

      for key,val in self.weights.items():
        if key not in self.weightsAvg.keys():
          self.weightsAvg[key]=0

      c = 1

      for i in range(0,iterations):
        combined = list(zip(self.X, self.y))
        shuffle(combined)
        self.X[:], self.y[:] = zip(*combined)
        for n in range(0,N):
          t = self.dotProduct(self.X[n]) + self.bias
          #print(t)
          if((self.y[n]*t)<=0):
            for key,val in self.X[n].items():
              if key in self.weights.keys():
                self.weights[key]= self.weights[key] + (self.y[n]*self.X[n][key])
                self.weightsAvg[key]= self.weightsAvg[key]+ (c* self.y[n]*self.X[n][key])
            self.bias = self.bias + self.y[n]
            self.biasA = self.biasA + (c*self.y[n])
          c+=1

      #num =0
      for key,val in self.weights.items():
        self.weights[key] = float(self.weights[key] - float(self.weightsAvg[key]/c))
        #num+=self.weights[key]

      #den = float(self.bias - float(biasA/c))
      self.bias = float(self.bias - float(self.biasA/c))

      #print(self.weights)
      #return (num/den)

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  

def test10Fold(args):
  pt = Perceptron()
  
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Perceptron()
    accuracy = 0.0
    classifier.train(split,iterations)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print ('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) )
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print ('[INFO]\tAccuracy: %f' % avgAccuracy)
    
    
def classifyDir(trainDir, testDir,iter):
  classifier = Perceptron()
  trainSplit = classifier.trainSplit(trainDir)
  iterations = int(iter)
  classifier.train(trainSplit,iterations)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print ('[INFO]\tAccuracy: %f' % accuracy)
    
def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)

if __name__ == "__main__":
    main()
