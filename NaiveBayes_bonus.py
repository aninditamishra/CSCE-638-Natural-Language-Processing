import sys
import getopt
import os
import math
import operator

class NaiveBayes:
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
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.BOOLEAN_NB = False
    self.stopList = set(self.readFile('../data/english.stop'))
    self.numFolds = 10
    #Adding variables for Vocabulary, dictionary for positive words, dictionary for negative words
    self.vocab = set()
    self.posDict = {}
    self.negDict = {}
    #Number of Positive Docs
    self.posDocCount = 0
    #Number of Negative Docs
    self.negDocCount = 0

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts.
  #
  #
  # If any one of the FILTER_STOP_WORDS and BOOLEAN_NB flags is on, the
  # other one is meant to be off.

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    bigramWords = []
    for i in range(0,len(words)-1):
        bigram = words[i]+'_'+words[i+1]
        if bigram not in bigramWords:
          bigramWords.append(bigram)

    words = words + bigramWords

    if self.FILTER_STOP_WORDS:
      words =  self.filterStopWords(words)

    if self.BOOLEAN_NB:
      wordSet = set()

      for word in words:
        wordSet.add(word)

      posProb = (self.posDocCount/(self.posDocCount+self.negDocCount))
      l_posProb = math.log10(posProb)

      pos_den =0
      for k,v in self.posDict.items():
        pos_den+=v

      pos_den+=len(self.vocab)+1
      l_posden = math.log10(pos_den)

      l_posnum =0

      for word in wordSet:
        if word in self.posDict.keys():
          l_posnum+=math.log10(self.posDict[word]+1)-l_posden
        else:
          l_posnum-=l_posden

      pos_calcProb = l_posProb + l_posnum

      negProb = (self.negDocCount/(self.posDocCount+self.negDocCount))
      l_negProb = math.log10(negProb)

      neg_den = 0
      for k,v in self.negDict.items():
        neg_den+=v

      neg_den+=len(self.vocab)+1

      l_negden = math.log10(neg_den)

      l_negnum = 0

      for word in wordSet:
        if word in self.negDict.keys(): 
          l_negnum+=math.log10(self.negDict[word]+1) - l_negden
        else:
          l_negnum -= l_negden


      neg_calcProb = l_negProb + l_negnum

    else:
      #Positive Class robability
      #Probability of P(c=Positive)
      posProb = (self.posDocCount/(self.posDocCount+self.negDocCount))
      l_posProb = math.log10(posProb)


      pos_den = 0
      for k,v in self.posDict.items():
        pos_den+=v

      pos_den+=len(self.vocab)+1

      l_posden = math.log10(pos_den)

      l_posnum=0

      for word in words:
        if word in self.posDict.keys():
          l_posnum+=math.log10(self.posDict[word]+1) - l_posden
        else:
          l_posnum -= l_posden

      pos_calcProb = l_posProb + l_posnum 

      #Negative Class Probability
      #Probability of P(c=Negative)
      negProb = (self.negDocCount/(self.posDocCount+self.negDocCount))
      l_negProb = math.log10(negProb)

      neg_den = 0
      for k,v in self.negDict.items():
        neg_den+=v

      neg_den+=len(self.vocab)+1

      l_negden = math.log10(neg_den)

      l_negnum = 0

      #account for unigrams
      for word in words:
        if word in self.negDict.keys(): 
          l_negnum+=math.log10(self.negDict[word]+1) - l_negden
        else:
          l_negnum -= l_negden

      neg_calcProb = l_negProb + l_negnum 

    if(neg_calcProb < pos_calcProb):
      return 'pos'
    else:
      return 'neg'
  

  def addExample(self, klass, words):

    bigramWords = []
    for i in range(0,len(words)-1):
        bigram = words[i]+'_'+words[i+1]
        if bigram not in bigramWords:
          bigramWords.append(bigram)

    words = words + bigramWords
    
    if self.BOOLEAN_NB:
      temp = {}
      for word in words:
        self.vocab.add(word)
        
        if word not in temp.keys():
          temp[word]=1
      
      if klass == 'pos':
        self.posDocCount+=1
        for k,v in temp.items():
          if k not in self.posDict.keys():
            self.posDict[k]=1
          else:
            self.posDict[k]+=1
      else:
        self.negDocCount+=1
        for k,v in temp.items():
          if k not in self.negDict.keys():
            self.negDict[k]=1
          else:
            self.negDict[k]+=1
    else:
      if klass == 'pos':
        self.posDocCount+=1
        #unigrams
        for word in words:
          self.vocab.add(word)
          if word not in self.posDict.keys():
            self.posDict[word]=1
          else:
            self.posDict[word]+=1
      else:
        self.negDocCount+=1
        for word in words:
          self.vocab.add(word)
          if word not in self.negDict.keys():
            self.negDict[word]=1
          else:
            self.negDict[word]+=1
    pass
      

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

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)


  def crossValidationSplits(self, trainDir):
    """Returns a list of TrainSplits corresponding to the cross validation splits."""
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
  
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB):
  nb = NaiveBayes()
  splits = nb.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    accuracy = 0.0
    for example in split.train:
      words = example.words
      classifier.addExample(example.klass, words)
  
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
    
    
def classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB, trainDir, testDir):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  classifier.BOOLEAN_NB = BOOLEAN_NB
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit)
  testSplit = classifier.trainSplit(testDir)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print ('[INFO]\tAccuracy: %f' % accuracy)


def main():
  FILTER_STOP_WORDS = False
  BOOLEAN_NB = False
  (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
  if ('-f','') in options:
    FILTER_STOP_WORDS = True
  elif ('-b','') in options:
    BOOLEAN_NB = True
  
  if len(args) == 2:
    classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB,  args[0], args[1])
  elif len(args) == 1:
    test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB)

if __name__ == "__main__":
    main()
