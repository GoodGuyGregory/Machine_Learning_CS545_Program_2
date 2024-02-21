import pandas as pd
import numpy as np
import math


class NaiveBayesEmailClassifier:
    def __init__(self, trainingData):
        self.trainingDataSet = trainingData

        # training spambase class priors
        self.trainingSpamClassPrior = self.determineClassPrior(self.trainingDataSet, 1)
        self.trainingHamClassPrior = self.determineClassPrior(self.trainingDataSet, 0)

        # defaults for each spam and ham mean and std
        self.spamsMean = []
        self.spamsStd = []

        self.hamsMean = []
        self.hamsStd = []

        self.determineClassFeatureMeansStds(trainingData)

    #  supply a 0 for ham and 1 for spam
    def determineClassPrior(self, dataSet, targetClass):
        return ((dataSet.iloc[:, -1] == targetClass).sum()) / len(dataSet)

    def determineClassFeatureMeansStds(self, trainingData):

        # calculates the mean for each feature within the training data
        foundSpams = trainingData[trainingData.iloc[:, -1] == 1]
        foundHam = trainingData[trainingData.iloc[:, -1] == 0]

        #  pull the std and mean for each class respectively.
        foundSpamsMean = foundSpams.iloc[:, :-1].mean(axis=0)
        foundSpamsStd = foundSpams.iloc[:, :-1].std(axis=0)

        foundHamsMean = foundHam.iloc[:, :-1].mean(axis=0)
        foundHamsStd = foundHam.iloc[:, :-1].std(axis=0)

        # replace 0 with a small number for underflow and prevent division by zero
        foundSpamsMean.replace(0, 0.0001)
        foundSpamsStd.replace(0, 0.0001)


        foundHamsMean.replace(0, 0.0001)
        foundHamsStd.replace(0, 0.0001)

        self.spamsMean = np.array(foundSpamsMean.to_list())
        self.spamsStd = np.array(foundSpamsStd.to_list())

        self.hamsMean = np.array(foundHamsMean.to_list())
        self.hamsStd = np.array(foundHamsStd.to_list())

    def classifyEmail(self, emailFeatures):
        # adds a posteriors list
        posteriors = []

        # classify ham
        hamPosterior = np.sum(np.log(self.gaussianNB(emailFeatures, self.hamsMean, self.hamsStd))) + self.trainingHamClassPrior
        posteriors.append(hamPosterior)

        # classify spam
        spamPosterior = np.sum(np.log(self.gaussianNB(emailFeatures, self.spamsMean, self.spamsStd))) + self.trainingSpamClassPrior
        posteriors.append(spamPosterior)


        return np.argmax(posteriors, axis=0)

    def coefficientEFraction(self, classStd):
        return 1 / math.sqrt(2 * math.pi * classStd)

    def exponentialEFractionTerm(self, featureVector, classMean, classStd):
        classMean = np.exp(classMean)
        classStd = np.exp(classStd)
        return ((featureVector - classMean) ** 2) / (2 * (classStd ** 2))

    def gaussianNB(self, emailFeatures, classMean, classStd):
        return self.coefficientEFraction(classStd) * (math.e ** (-1 * self.exponentialEFractionTerm(emailFeatures, classMean, classStd)))

# build 2300 instances of spam and ham
# split 40% spam 60% ham
def buildTrainingTestData(spamSplit, hamSplit):
    spambase = pd.read_csv("spambase.data", header=None)

    # get the number of spams and hams required
    spamPulls = int(2300 * spamSplit)
    hamPulls = int(2300 * hamSplit)

    #  shuffle original data
    spambase = spambase.sample(frac=1)


    # Training Data Preparations:
    # =======================================================

    #  pull only the desired number of spams
    trainingSpams = spambase[spambase.iloc[:, -1] == 1][:spamPulls]
    trainingHams = spambase[spambase.iloc[:, -1] == 0][:hamPulls]

    # removes data so test can't pull it into its own DF
    #  this is done with a boolean mask to remove the corresponding index values from the original df
    spambase = spambase[~spambase.index.isin(trainingSpams.index)]
    spambase = spambase[~spambase.index.isin(trainingHams.index)]


    training_spambase = pd.concat([trainingHams, trainingSpams])

    #  Testing Data
    # ==============================================================

    #  pull only the desired number of spams
    testingSpams = spambase[spambase.iloc[:, -1] == 1][:spamPulls]
    testingHams = spambase[spambase.iloc[:, -1] == 0][:hamPulls]

    # removes data so test can't pull it into its own DF
    # ~ negates the boolean mask to select the rows that are not present in spambase and filter them
    spambase = spambase[~spambase.index.isin(testingSpams.index)]
    spambase = spambase[~spambase.index.isin(testingHams.index)]

    testing_spambase = pd.concat([testingHams, testingSpams])
    # return both dataframes
    trainingTestingDfs = []

    trainingTestingDfs.append(training_spambase)
    trainingTestingDfs.append(testing_spambase)

    # return the data from split
    return trainingTestingDfs

def perpareTestData(testingData):
    # split the testing features from the label.

    # pull classifications and for comparison.



def main():

    trainingTestingDataList = buildTrainingTestData(.40,.60)

    trainingSpambaseData = trainingTestingDataList[0]
    testingSpambaseData = trainingTestingDataList[1]

    emailClassifier = NaiveBayesEmailClassifier(trainingSpambaseData)


    #  split target and features for testing.

    sanitizeTestData(testingSpambaseData)












main()


