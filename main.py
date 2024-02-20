import pandas as pd


class NaiveBayesEmailClassifier:
    def __int__(self, trainingData, testingData):
        self.trainingDataSet = trainingData
        self.testingDataSet = testingData
        # training spambase class priors
        self.trainingSpamClassPrior = self.determineClassPrior(self.trainingDataSet, 1)
        self.trainingHamClassPrior = self.determineClassPrior(self.trainingDataSet, 0)

    #  supply a 0 for ham and 1 for spam
    def determineClassPrior(self, dataSet, predictedClass):
        return (dataSet.iloc[: -1] == predictedClass).sum() / len(dataSet)

    def
    def determineClassFeature

# build 2300 instances of spam and ham
# split 40% spam 60% ham
def buildTrainingTestData(spamSplit, hamSplit):
    spambase = pd.read_csv("spambase.data")

    # get the number of spams and hams required
    spamPulls = int(2300 * spamSplit)
    hamPulls = int(2300 * hamSplit)

    #  shuffle original data
    spambase = spambase.sample(frac=1)


    # Training Data Preparations:
    # =======================================================

    #  pull only the desired number of spams
    trainingSpams = spambase[spambase.iloc[:, -1] == 1][:spamPulls]
    trainingHams = spambase[spambase.iloc[:,-1] == 0][:hamPulls]

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



def main():

    trainingTestingDataList = buildTrainingTestData(.40,.60)

    trainingSpambaseData = trainingTestingDataList[0]
    testingSpambaseData = trainingTestingDataList[1]

    # calculate priors for each class.
    trainingSpamPrior = (trainingSpambaseData.iloc[: -1] == 1).sum() / len(trainingSpambaseData)
    trainingHamPrior = (trainingSpambaseData.iloc[: -1] == 0).sum() / len(trainingSpambaseData)

    # calculates the mean for each feature within the training data
    trainingSpambaseMeans = trainingSpambaseData.loc[:, :, -1].mean()
    trainingSpambaseStds = trainingSpambaseData.loc[:, :, -1].std()

    # replace 0 with a small number for underflow and prevent division by zero
    trainingSpambaseMeans.replace(0, 0.0001)
    trainingSpambaseStds.replace(0, 0.0001)

    testingSpamPrior = (testingSpambaseData.iloc[: -1] == 1).sum() / len(testingSpambaseData)
    testingHamPrior = (testingSpambaseData.iloc[: -1] == 0).sum() / len(testingSpambaseData)

    # calculates the mean std for each feature within the testing data
    testingSpambaseMeans = testingSpambaseData.loc[:, :, -1].mean()
    testingSpambasStds = testingSpambaseData.loc[:, :, -1].std()

    # replace 0 with a small number for underflow and prevent division by zero
    testingSpambaseMeans.replace(0, 0.0001)
    testingSpambasStds.replace(0, 0.0001)










main()


