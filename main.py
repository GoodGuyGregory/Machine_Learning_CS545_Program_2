import pandas as pd



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


main()


