import pandas as pd



# build 2300 instances of spam and ham
# split 40% spam 60% ham
def buildTrainingData():
    training_spambase = pd.read_csv("spambase.data")
    spamPulls = 2300 * .40
    hamPulls = 2300 * .60
    pullSpams(spamPulls, training_spambase)
    pullHams(hamPulls, training_spambase)

    # shuffle the data.


def buildTestingData():

def pullHams(size, data):
    # finds the size specified but pulls hams

    # randomly pull the specified size of non-spam emails


def pullSpams(size, data):
    # finds the size specified but pulls spams


def main():



main()


