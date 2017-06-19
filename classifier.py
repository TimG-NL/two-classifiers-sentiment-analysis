# Gerben Timmerman
# Bachelor Scriptie Informatiekunde
# 20-06-2017

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from random import shuffle


def read_corpus(files):
    positiveDoc, negativeDoc, neutralDoc = [], [], []
    documents = [positiveDoc, negativeDoc, neutralDoc]
    categories = ['positive', 'negative', 'neutral']
    labelcounter = 0
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                documents[labelcounter].append([tokens, categories[labelcounter]])
        labelcounter += 1
    return documents


def createAbsenceOfPositiveNegativeSentiment(negNeuList, posNeuList):
    lists = [negNeuList, posNeuList]
    positiveAbsence, negativeAbsence = [], []
    counter = 0
    for list in lists:
        for documents in list:
            if counter == 0:
                positiveAbsence.append([documents[0], 'absencePositive'])
            elif counter == 1:
                negativeAbsence.append([documents[0], 'absenceNegative'])
        counter += 1
    shuffle(positiveAbsence), shuffle(negativeAbsence)
    return positiveAbsence, negativeAbsence


def getDocumentsLabels(documentList):
    labels = []
    documents = []
    for document in documentList:
        documents.append(document[0])
        labels.append(document[1])
    return documents, labels


def identity(x):
    # a dummy function that just returns its input
    return x


def trainClassifier(documents, labels):
    #  TF-IDF vectorizer
    tfidf = True
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor=identity,
                              tokenizer=identity)
    else:
        vec = CountVectorizer(preprocessor=identity,
                              tokenizer=identity)
    # combine the vectorizer with the classifier you want. (MultinominalNB(), LinearSVC(), KNeighborsClassifier(), DecisionTreeClassifier())
    classifier = Pipeline([('vec', vec), ('cls', LinearSVC())])
    # train the classifier
    classifier.fit(documents, labels)
    return classifier


def getDevelopmentTestData():
    devTestsetTweets, posListLabels, negListLabels = [], [], []
    with open("developmentTestSet.txt", 'r') as file:
        for line in file:
            devTestsetTweets.append(line.strip().split("\t")[2])
            posListLabels.append(line.strip().split("\t")[3][1])
            negListLabels.append(line.strip().split("\t")[3][4])
    dev_point = int(0.5 * len(devTestsetTweets))
    Xdev, Xtest = devTestsetTweets[:dev_point], devTestsetTweets[dev_point:]
    YdevPosClassifierAnswers = posListLabels[:dev_point]
    YdevNegClassifierAnswers = negListLabels[:dev_point]
    YtestPosClassifierAnswers = posListLabels[dev_point:]
    YtestNegClassifierAnswers = negListLabels[dev_point:]
    return Xdev, Xtest, YdevPosClassifierAnswers, YdevNegClassifierAnswers, YtestPosClassifierAnswers, YtestNegClassifierAnswers


def compareClassifiers(posResults, negResults):
    results, counter = [], 0
    for label in posResults:
        if label == "positive" and negResults[counter] != "negative":
            results.append(["1", "0"])
        elif label != "positive" and negResults[counter] == "negative":
            results.append(["0", "1"])
        elif label == "absencePositive" and negResults[counter] == "absenceNegative":
            results.append(["0", "0"])
        elif label == "positive" and negResults[counter] == "negative":
            results.append(["1", "1"])
        counter += 1
    labelGuessPos = [item[0] for item in results]
    labelGuessNeg = [item[1] for item in results]
    return labelGuessPos, labelGuessNeg


def main():
    filenames = ['positive.txt', 'negative.txt', 'neutral.txt']
    documentsList = read_corpus(filenames)

    posAbsense, negAbsense = createAbsenceOfPositiveNegativeSentiment(documentsList[1] + documentsList[2], documentsList[0] + documentsList[2])

    # Create two combinations of documents and labels for training two seperate classifiers
    # Positive Classifier trained with with an equal amount of positive and absense of positive data
    posNeutralDoc, posNeutralLab = getDocumentsLabels(documentsList[0] + posAbsense[0:len(documentsList[0])])
    positiveClassifier = trainClassifier(posNeutralDoc, posNeutralLab)
    # Negative Classifier trained with with an equal amount of negative and absence of negative data
    negNeutralDoc, negNeutralLab = getDocumentsLabels(documentsList[1] + negAbsense[0:len(documentsList[1])])
    negativeClassifier = trainClassifier(negNeutralDoc, negNeutralLab)

    # Retrieve data for developmentset and testset
    Xdev, Xtest, YdevPosAnswers, YdevNegAnswers, YtestPosAnswers, YtestNegAnswers = getDevelopmentTestData()

    # Predict developmentset
    posYguess = positiveClassifier.predict(Xtest)
    negYguess = negativeClassifier.predict(Xtest)

    # Convert predictions of classifer to right format to check scores
    labelGuessPos, labelGuessNeg = compareClassifiers(posYguess, negYguess)

    # evaluate and compare predictions with ansers of both classifiers
    positiveScore = accuracy_score(YtestPosAnswers, labelGuessPos)
    negativeScore = accuracy_score(YtestNegAnswers, labelGuessNeg)

    print("Positive report: ")
    print(classification_report(YtestPosAnswers, labelGuessPos))
    print("Negative report: ")
    print(classification_report(YtestNegAnswers, labelGuessNeg))
    print("Accuracy: ")
    print(positiveScore, negativeScore)
    print("Overall Accuracy: " + str((positiveScore + negativeScore) / 2))
main()
