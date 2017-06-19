from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
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
    shuffle(positiveDoc), shuffle(negativeDoc), shuffle(neutralDoc)
    return documents


def getFormatTrainingData(documentList):
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
    # combine the vectorizer with the classifier you want. (MultinominalNB(), LinearSVC(), )
    classifier = Pipeline([('vec', vec), ('cls', LinearSVC())])

    # train the classifier
    classifier.fit(documents, labels)
    return classifier


def getDevelopmentTestData():
    devTestDocuments, devTestLabels = [], []
    with open("developmentTestSet.txt", 'r') as file:
        for line in file:
            if line.strip().split("\t")[3][1] == '1' and line.strip().split("\t")[3][4] == '0':
                devTestLabels.append("positive")
            elif line.strip().split("\t")[3][1] == '0' and line.strip().split("\t")[3][4] == '1':
                devTestLabels.append("negative")
            elif line.strip().split("\t")[3][1] == '0' and line.strip().split("\t")[3][4] == '0':
                devTestLabels.append("neutral")
            elif line.strip().split("\t")[3][1] == '1' and line.strip().split("\t")[3][4] == '1':
                continue
            devTestDocuments.append(line.strip().split("\t")[2])
    dev_point = int(0.5 * len(devTestDocuments))
    Xdev, Xtest = devTestDocuments[:dev_point], devTestDocuments[dev_point:]
    Ydev = devTestLabels[:dev_point]
    Ytest = devTestLabels[dev_point:]
    return Xdev, Xtest, Ydev, Ytest


def main():
    filenames = ['positive.txt', 'negative.txt', 'neutral.txt']
    # Extract trainingdata from files
    data = read_corpus(filenames)

    # Get Correct format for trainingdata with documents and corresponding labels
    documents, labels = getFormatTrainingData(data[0][0:len(data[1])] + data[1] + data[2][0:len(data[1])])

    # Train the Classifier
    classifier = trainClassifier(documents, labels)

    XdevDoc, XtestDoc, YdevLab, YtestLab = getDevelopmentTestData()

    Yguess = classifier.predict(XdevDoc)

    accResult = accuracy_score(YdevLab, Yguess)
    print(classification_report(YdevLab, Yguess))
    print(accResult)

main()
