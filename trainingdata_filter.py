#!/usr/bin/python3

# Gerben Timmerman
# Bachelor Scriptie Informatiekunde
# 20-06-2017

import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from os import listdir  # to read files
from os.path import isfile, join  # to read files


def get_filenames_in_folder(folder):
    # return all the filenames in a folder
    return [f for f in listdir(folder) if isfile(join(folder, f))]


def getTokenizedTweets(tweetsFiles):
    listOfTokenizedTweets = []
    limit = 0
    for file in tweetsFiles[0:]:
        with open("trainingdata/" + file, 'r') as twitterData:
            for line in twitterData:
                # Limit the amount of data you want to filter trough
                if limit != -1:
                    dataList = line.strip("\n").split("\t")
                    # Filter Usernames
                    # Filter characters that occur more than 3 times following each other
                    listOfTokenizedTweets.append(TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(dataList[2]))
                    limit += 1
                else:
                    break
    return listOfTokenizedTweets


def getEmoticonTweets(tweetList):
    emoticonList = [":D", ":)", ";)", ";D", ":p", ";p", ":P", ";P", "xD", "XD", "xp", "xP", "Xp", "XP", ":')", ":-)",
                    ":-D", ";-)", ";-D", "=)", "=D", "<3", ] \
                   + [":(", ";(", ":'(", ":-(", "=(", ":'(", ";'(", "D:", ")-:", "):", ");"]
    acronymList = ["lol", "LOL", "haha"]
    filterList = ["RT"]
    positiveWords = ['geweldig', 'geweldige', 'fantastisch', 'fantastische', 'gezellig', 'gezellige', 'lekker', 'lekkere', 'fijn', 'fijne',
                    'leuk', 'leuke', 'aantrekkelijk', 'aantrekkelijke', 'heerlijk', 'heerlijke', 'eerlijk', 'eerlijke', 'enthousiast',
                    'enthousiaste', 'excellent', 'excellente', 'fascinerend', 'fascinerende', 'goed', 'goede', 'handig', 'handige',
                    'indrukwekkend', 'indrukwekkende', 'ideaal', 'ideale', 'kansrijk', 'kansrijke', 'kleurrijk', 'kleurrijke', 'lachen',
                    'lachend', 'lachende', 'optimistisch', 'optimistische', 'perfect', 'perfecte', 'populair', 'populaire', 'positief',
                    'positieve', 'populair', 'populaire', 'sensationeel', 'sensationele', 'slim', 'slimme', 'waanzinnig', 'waanzinnige',
                    'prima', 'mooi', 'mooie', 'lief', 'lieve', 'aardig', 'aardige', 'smakelijk', 'smakelijke', 'trots', 'trotse', 'knap',
                    'knappe', 'geluk', 'gelukkig', 'gelukkige', 'vrede', 'blij', 'uitstekend', 'prachtig', 'prachtige']
    negativeWords = ['hatelijk', 'boos', 'boze', 'guur', 'gure', 'onaangenaam', 'onaangename', 'teleurgesteld', 'teleurgestelde', 'raar',
                    'hopeloos', 'hopeloze', 'bang', 'bange', 'somber', 'laf', 'laffe', 'akelig', 'akelige', 'vreselijk', 'vreselijke',
                    'ongelukkig', 'ongelukkige', 'verdriet', 'verdrietig', 'verdrietige', 'wanhopig', 'wanhopige', 'triest', 'trieste',
                    'rare', 'onaardig', 'onaardige', 'mislukt', 'mislukte', 'verveeld', 'verveelde', 'stom', 'stomme', 'waardeloos',
                    'verschrikkelijk', 'verschrikkelijke', 'zwak', 'zwakke', 'kut', 'kutte', 'slecht', 'slechte', 'jammer', 'haat',
                    'waardeloze', 'onnozel', 'onnozele', 'vreemd', 'vreemde', 'naar', 'nare', 'vervelend', 'vervelende', 'rot', 'rotte',
                    'boos', 'boze', 'gefrustreerd', 'gefrustreerde']
    tweetsWithEmoticons, neutralTweets = [], []
    for tweet in tweetList:
        for word in tweet:
            # Get the relevant tweets with emoticons and acronyms
            if word in emoticonList or word in acronymList:
                # Filter Retweets
                if tweet[0] not in filterList:
                    tweetsWithEmoticons.append(tweet)
                    break
                else:
                    continue
            # Get neutral tweets
            elif word not in positiveWords and word not in negativeWords:
                if len(neutralTweets) < len(tweetsWithEmoticons):
                    neutralTweets.append(tweet)
                    break
            else:
                continue
    return tweetsWithEmoticons, neutralTweets


def preProcessingTweets(emoticonList, neutralTweets):
    fileList = [emoticonList, neutralTweets]
    counter = 0
    for tweetList in fileList:
        for tweet in tweetList:
            # Filter links
            tweetList[counter] = [item for item in tweet if item[0:4] != "http"]
            counter += 1
        counter = 0
    return emoticonList, neutralTweets


def annotateData(tweetList):
    positiveEmoticonList = [":D", ":)", ";)", ";D", ":p", ";p", ":P", ";P", "xD", "XD", "xp", "xP", "Xp", "XP", ":')", ":-)",
                            ":-D", ";-)", ";-D", "=)", "=D", "<3", ]
    negativeEmoticonList = [":(", ";(", ":'(", ":-(", "=(", ":'(", ";'(", "D:", ")-:", "):", ");"]
    positiveAcronymList = ["lol", "LOL", "haha"]
    positiveTweets, negativeTweets = [], []
    for tweet in tweetList:
        if len(positiveTweets) != -1:
            [positiveTweets.append(tweet) for word in tweet if word in positiveEmoticonList or word in positiveAcronymList]
        if len(negativeTweets) != -1:
            [negativeTweets.append(tweet) for word in tweet if word in negativeEmoticonList]
    return positiveTweets, negativeTweets


def postProcessingTweets(positiveTweets, negativeTweets, neutralTweets):
    positiveFile, negativeFile, neutralFile = open("positive.txt", 'a'), open("negative.txt", 'a'), open("neutral.txt", 'a')
    writeFiles = [positiveFile, negativeFile, neutralFile]

    tweets = [positiveTweets, negativeTweets, neutralTweets]
    # Remove stopwords if you want to
    stopwordsPunctuationList = [] # stopwords.words('dutch')
    # Remove punctuation from tweets
    [stopwordsPunctuationList.append(i) for i in string.punctuation]
    # Remove irrelevant words and emoticons
    stopwordsPunctuationList += ["...", ".."] + [":D", ":)", ";)", ";D", ":p", ";p", ":P", ";P", "xD", "XD", "xp", "xP",
                                                 "Xp", "XP", ":')", ":-)", ":-D", ";-)", ";-D", "=)", "=D", "<3", ] \
                                                + [":(", ";(", ":'(", ":-(", "=(", ":'(", ";'(", "D:", ")-:", "):", ");"]
    tweetsetCounter, counter = 0, 0
    for tweetset in tweets:
        for tweet in tweetset:
            # Remove stopwordsPunctuationList features and hashtags from tweets
            tweetset[counter] = [item for item in tweet if item not in stopwordsPunctuationList and item[0] != '#']
            # Remove tweets with 3 words or less
            if len(tweetset[counter]) <= 3:
                tweetset.pop(counter)
                continue
            # Write tweet to file
            writeFiles[tweetsetCounter].write(" ".join(tweetset[counter]) + "\n")
            counter += 1
        counter = 0
        tweetsetCounter += 1


def main():
    # Extract Tweets from files
    print("########### Getting Tweets from files ###########")
    files = get_filenames_in_folder('trainingdata/')
    tweetSplit = getTokenizedTweets(files)
    emoticonTweets, neutralTweets = getEmoticonTweets(tweetSplit)

    # Pre-Processing
    print("########### Pre-Processing ###########")
    preprocessedTweets, neutralTweetsPre = preProcessingTweets(emoticonTweets, neutralTweets)

    # Annotate training data
    print("########### Annotating ###########")
    positiveTweets, negativeTweets = annotateData(preprocessedTweets)

    # Post-Processing
    print("########### Post-Processing ###########")
    postProcessingTweets(positiveTweets, negativeTweets, neutralTweetsPre)

    print("Done!")

main()
