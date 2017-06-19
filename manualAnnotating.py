#!/usr/bin/python3

from os import listdir  # to read files
from os.path import isfile, join  # to read files


def get_filenames_in_folder(folder):
    # return all the filenames in a folder
    return [f for f in listdir(folder) if isfile(join(folder, f))]


def annotateData(files):
    counter, counterpos, counterneg, counterneu, countermix = 0, 0, 0, 0, 0
    for file in files[1:]:
        with open("trainingdata/" + file, 'r') as twitterData:
            print("\n" + file + "\n")
            for line in twitterData:
                print(counter)
                with open("developmentTestSet.txt", 'a') as devFile:
                    if counterneg != 250 and counterpos != 250:
                        dataList = line.strip("\n").split("\t")
                        # Filter Usernames
                        # Filter characters that occur more than 3 times following each other
                        print(dataList[2])
                        getInput = input()
                        if getInput == "p":
                            devFile.write(file + "\t" + str(counter) + "\t" + dataList[2] + "\t" + str([1, 0]) + "\n")
                            counterpos += 1
                        elif getInput == "n":
                            devFile.write(file + "\t" + str(counter) + "\t" + dataList[2] + "\t" + str([0, 1]) + "\n")
                            counterneg += 1
                        elif getInput == "o":
                            devFile.write(file + "\t" + str(counter) + "\t" + dataList[2] + "\t" + str([0, 0]) + "\n")
                            counterneu += 1
                        elif getInput == "m":
                            devFile.write(file + "\t" + str(counter) + "\t" + dataList[2] + "\t" + str([1, 1]) + "\n")
                            countermix += 1
                        else:
                            counter += 1
                            continue
                        counter += 1
                        #print("Positive: " + str(counterpos), "Negative: " + str(counterneg), "neutral: " + str(counterneu), "Mixed: " + str(countermix))
                        #print(sum([counterpos, counterneg, counterneu, countermix]))


def main():
    files = get_filenames_in_folder("trainingdata/")
    annotateData(files)
main()
