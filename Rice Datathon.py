#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import os
import numpy as np
import collections
import random

os.getcwd()

default = collections.defaultdict(int)
default[0] += 5
print(default)
print(default.keys())





# 0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
#
# 1 - the id of the tweet (2087)
#
# 2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
#
# 3 - the query (lyx). If there is no query, then this value is NO_QUERY.
#
# 4 - the user that tweeted (robotickilldozr)
#
# 5 - the text of the tweet (Lyx is cool)


headers = [
    "polarity",
    "id",
    "date",
    "query",
    "user",
    "tweet"
]


testing_data = pd.read_csv("/Users/benitogeordie/Downloads/trainingandtestdata/testdata.manual.2009.06.14.csv",
    names = headers,
    encoding = "latin1"
)

training_data = pd.read_csv("/Users/benitogeordie/Downloads/trainingandtestdata/training.1600000.processed.noemoticon.csv",
    names = headers,
    encoding = "latin1"
)




def lowercase(tweetslist):
    lowertweets = []
    copytweets = list(tweetslist)
    for i in range(len(tweetslist)):
        lowertweets.append(copytweets[i].lower())
    return lowertweets


training_data.head(10)


training_data['tweet']


tweetstrain = training_data['tweet']
tweetstest = testing_data['tweet']


tweetslisttrain = tweetstrain.tolist()
tweetslisttest = tweetstest.tolist()


polaritytrain = training_data['polarity']
polaritytest = testing_data['polarity']


polaritylisttrain = polaritytrain.tolist()
polaritylisttest = polaritytest.tolist()


training1 = list(zip(lowercase(tweetslisttrain), polaritylisttrain))
testing1 = list(zip(lowercase(tweetslisttest), polaritylisttest))


def random_sample(lst, perc):
    sample = []
    lst_len = len(lst)
    i = 0
    while i < lst_len*perc:
        sample.append(random.choice(lst))
        i += 1
    return sample


def weighting(dataset, fillers):
    b = 0
    g = 0
    d1 = collections.defaultdict(lambda : [0,0])
    for pair in dataset:
        words = pair[0].split()
        if pair[1] == 0:
            b += 1
            for word in words:
                d1[word][0] += 1
        elif pair[1] == 4:
            g += 1
            for word in words:
                d1[word][1] += 1
    d2 = collections.defaultdict(lambda : 0)
    for key in d1:
        d2[key] = (float(d1[key][1])/g-float(d1[key][0])/b)
    for key in d2:
        if key in fillers:
            d2[key] = 0
    return d2


def determine(tweet, weights):
    tweetscore = 0
    for word in tweet[0].split():
        tweetscore += weights[word]
    return tweetscore

def weighting2(dataset, fillers):
    gooddict = collections.defaultdict(int)
    baddict = collections.defaultdict(int)
    for pair in dataset:
        if pair[1] == 0:
            for word in pair[0].split():
                if word not in fillers:
                    if len(word) > 3:
                        baddict[word] -= 1
        if pair[1] == 4:
            for word in pair[0].split():
                if word not in fillers:
                    if len(word) > 3:
                        gooddict[word] += 1
    return gooddict, baddict

def weighting3(dataset, fillers):
    b = 0
    g = 0
    d1 = collections.defaultdict(lambda : [0,0])
    for pair in dataset:
        words = pair[0].split()
        if pair[1] == 0:
            b += 1
            for word in words:
                d1[word][0] += 1
        elif pair[1] == 4:
            g += 1
            for word in words:
                d1[word][1] += 1
    d2 = collections.defaultdict(lambda : 0)
    for key in d1:
        d2[key] = ((float(d1[key][1])**2)/(g*(float(d1[key][1])+float(d1[key][0])))-(float(d1[key][0])**2)/(b*(float(d1[key][1])+float(d1[key][0]))))
    for key in d2:
        if key in fillers:
            d2[key] = 0
    return d2

def determine2(newtweet,gooddict,baddict):
    goodscore = 0
    badscore = 0
    for word in newtweet[0].split():
        if word in gooddict:
            goodscore += gooddict[word]
        if word in baddict:
            badscore += baddict[word]

    score = goodscore / float(len(gooddict)) + badscore / float(len(baddict))
    return score

def make_training_test(dataset, perc):
    training = random_sample(dataset,perc)
    test = list(set(dataset).difference(set(training)))
    return training, test

def experiment1(weighting_f, determine_f, training, test, fillers):
    """
    :param weighting_f: weighting function
    :param determine_f: determine function
    :param dataset: list of pairs
    :param perc: percentage of dataset
    to be used as training data (the
    rest is used for testing)
    :return: percentage of correct predictions
    """
    correct = 0
    incorrect = 0
    weights = weighting_f(training, fillers)
    for pair in test:
        if pair[1] != 2:
            if determine_f(pair, weights) > 0 and pair[1] == 4:
                #print("dis right")
                correct += 1
            elif determine_f(pair, weights) <= 0 and pair[1] == 0:
                #print("dis right")
                correct += 1
            else:
                #print("dis wrong")
                incorrect += 1
    accuracy = float(correct)/(correct+incorrect)
    return accuracy

def experiment2(weighting_f, determine_f, training, test, fillers):
    """
    :param weighting_f: weighting function
    :param determine_f: determine function
    :param dataset: list of pairs
    :param perc: percentage of dataset
    to be used as training data (the
    rest is used for testing)
    :return: percentage of correct predictions
    """
    correct = 0
    incorrect = 0
    goodweights,badweights = weighting_f(training, fillers)
    for pair in test:
        if pair[1] != 2:
            if determine_f(pair, goodweights,badweights) > 0 and pair[1] == 4:
                #print("dis right")
                correct += 1
            elif determine_f(pair, goodweights,badweights) <= 0 and pair[1] == 0:
                #print("dis right")
                correct += 1
            else:
                #print("dis wrong")
                incorrect += 1
    accuracy = float(correct)/(correct+incorrect)
    return accuracy

def filler_sets(dataset):
    """
    intersection of (intersection of good and bad) and neutral
    :param dataset: pairs of tweet, polarity
    :return: filler words
    """
    good = set([])
    bad = set([])
    neutral = set([])
    for pair in dataset:
        if pair[1] == 0:
            for word in pair[0].split():
                bad.add(word)
        if pair[1] == 2:
            for word in pair[0].split():
                neutral.add(word)
        if pair[1] == 4:
            for word in pair[0].split():
                good.add(word)
    #print("good",good)
    #print("bad",bad)
    #print("neutral",neutral)
    fillers = neutral.intersection(good.intersection(bad))
    return fillers

fillers1 = filler_sets(training1)
print(fillers1)
training2,testing2 = make_training_test(testing1, 0.5)
fillers3 = set(["for","and","so","the","nor","if","but","that","how","or","as","yet","a","of"])
fillers2 = filler_sets(training2).union(["for","and","so","the","nor","if","but","that","how","or","as","yet","a","of"])
print("experiment1", experiment1(weighting, determine, training2, testing2, fillers2))
print("experiment2", experiment2(weighting2, determine2, training2, testing2, fillers2))
print("experiment3", experiment1(weighting3, determine, training2, testing2, fillers2))

weights1 = weighting(training1,fillers3)
dictionary = collections.defaultdict(lambda: [])
for item in weights1:
    dictionary[weights1[item]].append(item)

akeys = list(dictionary.keys())
akeys.sort()

print("worst:")
i=0
while i<10:
    for word in dictionary[akeys[i]]:
        print(word)
        i+=1
print()
print ("best:")
j=-1
while j>-10:
    for word in dictionary[akeys[j]]:
        print(word)
        j-=1
