#!/usr/bin/env python
# coding: utf-8

import sys
from textblob import TextBlob
from textblob import Word
from collections import defaultdict, Counter
import json

Dict = defaultdict(dict)


def makeDict1():
    file = open('file//tweets.txt','r')
    for line in file:
        file = json.loads(line)
        #读取text
        text = TextBlob(file['text']).words.singularize().lower()
        res = Counter(text)
        #print(res)
        res = sorted(res.items(), key=lambda x: x[0])
        print(type(res[1]))


def makeDict():
    global Dict
    f = open('file/text.txt', 'r')
    x = open('file/word.txt', 'w')
    for line in f:
        word = TextBlob(line).words.singularize()
        word[0] = Word(word[0])
        for i in word[1:]:
            # i=Word(i)
            if i not in Dict:
                Dict[i] = []
                Dict[i].append(word[0])
            else:
                Dict[i].append(word[0])
    for i in Dict:
        Dict[i].sort()
    # print(Dict['may'])
    x.write(str(Dict))


def token(doc):
    doc = doc.lower()
    terms = TextBlob(doc).words.singularize()
    result = []
    for word in terms:
        expected_str = Word(word)
        expected_str = expected_str.lemmatize("v")
        result.append(expected_str)
    print(result[0])
    return result


def main():
    makeDict()



if __name__ == "__main__":
    makeDict1()
