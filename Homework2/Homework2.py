#!/usr/bin/env python
# coding: utf-8

from textblob import TextBlob
from textblob import Word
import math

N = 30307


def getDict():
    global Dict,cos
    f = open('file/word.txt', 'r')
    file = f.read()
    Dict = eval(file)
    Cos = open('file/cosinelog.txt', 'r')
    cos = eval(Cos.read())


def wtq(terms, term):
    global Dict
    num = 0
    for i in terms:
        if i == term:
            num += 1
    idf = math.log10(N / len(Dict[term]))
    wtq =  1 + math.log10(num)
    return idf * wtq


def Search(terms):
    getDict()
    score = {}
    for w in terms:
        Wtq = wtq(terms, w)
        for i in Dict[w]:
            td = int(Dict[w][i])
            wtd = 1 + math.log10(td)
            if i not in score:
                score[i]=wtd*Wtq
            else:
                score[i]+=wtd*Wtq
    for doc in score:
        score[doc]=score[doc]/cos[doc]
    result = sorted(score.items(), key=lambda x: x[1], reverse=True)
    print("tweeetid            评分")
    for i in result[:10]:
        print(str(i[0])+"   "+str(i[1]))


def token(doc):
    doc = doc.lower()
    terms = TextBlob(doc).words.singularize()
    result = []
    for word in terms:
        expected_str = Word(word)
        expected_str = expected_str.lemmatize("v")
        result.append(expected_str)
    return result


def main():
    terms = token(input("Search Query >> "))
    Search(terms)


if __name__ == "__main__":
    main()
