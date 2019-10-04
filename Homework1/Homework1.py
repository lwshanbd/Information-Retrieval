#!/usr/bin/env python
# coding: utf-8

import sys
from textblob import TextBlob
from textblob import Word
from collections import defaultdict

Dict = defaultdict(dict)


def makeDict1():
    global Dict
    '''
    f=open('tweets.txt','r')
    x=open('text.txt','w')
    for i in f:
        #得到text
       # i
        pr1=i.split(', "text": "')
        line=pr1[1].split('", "timeStr"')
        text1=line[0]+"\n"
        #得到id
        pr2 = i.split(', "tweetId": "')
        line = pr2[1].split('", "errorCode": "')
        id = line[0]
        x.write(id+" "+text1.lower())

    f.close()
    x.close()
    '''
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
    #print(Dict['may'])
    x.write(str(Dict))


def And(term1, term2):
    global Dict
    answer = []
    if (term1 not in Dict) or (term2 not in Dict):
        return answer
    else:
        i = len(Dict[term1])
        j = len(Dict[term2])
        x = 0
        y = 0
        l1 = Dict[term1]
        l2 = Dict[term2]
        while x < i and y < j:
            if l1[x] == l2[y]:
                answer.append(l1[x])
                x += 1
                y += 1
            elif l1[x] < l2[y]:
                x += 1
            else:
                y += 1
        return answer


def Or(term1, term2):
    global Dict
    answer = []
    if (term1 not in Dict) or (term2 not in Dict):
        return answer
    else:
        answer = Dict[term1] + Dict[term2]
        return answer


def Not(term1, term2):
    global Dict
    answer = []
    if term1 not in Dict:
        return answer
    elif term2 not in Dict:
        answer = Dict[term1]
        return answer

    else:
        answer = Dict[term1]
        ANS = []
        for ter in answer:
            if ter not in Dict[term2]:
                ANS.append(ter)
        return ANS


def And3(term1, term2, term3):
    Answer = []
    if term3 not in Dict:
        return Answer
    else:
        Answer = And(term1, term2)
        if Answer == []:
            return Answer
        ans = []
        i = len(Answer)
        j = len(Dict[term3])
        x = 0
        y = 0
        while x < i and y < j:
            if Answer[x] == Dict[term3][y]:
                ans.append(Answer[x])
                x += 1
                y += 1
            elif Answer[x] < Dict[term3][y]:
                x += 1
            else:
                y += 1

        return ans


def Or3(term1, term2, term3):
    Answer = Or(term1, term2);
    if term3 not in Dict:
        return Answer
    else:
        if Answer == []:
            Answer = Dict[term3]
        else:
            for item in Dict[term3]:
                if item not in Answer:
                    Answer.append(item)
        return Answer


def AndOr(term1, term2, term3):
    Answer = And(term1, term2)
    if term3 not in Dict:
        return Answer
    else:
        if Answer == []:
            Answer = Dict[term3]
            return Answer
        else:
            for item in Dict[term3]:
                if item not in Answer:
                    Answer.append(item)
            return Answer


def OrAnd(term1, term2, term3):
    Answer = Or(term1, term2)
    if (term3 not in Dict) or (Answer == []):
        return Answer
    else:
        ans = []
        i = len(Answer)
        j = len(Dict[term3])
        x = 0
        y = 0
        while x < i and y < j:
            if Answer[x] == Dict[term3][y]:
                ans.append(Answer[x])
                x += 1
                y += 1
            elif Answer[x] < Dict[term3][y]:
                x += 1
            else:
                y += 1
        return ans


from collections import defaultdict
import operator


def Fre(term):
    global Dict

    data = Dict[term]
    word_frequency = defaultdict(int)
    for i in data:
        word_frequency[i] += 1
    word_sort = sorted(word_frequency.items(), key=operator.itemgetter(1), reverse=True)  # 根据词频降序排序
    print(word_sort)


def Search():
    global Dict
    terms = token(input("Search Query >> "))
    if terms == []:
        sys.exit()

    if len(terms) == 3:
        if terms[1] == "and":
            answer = And(terms[0], terms[2])
            print(answer)
        elif terms[1] == "or":
            answer = Or(terms[0], terms[2])
            print(answer)
        elif terms[1] == "not":
            answer = Not(terms[0], terms[2])
            print(answer)
    elif len(terms) == 5:
        if terms[1] == "and" and terms[3] == "and":
            answer = And3(terms[0], terms[2], terms[4])
            print(answer)
        elif terms[1] == "or" and terms[3] == "or":
            answer = Or3(terms[0], terms[2], terms[4])
            print(answer)
        elif terms[1] == "or" and terms[3] == "and":
            answer = OrAnd(terms[0], terms[2], terms[4])
            print(answer)
        elif terms[1] == "and" and terms[3] == "or":
            answer = AndOr(terms[0], terms[2], terms[5])
            print(answer)
    elif len(terms) == 1:
        Fre(terms[0])


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
    makeDict1()

    while True:
        Search()


if __name__ == "__main__":
    main()
