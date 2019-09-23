#!/usr/bin/env python
# coding: utf-8

# In[8]:


import sys
from textblob import TextBlob
from textblob import Word
from collections import defaultdict
postings = defaultdict(dict)


# In[9]:


def makeDict():
    global postings
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
    f=open('text.txt','r')

    x=open('word.txt','w')
    dict1=dict()

    for line in f:
        word=TextBlob(line).words.singularize()
        word[0]=Word(word[0])
        for i in word[1:]:
            #i=Word(i)
            if i not in dict1:
                dict1[i] = []
                dict1[i].append(word[0])
            else:
                dict1[i].append(word[0])
    postings=dict1
    for i in dict1:
        dict1[i].sort()
    #print(postings['may'])
    x.write(str(postings))


# In[10]:


def And(term1, term2):
    global postings
    answer = []
    if (term1 not in postings) or (term2 not in postings):
        return answer
    else:
        i = len(postings[term1])
        j = len(postings[term2])
        x = 0
        y = 0
        l1=postings[term1]
        l2=postings[term2]
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
    global postings
    answer = []
    if (term1 not in postings) or (term2 not in postings):
        return answer
    else:
        answer=postings[term1]+postings[term2]
        return answer

def Not(term1, term2):
    global postings
    answer = []
    if term1 not in postings:
        return answer
    elif term2 not in postings:
        answer = postings[term1]
        return answer

    else:
        answer = postings[term1]
        ANS = []
        for ter in answer:
            if ter not in postings[term2]:
                ANS.append(ter)
        return ANS####
from collections import defaultdict
import operator
def Fre(term):
    global postings

    data = postings[term]
    word_frequency = defaultdict(int)
    for i in data:
        word_frequency[i] += 1
    word_sort = sorted(word_frequency.items(), key=operator.itemgetter(1), reverse=True)  # 根据词频降序排序
    print(word_sort)

# In[11]:


def do_search():
    global postings
    terms = token(input("Search query >> "))
    if terms == []:
        sys.exit()
        # 搜索的结果答案

    if len(terms) == 3:
        # A and B
        if terms[1] == "and":
            answer = And(terms[0], terms[2])
            print(answer)
        elif terms[1] == "or":
            answer = Or(terms[0], terms[2])
            print(answer)
        elif terms[1] == "not":
            answer = Not(terms[0], terms[2])
            print(answer)
    if len(terms) == 1:
        Fre(terms[0])
'''
    else:
        leng = len(terms)
        answer = do_rankSearch(terms)
        print("[Rank_Score: Tweetid]")
        for (tweetid, score) in answer:
            print(str(score / leng) + ": " + tweetid)

'''


# In[12]:


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


# In[ ]:


def main():
    makeDict()
    while True:
        do_search()


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




