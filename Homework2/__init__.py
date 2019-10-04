import json
from collections import Counter
import collections
import math

Index={}          #    总索引
Paper_number=0    #标记文章总数量
Word_frequency={} #“有该词”文章的数量

for i in open('tweets.txt'):
    Paper_number=Paper_number+1
    index={}      #针对每个tweets单独的索引
    dict = json.loads(i)
    array_text=(dict['text']).lower().split(" ")
    array_username=dict['userName'].lower().split(" ")
    array=array_text+array_username
    res = Counter(array)
    print(res)
    res = sorted(res.items(), key=lambda x: x[0])
    print(res)
    indexed={}
    for i in res:
        if(i[0] not in Word_frequency.keys()):
            Word_frequency[i[0]]=0
        Word_frequency[i[0]]=Word_frequency[i[0]]+1
        indexed[i[0]]=math.log(1+i[1], 10)#计算tft
    Index[dict["tweetId"]]=indexed
    print("-------------------------")
