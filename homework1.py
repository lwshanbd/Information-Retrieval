#", "timeStr"
import os
f=open('tweets.txt','r')
x=open('text.txt','w')
for i in f:
    #得到text
    pr1=i.split(', "text": "')
    line=pr1[1].split('", "timeStr"')
    text1=line[0]+"\n"
    #得到id
    pr2 = i.split(', "tweetId": "')
    line = pr2[1].split('", "errorCode": "')
    id = line[0]

    #print(id+" "+text1.replace('#','').replace('@',''))
    x.write(id+" "+text1.replace('#','').replace('@','').replace('?','').replace('!','').lower())

f.close()
x.close()

f=open('text.txt','r')
dict=dict()

for line in f:
    word=line.split()
    for i in word[1:]:
        if i not in dict:
            dict[i] = []
            dict[i].append(word[0])
        else:
            dict[i].append(word[0])
a=open('dict.txt','w')
a.write(str(dict))
print(dict)

