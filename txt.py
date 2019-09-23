f=open('a.txt','r')

line=f.readline()

for i in line:
    i.replace(',','\n')
