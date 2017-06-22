# -*- coding: utf-8 -*-
import sys

print (sys.stdout.encoding)
print (u"Stöcker".encode(sys.stdout.encoding, errors='replace'))
print (u"Стоескер".encode(sys.stdout.encoding, errors='replace'))


categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
print(twenty_train.target_names)
print (type(twenty_train.data))
f = open("twenty_train.txt",'w')
for item in twenty_train.data:
  f.write("%s\n" % item)
f.close()
print(twenty_train.data)