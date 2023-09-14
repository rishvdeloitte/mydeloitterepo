# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:11:17 2023

@author: rraychoudhury
"""

file=open('C:\\Users\\RRAYCHOUDHURY\\Desktop\\wedenesday.txt')
for line in file:
    print(line)



file2=open('C:\\Users\\RRAYCHOUDHURY\\Desktop\\ml.txt','w')
file2.write('yes i am a good machine learning engineer')
file2.write('i am good')

file3=open('C:\\Users\\RRAYCHOUDHURY\\Desktop\\ml.txt','a')
file3.write('i am a good man')