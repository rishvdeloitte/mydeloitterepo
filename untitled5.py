# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:17:17 2023
'''i know i am right'''
@author: rraychoudhury
"""

import pandas as pd

df=pd.read_csv("C:\\Users\\RRAYCHOUDHURY\\Documents\\Practice Case Studies\\CS1_BasicDataManipulation\\book1.csv")
print(df['sal'].value_counts())
print(df.groupby('gen')['id'].count())
print(df.groupby('city')['sal'].sum().unique())
print(df['city'].unique())
print(df.groupby('city')['sal'].max())
print(df.groupby('gen')['sal'].sum())
'''
dict={'name':['raj','ravi','mansi','apruv'],'game':[4,5,6,7],'gamename':['checkers','snake','chess','new'],'score':[10,20,10,10,]}
df1=pd.DataFrame(dict)
print(df1)
'''
df3=pd.DataFrame({'id': [11,12,13,14,15],'name':['Kunal', 'Shriya', 'Ram', 'Nisha', 'Vinit'],"Maths": [90,97,88,90,99],
'Chemistry': [89,80,78,66,67], "Physics":[50,56,88,80,75],"Bio":[72,87,56,67,82],"Eng":[79,97,78,90,89],})
'''
print(df3["Maths"].unique())
print(df3.columns)
print(df3.min())
print(df3['Maths'].min())
print(df3.groupby('Maths'))
'''
total=[0,0,0,0,0]
df3['total']=total

df3['total']=df3.iloc[:,2:7].sum(axis=1)
print(df3)
