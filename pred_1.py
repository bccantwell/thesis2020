#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:07:14 2019

@author: admin
"""
import pandas as pd
import xml.etree.ElementTree as ET
pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option("display.max_colwidth", 10000)
import numpy as np
np.set_printoptions(threshold=np.inf)

tree = ET.parse('/Users/admin/Downloads/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-edit4.xml')
root = tree.getroot()


df_cols = ["author","text"]
out_df = pd.DataFrame(columns = df_cols)

for message in root.iter('message'):
    text = message.find('text').text
    author = message.find('author').text
    
    out_df = out_df.append(pd.Series([author, text], 
                                     index = df_cols), 
                           ignore_index=True)
#print(out_df.author.unique())

    
df2 = pd.read_csv('/Users/admin/Downloads/pan12-sexual-predator-identification-training-corpus-2012-05-01/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.csv', names=["author"])

#print(df2)

df_merge = pd.concat([out_df, df2])
print(df_merge)