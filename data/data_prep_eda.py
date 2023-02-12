import requests
import pandas as pd
import wbgapi as wb
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import json_normalize 

import wbgapi as wb #import World Bank API package


wb_data = wb.data.DataFrame(["SL.UEM.TOTL.ZS","SM.POP.REFG.OR","BX.KLT.DINV.CD.WD","SP.POP.TOTL","SP.DYN.IMRT.IN","NY.GDP.MKTP.CD"], time = 2020, labels=True)

col_names = {'SL.UEM.TOTL.ZS': 'Unemployment Rate (%)',
        'SM.POP.REFG.OR': 'Refugee Population',
        'SP.POP.TOTL': 'Population',
        'NY.GDP.MKTP.CD': 'GDP',
        'SP.DYN.IMRT.IN': 'Infant Mortality Rate',
        'BX.KLT.DINV.CD.WD': 'Net FDI'}
wb_data.rename(columns=col_names,
          inplace=True)

#Other data

freedom = pd.read_excel("/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project/data/Country_and_Territory_Ratings_and_Statuses_FIW_1973-2022 .xlsx", 
                           skiprows=2, sheet_name=1, na_values = '-', usecols = 'A, EM:EO')

freedom_cols = ['Country', 'FH Political Rights Score', 'FH Civil Liberties Score', 'FH Status']
freedom.columns = freedom_cols

corruption = pd.read_excel('/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project/data/CPI 2021 Full Data Set/CPI2021_GlobalResults&Trends.xlsx',
                          skiprows = 2, usecols ='A,B,D')

aid = pd.read_csv('/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project/data/us_foreign_aid_country.csv')
                  
aid.dtypes
# Drop trailing letters from Fiscal Year column and convert to integer
aid['Fiscal Year'] = aid['Fiscal Year'].str.replace(r'\D+', '', regex=True).astype('int')

# Only include 2020 data and drop unneccessary columns
aid_new = aid[aid['Fiscal Year'] == 2020].pivot(index='Country Code', columns='Transaction Type Name', values='current_amount')
aid_new = aid_new.drop(aid_new.columns[[0, 1, 3]], axis = 1)

#merge datasets
df = corruption.merge(freedom, left_on = corruption.columns[0], right_on = freedom.columns[0]).merge(wb_data, left_on = 'ISO3', right_index = True).merge(aid_new, left_on='ISO3', right_index = True)

# Drop extra country name columns
df.drop(df.columns[[3, 7]], axis=1, inplace = True)

# Create new 'Aid Level' column
def applyFunc(s):
    if s <= 0:
        return 'No Aid'
    elif s < df['Obligations'].median():
        return 'Low Aid'
    else:
        return 'High Aid'
df['Aid Level'] = df['Obligations'].apply(applyFunc)

#Data visualization
for i, col in enumerate(df.columns):
    if df.dtypes[i] != 'O' and i <  12:
        if(2 < i < 5):
            sns.displot(data = df, x = col, hue = 'Aid Level', multiple = 'stack')
        else : 
            sns.displot(data = df, x = col, hue = 'Aid Level', kind = 'kde')

sns.catplot(data = df, x = 'Aid Level', y = 'Obligations', kind = 'box', showfliers = False) #boxplots