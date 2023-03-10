import pandas as pd
import wbgapi as wb
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas.api.types import is_numeric_dtype
import wbgapi as wb #import World Bank API package

os.chdir('/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project/data')
os.getcwd()

#Quantitative Data
## World Bank API Data
wb_data = wb.data.DataFrame(["SL.UEM.TOTL.ZS","SM.POP.REFG.OR","BX.KLT.DINV.CD.WD","SP.POP.TOTL","SP.DYN.IMRT.IN","NY.GDP.MKTP.CD"], time = 2020, labels=True)

col_names = {'SL.UEM.TOTL.ZS': 'Unemployment Rate (%)',
        'SM.POP.REFG.OR': 'Refugee Population',
        'SP.POP.TOTL': 'Population',
        'NY.GDP.MKTP.CD': 'GDP',
        'SP.DYN.IMRT.IN': 'Infant Mortality Rate',
        'BX.KLT.DINV.CD.WD': 'Net FDI'}
wb_data.rename(columns=col_names,
          inplace=True)

## Other data
freedom = pd.read_excel("imports/Country_and_Territory_Ratings_and_Statuses_FIW_1973-2022 .xlsx", 
                           skiprows=2, sheet_name=1, na_values = '-', usecols = 'A, EM:EO')

freedom_cols = ['Country', 'FH Political Rights Score', 'FH Civil Liberties Score', 'FH Status']
freedom.columns = freedom_cols

corruption = pd.read_excel('imports/CPI2021_GlobalResults&Trends.xlsx',
                          skiprows = 2, usecols ='A,B,D')

aid = pd.read_csv('imports/us_foreign_aid_country.csv')
                  
aid.dtypes
# Drop trailing letters from Fiscal Year column and convert to integer
aid['Fiscal Year'] = aid['Fiscal Year'].str.replace(r'\D+', '', regex=True).astype('int')

# Only include 2020 data and drop unneccessary columns
aid_new = aid[aid['Fiscal Year'] == 2020].pivot(index='Country Code', columns='Transaction Type Name', values='current_amount')
aid_new = aid_new.drop(aid_new.columns[[0, 1, 3]], axis = 1)
aid_new['Obligations'].fillna(0, inplace = True) # replace na with 0
aid_new['Obligations'] = abs(aid_new['Obligations']) # fix negative values

#merge datasets
quant = corruption.merge(freedom, left_on = corruption.columns[0], right_on = freedom.columns[0]).merge(wb_data, left_on = 'ISO3', right_index = True).merge(aid_new, left_on='ISO3', right_index = True)

# Drop extra country name columns
quant.drop(quant.columns[[3, 7]], axis=1, inplace = True)

#replace na with group means

# Create new 'Aid Level' column
def aidLevel(s):
    if s < quant['Obligations'].median():
        return 'Low Aid'
    # elif s > quant['Obligations'].quantile(0.67):
    #     return 'High Aid'
    else:
        return 'High Aid'
quant['Aid Level'] = quant['Obligations'].apply(aidLevel)


#Data visualization
for i, col in enumerate(quant.columns):
    if quant.dtypes[i] != 'O' and i <  12:

        if(2 < i < 5):
            sns.displot(data = quant, x = col, hue = 'Aid Level', multiple = 'stack')
        else : 
            sns.displot(data = quant, x = col, hue = 'Aid Level', kind = 'kde', warn_singular=False)

sns.catplot(data = quant, x = 'Aid Level', y = 'Obligations', kind = 'box', showfliers = False) #boxplots


plt.show()

quant.drop('Obligations', axis =1, inplace = True) # remove obligations variable

#Categorical data

#political data
path_pol = 'imports/DPI2020.csv'
dpi = pd.read_csv(path_pol, na_values = ["", "NA", -999,0],low_memory = False)
dpi.query("year == 2020 & ifs.notnull()", inplace=True) # Only select year 2020
dpi = dpi.select_dtypes(exclude=np.number)# only categorical data
dpi = dpi[['ifs', 'system', 'gov1rlc', 'housesys', 'state']] # relevant variables
dpi.rename(columns = {"ifs":'Code'}, inplace = True) # rename iso code for merging

#economic data
path_econ = 'imports/CLASS.xlsx'
econ = pd.read_excel(path_econ)
econ.drop('Economy', inplace = True, axis = 1) # Drop country name

cat = dpi.merge(econ, how='left',on='Code')

df = quant.merge(cat, how='left',left_on ='ISO3', right_on='Code').drop('Code', axis = 1)

#replace NA by aid group
df[['Lending category','Other (EMU or HIPC)']] = df[['Lending category','Other (EMU or HIPC)']].fillna('None')
for i in df.columns:
    if is_numeric_dtype(df[i]):
        df[i].fillna(df.groupby('Aid Level')[i].transform('median'), inplace = True) # replace with group median (quantitative)
    else:
        df[i].fillna(df.groupby('Aid Level')[i].transform(lambda x: x.mode()[0]), inplace = True) # replace with group mode (categorical)

df.to_csv('final_clean_data.csv', index=False)
