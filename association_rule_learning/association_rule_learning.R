#install.packages('arules')
#install.packages('arulesViz')
library(tidyverse)
library(arules)
library(arulesViz)

#Qualitiative data for association rules
setwd('data')

path_politics = 'DPI2020.csv'
politics_old = read_csv(path_politics, na = c("", "NA", -999,0))
politics = politics_old %>% 
  filter(year == 2020, ifs != 0) %>% # Only in year 2020
  select_if(negate(is.numeric)) %>% #Drop numeric columns
  select(c(ifs, system, gov1rlc, housesys, state)) %>% #Relevant variables
  rename(Code = ifs) #for merging
#Note: PR = Proportional Representation

path_econ = 'CLASS.xlsx'
economy= readxl::read_excel(path_econ) %>% select(-1)

df = politics %>% left_join(economy, by = "Code") %>% select(-Code)



minConf = 0.3
minSup = 0.2

path = "arm_data.csv"
transactions = read.transactions(path,format = "basket",sep=",")  # read csv basket data
inspect(transactions)

rules = apriori(transactions, parameter = list(support=minSup, confidence=minConf, minlen=2))
inspect(rules)

sortRulesBy = function(metric) {
  sortedRules = sort(rules, by=metric)
  inspect(sortedRules[1:15])
}

sortRulesBy('confidence')
sortRulesBy('support')
sortRulesBy('lift')


#Frequency Plot for all items
itemFrequencyPlot(transactions, topN=10, type="absolute")

#Scatter plot of rules
sortedRules = sort(rules, by="confidence", decreasing=TRUE)
subrules = head(sort(sortedRules, by="lift"),15)
plot(subrules)

#Graph plot
plot(subrules, method="graph", engine="htmlwidget")
