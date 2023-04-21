library(tidyverse)

setwd('/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project')
df = read_csv('data/final_clean_data.csv')

aid = read_csv('data/imports/us_foreign_aid_country.csv')

aid = aid %>% 
  filter(`Fiscal Year` == 2020 & `Transaction Type Name` == 'Disbursements') %>% 
  select(`Country Code`, current_amount) %>% 
  mutate(current_amount = abs(current_amount)) %>% 
  rename('ISO3' = 'Country Code')

df = df %>% 
  left_join(aid) %>% 
  mutate(current_amount = ifelse(is.na(current_amount), 0, current_amount),
         log_aid = log(current_amount + 1)) %>% 
  select(-current_amount) %>% 
  column_to_rownames(var = 'ISO3') %>% 
  select_if(is.numeric)

mod1 = lm(log_aid ~ ., data = df)
summary(mod1) # Select column with lowest p-value

df_slr = df %>% select(log_aid, `CPI score 2021`)

write.csv(df_slr, 'linear_regression/slr_data.csv')

mod = lm(log_aid ~ `CPI score 2021`, data = df)


plt = ggplot(df,aes(`CPI score 2021`, log_aid)) + 
  geom_point() +
  stat_smooth(method = "lm",formula = y ~ x,geom = "smooth") + 
  labs(title = 'Scatter plot with linear regression line',
       y = "Log of Aid Disbursements")
plt
ggsave('linear_regression/imgs/scatter.png')

plot(mod, which = 1)
summary(mod)
