library(tidyverse)

setwd('/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project')

df = read_csv('data/final_clean_data.csv')
df= df %>% select(-c(1,2)) # Remove keys (ISO code, Country name)

# Train test split
train_n = floor(nrow(df) * 0.75)
test_n = nrow(df) - train_n
train = sample(c(rep(TRUE, train_n), rep(FALSE, test_n)))

# Naive Bayes for mixed data
nb = e1071::naiveBayes(`Aid Level` ~ ., laplace = 1, data = df, subset=train)
print(fit)

df_test = df[!train, ]
y_pred = predict(nb, df_test)
y_test = df_test$`Aid Level`

caret::confusionMatrix(factor(y_pred),factor(y_test))
