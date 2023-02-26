setwd('/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project/data/')
path = 'final_clean_data.csv'

df = readr::read_csv(path,col_select=-1)
head(df)

df_num = dplyr::select_if(df, is.numeric)# only select numeric data
df_num_scaled = scale(df_num) # normalize the data
rownames(df_num_scaled) = df$ISO3 # Keep country names


X = as.dist(philentropy::distance(df_num_scaled, method="cosine", use.row.names = TRUE))

hc = hclust(X, method = 'ward.D2')
plot(hc)
abline(h = 0.95, col = 'red')
y_pred = cutree(hc, h = 0.95) # predictions
