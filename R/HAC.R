library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering visualization
library(dendextend) # for comparing two dendrograms
df <- USArrests
df <- na.omit(df)
df <- scale(df)
head(df)
# Dissimilarity matrix
d <- dist(df, method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 <- hclust(d, method = "complete" )

# Plot the obtained dendrogram
plot(hc1, cex = 0.6, hang = -1)
set.seed(23235)
ss <- sample(1:150, 10 )
hc <- iris[ss,-5] %>% dist %>% hclust
# dend <- hc %>% as.dendrogram
plot(hc)
order.hclust(hc)
