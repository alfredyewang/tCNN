library(Rcpp)
library(dirmult)
library(ape)
library(ade4)
library(cluster)
library(MASS)
library(glmnet)
library(vegan)
library(GMPR)
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering visualization
library(dendextend) # for comparing two dendrograms
library(keras)
load('./twinsgut.Rdata')
age=data.obj$meta.dat$age
md=data.obj$meta.dat$zygosity
tree=data.obj$tree
dim(data.obj$meta.dat)
dim(data.obj$otu.tab)
sex= data.obj$meta.dat$sex

sex = as.integer(sex)
sex = sex-2
sex[sex==-1] <- NA
id=!is.na(sex)
samples=rownames(data.obj$meta.dat)[id]
otu.tab=data.obj$otu.tab[,match(samples,colnames(data.obj$otu.tab))]
otu.tab=t(otu.tab)
sex=sex[id]
n=length(sex)
y=sex
md=md[id]



threshold=0.90
zero.per = colSums(otu.tab==0)/n
nzero.idx = which(zero.per<threshold)
zero.idx=which(zero.per>=threshold)
p=length(nzero.idx)
otu.ids=colnames(otu.tab)[nzero.idx]
tree.tips=tree$tip.label
common.tips=intersect(tree.tips,otu.ids); length(common.tips)
tree=drop.tip(tree, setdiff(tree.tips, common.tips))
D=cophenetic(tree)
otu.ids=common.tips
otu.tab=otu.tab[,common.tips]

dim(otu.tab)


#normalization
gmpr.size.factor=GMPR(otu.tab)
summary(gmpr.size.factor)
tab.norm= otu.tab / gmpr.size.factor

q97=apply(tab.norm,2,function(x) quantile(x,0.97))
tab.win=apply(tab.norm,2,function(x) {x[x>quantile(x,0.97)]=quantile(x,0.97); x} )
dim(tab.win)
z=tab.win
z[z!=0]=sqrt(z[z!=0])
z=as.matrix(z)
C=exp(-2*D)

C[upper.tri(C)] <- NA
C <- as.dist(C, diag = TRUE)
hc1 <- hclust(C, method = "complete" )
idx = order.hclust(hc1)
z_e = z[,c(idx)]



# Data Preparation ---------------------------------------------------

  batch_size <- 32
  num_classes <- 2
  epochs <- 10

  # The data, shuffled and split between train and test sets
#  c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()

#  x_train <- array_reshape(x_train, c(nrow(x_train), 784))
 # x_test <- array_reshape(x_test, c(nrow(x_test), 784))
  x_train <- z_e
  x_test <- z_e
   x_train <- array_reshape(x_train, c(nrow(x_train), 418,1))
   x_test <- array_reshape(x_test, c(nrow(x_test), 418,1))
  # Transform RGB values into [0,1] range
   input_shape <- c(418,1)

  cat(nrow(x_train), 'train samples\n')
  cat(nrow(x_test), 'test samples\n')

  # Convert class vectors to binary class matrices
  y_train <- to_categorical(y, num_classes)
  y_test <- to_categorical(y, num_classes)

  # Define Model --------------------------------------------------------------
  filters = 32
  kernel_size = 8
  model <- keras_model_sequential()
  model %>%
    layer_conv_1d(filters = 32, kernel_size = 3, activation = 'relu',strides = 1,
                  input_shape = input_shape) %>%
    layer_dropout(rate = 0.25) %>%
    layer_flatten() %>%
    layer_dense(units = 32, activation = 'relu') %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 2, activation = 'softmax')

  summary(model)

  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )

  # Training & Evaluation ----------------------------------------------------

  # Fit model to data
  history <- model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 1,
    validation_split = 0.2
  )

  plot(history)

  score <- model %>% evaluate(
    x_test, y_test,
    verbose = 0
  )

  # Output metrics
  cat('Test loss:', score[[1]], '\n')
  cat('Test accuracy:', score[[2]], '\n')


