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
#(x_train,y_train,x_test,y_test,C,nCluster,num_classes,batch_size,epochs,num_filters,window_size,strides_size,dropout_rate,fc1_units,fc1_activate_function,fc2_units,fc2_activate_function,fc3_units,fc3_activate_function)
tCNN(x_train=z,y_train=y,x_test=z,y_test=y,C=C,nCluster=20,num_classes=2,batch_size=16,epochs=20,num_filters=32,window_size=16,strides_size=8,dropout_rate=0.5,fc1_units=64,fc1_activate_function='relu',fc2_units=32,fc2_activate_function='relu',fc3_units=8,fc3_activate_function='relu')
