## Tree-regularized Convolutional NeuralNetwork
We develop a deep learning prediction method "Tree-regularized convolutional Neural Network,"(tCNN) for microbiome-based prediction. The advantage of tCNN is that it uses the convolutional kernel to capture the signals of microbiome species with close evolutionary relationship in a local receptive field. Moreover, cCNN uses different convolutional layer to capture different taxonomic rank (e.g. species, genus, family, etc). Together, the convolutional layers with its built-in convolutional kernels capture microbiome signals at different taxonomic levels while encouraging local smoothing induced by the phylogenetic tree.


<center>

<div align=center><img width="800" height="400" src="https://raw.githubusercontent.com/alfredyewang/tCNN/master/docs/Architecture.jpg"/></div>
</center>  

## Requirements and

tCNN is implemented by Keras for R. Please check the guide on official website for detail instruction of installing Keras for R. To use tCNN, R version >= 3.0 is required. Versions for other packages are suggested.

-R >= 3.0 (64-bit)

-Python 2.7 (64-bit)

-Keras 2.2.4 in R and Python

-cluster 2.1.0

## Installation
Install and load tCNN:
```
devtools::install_github("alfredyewang/tCNN")
library("tCNN")
```
## Usage

```
tCNN(x_train,y_train,x_test,y_test,
  C,nCluster,num_classes,
  batch_size,epochs,num_filters,window_size,strides_size,
  dropout_rate,fc1_units,fc1_activate_function,
  fc2_units,fc2_activate_function,
  fc3_units,fc3_activate_function
  )

```
## Arguments
| Arguments     | Character |
| ------------- | ------------- |
| x_train |The training dataset|
| y_train |The label of training dataset|
| x_test  |    The testing dataset|
| y_test      |The label of testing dataset|     
|num_classes     |The number of classes|
|C      | The correlation matrix for Tree Structure |
|nCluster | number of Cluster |
|batch_size | The batch size for neural network |
|epochs | The max epoch batch size for training  neural network |
|num_filters | The number of filters in convolutional layers |
|window_size | The window size in convolutional layers|
|strides_size | The strides size in convolutional layers |
|dropout_rate | The dropout rate for training  neural network
|fc1_units | The number of node in the first fully connected layers |
|fc1_activate_function | The activation function for the first fully connected layers (relu, tanh, softmax)|
|fc2_units | The number of node in the second fully connected layers |
|fc2_activate_function |The activation function for the second fully connected layers (relu, tanh, softmax)|
|fc3_units | The number of node in the third fully connected layers |
fc3_activate_function |The activation function for the third fully connected layers (relu, tanh, softmax)|
## Example

Examples
```
library(Rcpp)
library(dirmult)
library(ape)
library(ade4)
library(MASS)
library(vegan)
source('R/tCNN.R')
load('./twinsgut.Rdata')
age=data.obj$meta.dat$age
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
z=otu.tab
z[z!=0]=sqrt(z[z!=0])
z=as.matrix(z)
C=exp(-2*D)
cut = as.integer(dim(z)[1]*0.8)
x_train = z[1:cut,]
y_train = y[1:cut]
x_test = z[(cut+1):dim(z)[1],]
y_test = y[(cut+1):dim(z)[1]]


tCNN(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,C=C,nCluster=20,num_classes=2,batch_size=16,epochs=20,num_filters=32,window_size=16,strides_size=8,dropout_rate=0.5,fc1_units=64,fc1_activate_function='relu',fc2_units=32,fc2_activate_function='relu',fc3_units=8,fc3_activate_function='relu')

```
