## Tree-regularized Convolutional NeuralNetwork
We develop a deep learning prediction method "Tree-regularized convolutional Neural Network,"(tCNN) for microbiome-based prediction. The advantage of tCNN is that it uses the convolutional kernel to capture the signals of microbiome species with close evolutionary relationship in a local receptive field. Moreover, cCNN uses different convolutional layer to capture different taxonomic rank (e.g. species, genus, family, etc). Together, the convolutional layers with its built-in convolutional kernels capture microbiome signals at different taxonomic levels while encouraging local smoothing induced by the phylogenetic tree.


<center>

<div align=center><img width="600" height="400" src="https://raw.githubusercontent.com/alfredyewang/tCNN/master/docs/Architecture.jpg"/></div>
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
| Arguments     | Character |\
| x_train |The training dataset|\
| y_train |The label of training dataset|


- x_test      The testing dataset
- y_test      The label of testing dataset     
- num_classes     The number of classes
- C
- nCluster
  batch_size,epochs,num_filters,window_size,strides_size,
  dropout_rate,fc1_units,fc1_activate_function,
  fc2_units,fc2_activate_function,
  fc3_units,fc3_activate_function
## Example

Examples
```
## generate 25 objects, divided into 2 clusters.
x <- rbind(cbind(rnorm(10,0,0.5), rnorm(10,0,0.5)),
           cbind(rnorm(15,5,0.5), rnorm(15,5,0.5)))
pamx <- pam(x, 2)
pamx # Medoids: '7' and '25' ...
summary(pamx)
plot(pamx)
## use obs. 1 & 16 as starting medoids -- same result (typically)
(p2m <- pam(x, 2, medoids = c(1,16)))
## no _build_ *and* no _swap_ phase: just cluster all obs. around (1, 16):
p2.s <- pam(x, 2, medoids = c(1,16), do.swap = FALSE)
p2.s

p3m <- pam(x, 3, trace = 2)
## rather stupid initial medoids:
(p3m. <- pam(x, 3, medoids = 3:1, trace = 1))


pam(daisy(x, metric = "manhattan"), 2, diss = TRUE)

data(ruspini)
## Plot similar to Figure 4 in Stryuf et al (1996)
## Not run: plot(pam(ruspini, 4), ask = TRUE)
```
