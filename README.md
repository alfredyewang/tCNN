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
tCNN(#(x_train,y_train,x_test,y_test,
  C,nCluster,num_classes,
  batch_size,epochs,num_filters,window_size,strides_size,
  dropout_rate,fc1_units,fc1_activate_function,
  fc2_units,fc2_activate_function,
  fc3_units,fc3_activate_function)
)

```
## Arguments

- x_train     The training dataset
- y_train     The label of training dataset
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

### USA Human Gut Microbiome data (Continous-Outcome)
#### Train the model

The USA Human Gut Microbiome data contains 308 samples with 1087 OTUs. For details of description, please check our paper.
```
python3 src/pCNN.py --train --data_dir data/USA --model_dir model --outcome_type continous --batch_size 16 --max_epoch 2000 --learning_rate 5e-3 --dropout_rate 0.5 --window_size 8 8 8 --kernel_size 64 64 32 --strides 4 4 4
```
After training, the well-trained model will be saved to model directory.
#### Evaluate the well-trained model

```
python3 src/pCNN.py --evaluation --data_dir data/USA --result_dir result/USA --model_dir model --outcome_type continous --batch_size 16 --max_epoch 2000 --learning_rate 5e-3 --dropout_rate 0.5 --window_size 8 8 8 --kernel_size 64 64 32 --strides 4 4 4
```
The program will evaluate the well-trained model, draw a R-squared figure, and save it to result directory.

<center>
<div align=center><img width="400" height="300" src="https://github.com/alfredyewang/pCNN/blob/master/result/USA/result.jpg"/></div>
</center>  


#### Test the model with unlabelled data

```
python3 src/pCNN.py --test --test_file data/USA/X_test.npy  --correlation_file data/USA/c.npy --result_dir result/USA --model_dir model --outcome_type continous --batch_size 16 --max_epoch 2000 --learning_rate 5e-3 --dropout_rate 0.5 --window_size 8 8 8 --kernel_size 64 64 32 --strides 4 4 4
```
The program will take the unlabelled test file and save the prediction result to result directory.


### Malawian Twin pairs Human Gut Microbiome data (Binary-Outcome)
#### Train the model
The USA Human Gut Microbiome data contains 995 samples with 2291 OTUs.
```
python3 src/pCNN.py --train --data_dir data/Malawiantwin_pairs --model_dir model --outcome_type binary --batch_size 32 --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2
```
#### Evaluate the well-trained model

```
python3 src/pCNN.py --evaluation --data_dir data/Malawiantwin_pairs --result_dir result/Malawiantwin_pairs --model_dir model --outcome_type binary --batch_size 32 --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2
```
The program will draw a ROC figure and save it to result directory.

<center>
<div align=center><img width="400" height="300" src="https://github.com/alfredyewang/pCNN/blob/master/result/Malawiantwin_pairs/result.jpg"/></div>
</center>  

#### Test the model with unlabelled data
```
python3 src/pCNN.py --test --test_file data/Malawiantwin_pairs/X_test.npy --correlation_file data/Malawiantwin_pairs/c.npy --result_dir result/Malawiantwin_pairs --model_dir model --outcome_type binary --batch_size 32 --max_epoch 500 --learning_rate 1e-4 --dropout_rate 0.5 --window_size 128 4 --kernel_size 32 32 --strides 64 2
```
