library(ape)
#library(tCNN)

# Get raw data from twinsgut dataset
load('./twinsgut.Rdata')
source('R/tCNN.R')
age=data.obj$meta.dat$age
tree=data.obj$tree
sex= data.obj$meta.dat$sex
sex = as.integer(sex)
#sex = sex-2
sex[sex==1] <- NA
sex[sex==2] <- 0
sex[sex==3] <- 1

id=!is.na(sex)
samples=rownames(data.obj$meta.dat)[id]
otu.tab=data.obj$otu.tab[,match(samples,colnames(data.obj$otu.tab))]
otu.tab=t(otu.tab)
sex=sex[id]
n=length(sex)
y=sex

# preprocessing  data

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
z=otu.tab[,common.tips]
z[z!=0]=sqrt(z[z!=0])
z=as.matrix(z)

# Correlation of tree
C=exp(-2*D)

# 8-2 Training set and Testing set
cut = as.integer(dim(z)[1]*0.8)
x_train = z[1:cut,]
y_train = y[1:cut]
x_test = z[(cut+1):dim(z)[1],]
y_test = y[(cut+1):dim(z)[1]]

# Call tCNN package
tCNN(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,C=C,nCluster=20,num_classes=2,batch_size=16,epochs=20,num_filters=32,window_size=16,strides_size=8,dropout_rate=0.5,fc1_units=64,fc1_activate_function='relu',fc2_units=32,fc2_activate_function='relu',fc3_units=8,fc3_activate_function='relu')
