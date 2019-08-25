library(tCNN)

# Get raw data from twinsgut dataset

C<-read.table("C.txt",header=FALSE,sep="\t")
z<-read.table("X.txt",header=FALSE,sep="\t")
y<-read.table("Y.txt",header=FALSE,sep="\t")
z = as.matrix(z)
y=as.matrix(y)
C=as.matrix(C)

cut = as.integer(dim(z)[1]*0.8)
x_train = z[1:cut,]
y_train = y[1:cut,]
x_test = z[(cut+1):dim(z)[1],]
y_test = y[(cut+1):dim(z)[1],]

# Call tCNN package
tCNN(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,C=C,num_classes=2,batch_size=16,epochs=100,num_filters=64,window_size=256,strides_size=32,conv1_activate_function='relu',dropout_rate=0.8,fc1_units=128,fc1_activate_function='tanh',fc2_units=32,fc2_activate_function='tanh')
