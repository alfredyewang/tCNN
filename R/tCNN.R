library(keras)
library(cluster)

# Data Preparation ---------------------------------------------------
tCNN <- function(x_train,y_train,x_test,y_test,C,nCluster,num_classes,batch_size,epochs,num_filters,window_size,strides_size,dropout_rate,fc1_units,fc1_activate_function,fc2_units,fc2_activate_function,fc3_units,fc3_activate_function) {
  obj = pam(C,nCluster)
  clustering=obj$clustering
  idx = order(clustering)
  x_train = x_train[,c(idx)]
  x_test = x_test[,c(idx)]

  x_train <- array_reshape(x_train, c(nrow(x_train), dim(x_train)[2],1))
  x_test <- array_reshape(x_test, c(nrow(x_test), dim(x_test)[2],1))
  input_shape <- c(dim(x_test)[2],1)

  cat(nrow(x_train), 'train samples\n')
  cat(nrow(x_test), 'test samples\n')

  y_train <- to_categorical(y, num_classes)
  y_test <- to_categorical(y, num_classes)

  # Define Model --------------------------------------------------------------
  model <- keras_model_sequential()
  model %>%
  layer_conv_1d(filters = num_filters, kernel_size = window_size, activation = 'relu',strides = strides_size,
                  input_shape = input_shape) %>%
  layer_dropout(rate = dropout_rate) %>%
  layer_flatten() %>%
  layer_dense(units = fc1_units, activation = fc1_activate_function,) %>%
  layer_dropout(rate = dropout_rate) %>%
  layer_dense(units = fc1_units, activation = fc1_activate_function,) %>%
  layer_dropout(rate = dropout_rate) %>%
  layer_dense(units = fc1_units, activation = fc1_activate_function,) %>%
  layer_dropout(rate = dropout_rate) %>%
  layer_dense(units = num_classes, activation = 'softmax')

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
}
