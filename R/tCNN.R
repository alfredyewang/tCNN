library(keras)
library(cluster)
library(dendextend)

tCNN <- function(x_train,y_train,x_test,y_test,C,num_classes,batch_size,epochs,num_filters,window_size,strides_size,conv1_activate_function,dropout_rate,fc1_units,fc1_activate_function,fc2_units,fc2_activate_function) {
#  obj = pam(C,nCluster)
#  clustering=obj$clustering
#  idx = order(clustering)
  C[upper.tri(C)] <- NA
  C <- as.dist(C, diag = TRUE)
  hc1 <- hclust(C, method = "complete" )
  idx = order.hclust(hc1)
  x_train = x_train[,c(idx)]
  x_test = x_test[,c(idx)]

  x_train <- array_reshape(x_train, c(nrow(x_train), dim(x_train)[2],1))
  x_test <- array_reshape(x_test, c(nrow(x_test), dim(x_test)[2],1))
  input_shape <- c(dim(x_test)[2],1)

  cat(nrow(x_train), 'train samples\n')
  cat(nrow(x_test), 'test samples\n')

  y_train <- to_categorical(y_train, num_classes)
  y_test <- to_categorical(y_test, num_classes)

  # Define Model --------------------------------------------------------------
  model <- keras_model_sequential()
  model %>%
  layer_conv_1d(filters = num_filters, kernel_size = window_size, activation = conv1_activate_function,strides = strides_size,
                  input_shape = input_shape) %>%
  layer_dropout(rate = dropout_rate) %>%
  layer_flatten() %>%
  layer_dense(units = fc1_units, activation = fc1_activate_function,) %>%
  layer_dropout(rate = dropout_rate) %>%
  layer_dense(units = fc2_units, activation = fc2_activate_function,) %>%
  layer_dropout(rate = dropout_rate) %>%
  layer_dense(units = num_classes, activation = 'softmax')

  summary(model)

  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_adam(),
    metrics = c('accuracy')
  )


  checkpoint_dir <- "checkpoints"
  unlink(checkpoint_dir, recursive = TRUE)
  dir.create(checkpoint_dir)
  filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

  # Create checkpoint callback
  cp_callback <- callback_model_checkpoint(
    filepath = filepath,
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 1
  )

  # Training & Evaluation ----------------------------------------------------

  # Fit model to data
  history <- model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 1,
    validation_split = 0.2,
    shuffle = TRUE,
    callbacks = list(cp_callback)
  )
  model %>% save_model_hdf5("model.h5")
 plot(history,metrics = c('acc'))

  score <- model %>% evaluate(
    x_test, y_test,
    verbose = 0
  )

  # Output metrics
  cat('Test loss:', score[[1]], '\n')
  cat('Test accuracy:', score[[2]], '\n')
}
