# Importing the dataset 
#setwd("~/Documents/info411/project/") 

#------ visualisation of data -----
# remove scientific notation
options(scipen=999)  

# Load data
df.train <- read.csv("training.csv")
df.test <- read.csv("testing.csv")
df.train = df.train[2:514]
df.test = df.test[2:514]

tbl <- with(dataset_training, table(X1))
barplot(tbl, beside = TRUE, legend = FALSE)

# correlation against target variable
correlations = cor(x=df.train, y=df.train$X1)
correlations = sort(correlations, decreasing=TRUE)
require(lares)
corr_var(df.train, X1, top = 30)
#---------svm--------------
library(readr)
library(caret)
library(e1071)
library(ggplot2)
library(klaR)

data <- read.csv("training.csv", header = FALSE)
train <- data[,-1] #remove column one
colnames(train)[1] <- "label" #change first column name to label


data <- read.csv("testing.csv", header = FALSE)
test <- data[,-1] #remove column one
colnames(test)[1] <- "label" #change first column name to label


#min max scaling bring all values to range 0-1
process_train <- preProcess(train[,-1], method=c("range"))#normalize w/o label
train <- predict(process_train, train)


process_test <- preProcess(test[,-1], method=c("range"))#normalize w/o label
test <- predict(process_test, test)

#generate header
col_names <- c("label", paste0("x", 2:ncol(train)))
colnames(train) <- col_names
colnames(test) <- col_names

#find corelation
corTable = abs(cor(train, y = train$label))
corTable = corTable[order(corTable, decreasing = TRUE),,drop = FALSE]

#set top 100 corelation features as dataset features
top_cor_cols <- row.names(head(corTable, 100))
train_subset <- train[, c("label", top_cor_cols)]
train <- train_subset[,-2] #remove column one

test_subset <- test[, c("label", top_cor_cols)]
test <-test_subset[,-2]

svmModel = svm(formula = label ~., data = train,type = "C-classification",kernel = "radial")

test_pred = predict(svmModel, newdata = test)
test_pred

# Create a confusion matrix to evaluate model performance
cm <- table(test$label, test_pred)

# Calculate the accuracy of the model
acc <- sum(diag(cm)) / sum(cm)
acc
# Print the accuracy
cat("Accuracy:", round(acc, 4), "\n")
#Accuracy: 0.4739


#svm fine tuning
train$label <- as.factor(train$label)
test$label <- as.factor(test$label)

# tune SVM hyperparameters using cross-validation
svm_model <- train(label ~ ., data = train, method = "svmRadial",
                   trControl = trainControl(method = "cv", number = 3),
                   preProcess = c("center", "scale"))

# evaluate performance on test data
test_pred <- predict(svm_model, newdata = test)

# Create the confusion matrix
confusion <- table(test$label, test_pred)
confusion

# Compute the accuracy
accuracy <- sum(diag(confusion)) / sum(confusion)
accuracy
#accuracy 0.4748015






#----------nb-------------
library(klaR)
library(pROC)

data <- read.csv("training.csv", header = FALSE)
traind <- data[,-1] #remove column one
# generate random dummy names
new_names <- paste0("placeholder.", 1:ncol(traind))
# rename columns in train and test
colnames(traind) <- new_names
colnames(traind)[1] <- "predicton" #change first column name to label
predicton <- traind$predicton
traind <- traind[, -which(names(traind) == "predicton")]
traind <- cbind(traind, predicton)
traind$predicton <- as.factor(traind$predicton)
levels(traind$predicton) <- paste0("Bird", 1:200)



data <- read.csv("testing.csv", header = FALSE)
testd <- data[,-1] #remove column one
# rename columns in train and test
colnames(testd) <- new_names
colnames(testd)[1] <- "label" #change first column name to label
predicton <- testd$predicton
testd <- testd[, -which(names(testd) == "predicton")]
testd <- cbind(testd, predicton)
testd$predicton <- as.factor(testd$predicton)
levels(testd$predicton) <- paste0("Bird", 1:200)


nb <- naiveBayes(traind[, -200], traind$predicton) #train the model
pred <- predict(nb, newdata = testd)#predict with test data
#testd$predicton <- factor(pred, levels = 1:200)
cm <- table(testd$predicton, pred)

# Calculate accuracy
accuracy <- sum(diag(cm)) / sum(cm)
cat("Accuracy:", round(accuracy, 3))

# Define cross-validation method
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE)



# Define hyperparameter grid
tuneGrid <- expand.grid(fL = c(0, 0.5, 1), usekernel= c(TRUE) ,adjust = c(0, 0.5, 1))

# Perform hyperparameter tuning
set.seed(123)
nb_tuned <- train(x = traind[, -ncol(traind)], y = traind[, ncol(traind)],
                  method = "nb", trControl = ctrl, tuneGrid = tuneGrid)


# Train final model with optimal hyper parameters
nb_model <- naiveBayes(x = traind[, -ncol(traind)], y = traind[, ncol(traind)],
                       laplace = nb_tuned$bestTune$fL, adjust = nb_tuned$bestTune$adjust)

# Get predicted class labels for test set
pred <- predict(nb_model, newdata = testd)
cm <- table(testd$predicton, pred)
# Calculate accuracy
accuracy <- sum(diag(cm)) / sum(cm)
cat("Accuracy:", round(accuracy, 3))

# Plot ROC curve
pred2 <- as.numeric(pred)
roc <- multiclass.roc(testd$predicton, pred2)
plot(roc, print.auc = TRUE, print.thres = TRUE)


#---------neural network-----------
# Load required packages
library(neuralnet)
library(dplyr)
library(ROCR)
library(caret)

# Load data
train_df <- read.csv("training.csv")
test_df <- read.csv("testing.csv")

# Split data into predictors and labels
train_y <- as.factor(train_df[, 2])
train_x <- train_df[, -c(1, 2)]
test_y <- as.factor(test_df[, 2])
test_x <- test_df[, -c(1, 2)]

# Normalize data
preprocess_train <- preProcess(train_x, method = c("range"))
preprocess_test <- preProcess(test_x, method = c("range"))
train_x_norm <- predict(preprocess_train, train_x)
test_x_norm <- predict(preprocess_test, test_x)

model = neuralnet(
  train_y ~.,
  train_x_norm,
  hidden=c(264, 200),
  threshold=0.04,
  learningrate = 0.1,
  algorithm = "backprop",
  lifesign = "full",
  lifesign.step = 1000,
  rep = 2,
  linear.output = FALSE
)

pred <- predict(model, test_x_norm)
prediction_label <- data.frame(apply(pred, 1, which.max)) %>%
  mutate(pred = levels(factor(apply(pred, 1, which.max)))) %>%
  select(2) %>%
  unlist()
# 50% correct

# Evaluate model performance
confusion_matrix <- table(test_y, prediction_label)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
accuracy

# Print results
print(confusion_matrix)
print(paste0("Accuracy: ", accuracy * 100))



#PROOOFFFFFF PLZZZ dont run this!!!!!
#.                 *README*                       .#
#----> refer to slide 32 in our presentation
#FINE TUNING DONT WORK BECAUSE R has run out of
#memory to allocate for the neural network training 
#process. This happen when the dataset is very 
#large or when the neural network architecture is too complex.
#We already tried reducing the nodes, reduced complexity
#of our architecyture :(((((( only keras and h2o can 
# support the large multinomial classification models
#  BUT we decided to use h2o as our laptops cannot support 
#keras too-- we also dont have enough computing power and memory hahaha.
# we found out that h2o allows us to connect to their 
# servers so we can use their computing powers instead!! YAYS LUCKY.
# Our conclusion is that neuralnet() function has some limitations and 
# is not optimized for large-scale neural network training. -.- 
# 

# fine tuning-- dont work pls dont run this profffff
# fine_tuned_model <- neuralnet(
#   train_y ~.,
#   train_x_norm,
#   hidden=c(100, 50),
#   threshold=0.04,
#   learningrate = 0.01,
#   algorithm = "backprop",
#   lifesign = "full",
#   lifesign.step = 1000,
#   rep = 2,
#   linear.output = FALSE,
#   startweights = model$weights #taking the best weights to tune model
# )
# 
# # Evaluate the performance of the fine-tuned model on the validation set
# predictions <- compute(fine_tuned_model, test_x_norm)

#-------------h2o model to support findings via deep learning---------
library(h2o)
h2o.init(nthreads=-1, max_mem_size="20G") #connect to h2o server
dataset_training = read.csv('training.csv') 
dataset_training = dataset_training[2:514] 
dataset_testing = read.csv('testing.csv') 
dataset_testing = dataset_testing[2:514]   
#install.packages("h2o", type="source", repos="https://h2o-release.s3.amazonaws.com/h2o/rel-zz_kurka/1/R")  library(h2o) h2o.init(nthreads=-1, max_mem_size="20G") args(h2o.deeplearning) 
help(h2o.deeplearning) 
example(h2o.deeplearning) 
h2o.no_progress()  # Disable progress bars for Rmd   
y <- "X1"  
#response column: digits 1-200 
x <- setdiff(names(dataset_training), y)  #vector of predictor column names  
# Since the response is encoded as integers, we need to tell H2O that 
# the response is in fact a categorical/factor column.  Otherwise, it  
# will train a regression model instead of multiclass classification. 
dataset_training[,y] <- as.factor(dataset_training[,y]) 
dataset_testing[,y] <- as.factor(dataset_testing[,y])   
dl_fit1 <- h2o.deeplearning(x = x,                             
                            y = y,                             
                            training_frame = as.h2o(dataset_training),                             
                            model_id = "dl_fit1",                             
                            hidden = c(200,200),                             
                            seed = 1)  
dl_fit2 <- h2o.deeplearning(x = x,                             
                            y = y,                             
                            training_frame = as.h2o(dataset_training),                             
                            model_id = "dl_fit2",                             
                            epochs = 50,                             
                            hidden = c(200,200),                             
                            stopping_rounds = 0,  # disable early stopping                             
                            seed = 1)  
dl_fit3 <- h2o.deeplearning(x = x,                             
                            y = y,                             
                            training_frame = as.h2o(dataset_training),                             
                            model_id = "dl_fit3",                             
                            epochs = 50,                             
                            hidden = c(200,200),                             
                            nfolds = 3,                            
                            #used for early stopping                             
                            score_interval = 1,                    
                            #used for early stopping                             
                            stopping_rounds = 5,                   
                            #used for early stopping                             
                            stopping_metric = "misclassification", 
                            #used for early stopping                             
                            stopping_tolerance = 1e-3,             
                            #used for early stopping                             
                            variable_importances = T,                              
                            seed = 1)   
h2o.performance(dl_fit3,xval = T) 
h2o.mse(dl_fit3)  
dl_perf1 <- h2o.performance(model = dl_fit1, newdata = as.h2o(dataset_testing)) 
dl_perf2 <- h2o.performance(model = dl_fit2, newdata = as.h2o(dataset_testing)) 
dl_perf3 <- h2o.performance(model = dl_fit3, newdata = as.h2o(dataset_testing))  
# Retreive test set MSE 
h2o.mse(dl_perf1) 
h2o.mse(dl_perf2)  
h2o.mse(dl_perf3)

h2o.scoreHistory(dl_fit3) 
h2o.confusionMatrix(dl_fit3)

graph1 = plot(dl_fit3,       
              timestep = "epochs",       
              metric = "classification_error")  
graph1  

cv_models <- sapply(dl_fit3@model$cross_validation_models,                      
                    function(i) h2o.getModel(i$name))  

# Plot the scoring history over time 
graph2 = plot(cv_models[[1]],       
              timestep = "epochs",       
              metric = "classification_error")  
graph2

# mean per class error for train/test esets
h2o.mean_per_class_error(dl_fit1)
h2o.mean_per_class_error(dl_perf1)

h2o.mean_per_class_error(dl_fit2)
h2o.mean_per_class_error(dl_perf2)

h2o.mean_per_class_error(dl_fit3)
h2o.mean_per_class_error(dl_perf3)

# get hit ratio for each model
h2o.hit_ratio_table(dl_fit1)
h2o.hit_ratio_table(dl_perf1)

h2o.hit_ratio_table(dl_fit2)
h2o.hit_ratio_table(dl_perf2)

h2o.hit_ratio_table(dl_fit3)
h2o.hit_ratio_table(dl_perf3)

# hyperparameter tuning for first model
# goal is to generalize model 
# so put in L2 regularization 
dl_params1 <- list( l2 = c(0, 0.00001, 0.001, 0.1, 1),
                    activation = c("Rectifier", "Tanh", "Rectifier with dropout",
                                   "Tanh with dropout"),
                    epochs = c(10,20,30)
)

# Train and validate a cartesian grid of deep learning models
dl_grid <- h2o.grid("deeplearning", x = x, y = y,
                    grid_id = "dl_grid_tuned",
                    training_frame = as.h2o(dataset_training),
                    seed = 1,
                    balance_classes = TRUE,
                    train_samples_per_iteration = 0,
                    hyper_params = dl_params1)

# Get the grid results, sorted by validation mean per class error
dl_gridperf_mean <- h2o.getGrid(grid_id = "dl_grid_tuned",
                                sort_by = "mean_per_class_error",
                                decreasing = TRUE)
print(dl_gridperf_mean)

# don't take the "best" model because it can cause overfitting
# instead, take average performing model with 
# mean per class error < 0.2
best_dl_model <- h2o.getModel(dl_gridperf_mean@model_ids[[50]])

# build the confusion matrix with train data:
h2o.confusionMatrix(best_dl_model)


dl_perf <- h2o.performance(best_dl_model, newdata = as.h2o(dataset_testing)) 
dl_perf
h2o.mean_per_class_error(dl_perf)


dl_pred <- h2o.predict(best_dl_model, as.h2o(dataset_testing)) 
actual <- dataset_testing$X1

# retrieve the model metrics:
metrics <- h2o.make_metrics(dl_pred[,2:201], as.h2o(actual))
h2o.mean_per_class_error(metrics)
