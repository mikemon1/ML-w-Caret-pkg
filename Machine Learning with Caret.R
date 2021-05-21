# Kept getting package install errors and found this work- around on stack
options("install.lock" = FALSE)

library(ggplot2)
install.packages("rlang")
library(rlang)
install.packages("caret")
library(caret)

diamonds <- ggplot2::diamonds
dim(diamonds)

model <- lm(price~.,data=diamonds)
p <-predict(model, diamonds)
model$fitted.values

error <- diamonds$price - model$fitted.values

library(Metrics)
rmse(diamonds$price, p)

rmse(diamonds$price, model$fitted.values)

# Out of sample error measures

data("mtcars")
# Fit a model to mtcars
model <- lm(mpg ~ hp, data = mtcars[1:20,])

# Predict out-of-sample
predicted <- predict(model, mtcars[21:32,], type = "response")

# Evaluate error
actual <- mtcars[21:32,"mpg"]
# Determine RMSE
error <- actual - predicted
sqrt(mean(error^2))
rmse(actual, predicted)

# Randomly ordering the dataset 
# Set seed
set.seed(42)

# Shuffle row indices: rows
rows <-sample(nrow(diamonds))

# Randomly order data
shuffled_diamonds <- diamonds[rows, ]

# Determine row to split on: split
split <- round(nrow(shuffled_diamonds) * 0.80)

# Create train
train <- shuffled_diamonds[1:split, ]

# Create test
test <- shuffled_diamonds[(split + 1):nrow(shuffled_diamonds), ]

dim(train)
dim(test)

# Fit lm model on train: model
model <- lm(price ~., data = train)

# Predict on test: p
p <- predict(model, test)

# Compute errors: error
error <- p - test$price

# Calculate RMSE
sqrt(mean(error^2))

## Cross Validation
# Set seed for reproducibility
set.seed(42)

# Fit the linerar regression model
model_lm <- train(mpg ~ hp, data = mtcars,
               method = "lm",
               trControl = trainControl(
                 method = "cv",
                 number = 10,
                 verboseIter = TRUE
               )
)

# Fit lm model using 10-fold CV: model
model <- train(
  price ~., 
  data = diamonds,
  method = "lm",
  trControl = trainControl(
    method = "cv", 
    number = 10,
    verboseIter = TRUE
  )
)

# Print model to console
model

Boston <- MASS::Boston

# Fit lm model using 5-fold CV: model
model <- train(
  medv ~ ., 
  data = Boston,
  method = "lm",
  trControl = trainControl(
    method = "cv", 
    number = 5,
    verboseIter = TRUE
  )
)

# Print model to console
model

# Fit lm model using 5 x 5-fold CV: model
model <- train(
  medv ~ ., 
  Boston,
  method = "lm",
  trControl = trainControl(
    method = "repeatedcv", 
    number = 5,
    repeats = 5, 
    verboseIter = TRUE
  )
)

# Print model to console
model

# Logistic regression on sonar

# Load the Sonar dataset
library(mlbench)
data("Sonar")

# Examine the data
Sonar[1:6, c(1:5, 61)]

# Randomly ored the dataset
# Get the number of observations
n_obs <- nrow(Sonar)

# Shuffle row indices: permuted_rows
permuted_rows <- sample(nrow(Sonar))

# Randomly order data: Sonar
Sonar_shuffled <- Sonar[permuted_rows, ]

# Identify row to split on: split
split <- round(n_obs * 0.60)

# Create train
train <- Sonar_shuffled[1:split, ]

# Create test
test <- Sonar_shuffled[(split +1):n_obs, ]

# Confirm train set size
nrow(train) / nrow(Sonar)

# Fit glm model: model
model <- glm(Class ~ ., data = train, family = "binomial")

# Predict on test: p
p <- predict(model, test, type = "response")

summary(p)

# Build a confusion matrix

# Turn probabilities into classes and look at their frequencies
# If p exceeds threshold of 0.5, M else R: m_or_r
m_or_r <- ifelse(p > 0.50, "M", "R")

# Convert to factor: p_class
p_class <- as.factor(m_or_r)

# Create confusion matrix
table(p_class, test[["Class"]])

# Class probabilities and predictions

m_or_r <- ifelse(p > 0.99, "M", "R")

# Convert to factor: p_class
p_class <- as.factor(m_or_r)
is.factor(p_class)
table(p_class)

# Create confusion matrix
table(p_class, test[["Class"]])

## Inrtroducing the ROC curve
library(caTools)
colAUC(p, test[["Class"]], plotROC = TRUE)

# Area under the curve (AUC)


# Create trainControl object: myControl
myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
)

# Train glm with custom trainControl: model
model <- train(Class ~.,data = Sonar, method = "glm", family = "binomial", trControl = myControl)


# Print model to console
model

## Random Forests and wine

set.seed(42)
data(Sonar)
model <- train(Class ~., 
               data = Sonar,
               method = "ranger")

plot(model)

library(rattle.data)

wine <- rattle.data::wine
head(wine)

model <- train(
  Type ~., 
  tuneLength = 1,
  data = wine, 
  method = "ranger", 
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)

model

# Random forests require tuning

# Hyperparameters control how the model is fit
# Selected by hand before the model is fit
# Most important is mtry
# mtry is the number of randomly selected variables used at each split
# Lower mtry values = more random
# Higher mtry values = less random

model <- train(
  Class ~.,
  data = Sonar,
  method = "ranger",
  tuneLength = 10
)
plot(model)

# Fit random forest: model
model <- train(
  Type ~.,
  tuneLength = 3,
  data = wine, 
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE
  )
)

# Print model to console
model

# Plot model
plot(model)

# Custom tunong grids
set.seed(42)
# Define a custom tuning grid

myGrid <- data.frame(
  mtry = c(2,3,7),
  splitrule = "variance",
  min.node.size = 5
)

# Fit a model with a custom tuning grid

# Fit random forest: model
model <- train(
  Type ~.,
  tuneGrid = myGrid,
  data = wine, 
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE
  )
)

# Print model to console
model

# Plot model
plot(model)


# Introducing glmnete

# lasso and ridge models: lasso penalizes non-zero coefficients, ridge penalizes absolute values
# alpha [0,1] : pure ridge to pure lasso
# lambda (0,inf): the size of the penalty


# Make a custom trainControl 

myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # <- SUPER IMPORTANT
  verboseIter = TRUE
)

# Fit glmnet model: model
model <- train(
  Class ~., 
  Sonar,
  method = "glmnet",
  trControl = myControl
)

# Print model to console
model

# Print maximum ROC statistic
max(model[["results"]]$ROC)

# glmnet with custom tuning grid

# Make a custom tuning grid
myGrid <- expand.grid(
  alpha = 0:1,
  lambda = seq(0.0001, 0.1, length = 10)
)

model <- train(
  Class ~., 
  Sonar,
  method = "glmnet",
  tuneGrid = myGrid,
  trControl = myControl
)

max(model[["results"]]$ROC)

plot(model)


# Median Imputation

# Generate some data with missing values

data(mtcars)
set.seed(42)
mtcars[sample(1:nrow(mtcars),10),"hp"] <- NA

mtcars$hp

# Split target from predictors

y <- mtcars$mpg
x <- mtcars[, 2:4]

# Try to fit a caret model
model <- train(x,y)

# A simple solution
# Now fit with median imputation

mtcars[mtcars$disp < 140,"hp"] <- NA
Y <- mtcars$mpg
X <- mtcars[, 2:4]


model <- train(X, Y, method = "glm", preProcess = "medianImpute")
print(min(model$results$RMSE))

# KNN Imputation for missingness not at random

set.seed(42)
model <- train(
  X,Y, method = "glm", preProcess = "knnImpute"
)


# Multiple Preprocessing Methods

data(mtcars)
set.seed(42)
mtcars[sample(1:nrow(mtcars),10),"hp"] <- NA
Y <- mtcars$mpg
X <- mtcars[, 2:4] # <- Missing at random

set.seed(42)
model <- train(
  X, Y, method = "glm",
  preProcess = c("medianImpute", "center", "scale")
)

print(min(model$results$RMSE))
print(model)

# With PCA

data(mtcars)
set.seed(42)
mtcars[sample(1:nrow(mtcars),10),"hp"] <- NA
Y <- mtcars$mpg
X <- mtcars[, 2:4] # <- Missing at random

set.seed(42)
model <- train(
  X, Y, method = "glm",
  preProcess = c("medianImpute", "center", "scale", "pca")
)

print(min(model$results$RMSE))
print(model)

# Spaitial sign transformation 
# for dat awith large outliers or high dimensionality

data(mtcars)
set.seed(42)
mtcars[sample(1:nrow(mtcars),10),"hp"] <- NA
Y <- mtcars$mpg
X <- mtcars[, 2:4] # <- Missing at random

set.seed(42)
model <- train(
  X, Y, method = "glm",
  preProcess = c("medianImpute", "center", "scale", "spatialSign")
)

print(min(model$results$RMSE))
print(model)

# Handling low information predictors

# Try to remove low variance predictors



set.seed(42)
mtcars[sample(1:nrow(mtcars),10),"hp"] <- NA
Y <- mtcars$mpg
X <- mtcars[, 2:4] # <- Missing at random
# Add constant-valued column to mtcars
X$bad <- 1

set.seed(42)
model <- train(
  X, Y, method = "glm",
  preProcess = c("medianImpute", "center", "scale", "pca")
)

# Remove constant or near constat predictors from teh data preProcess "zv" or "nzv"

set.seed(42)
model <- train(
  X, Y, method = "glm",
  preProcess = c("zv","medianImpute", "center", "scale", "pca")
)

min(model$results$RMSE)

## PCA
# Remove zero variance data
set.seed(42)
data("BloodBrain")
model <- train(
  bbbDescr,
  logBBB,
  method = "glm",
  trControl = trainControl(
    method = "cv", number = 10, verbose = TRUE
  ),
  preProcess = c("zv", "center","scale")
)
min(model$results$RMSE)

# Remove low-variance data
set.seed(42)
data("BloodBrain")
model <- train(
  bbbDescr,
  logBBB,
  method = "glm",
  trControl = trainControl(
    method = "cv", number = 10, verbose = TRUE
  ),
  preProcess = c("nzv", "center","scale")
)
min(model$results$RMSE)

# Add PCA
set.seed(42)
data("BloodBrain")
model <- train(
  bbbDescr,
  logBBB,
  method = "glm",
  trControl = trainControl(
    method = "cv", number = 10, verbose = TRUE
  ),
  preProcess = c("nzv", "center","scale", "pca")
)
min(model$results$RMSE)

model

# Reusing trainControl

# Example 
myFolds <- createFolds(data, k = x)
i <- myFolds$Fold1
table(data[i])/ length(i)

myControl <- trainControl(
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)

model_glmnet <- train(Churn ~.,data = churn, metric = "ROC", method = "glmnet",
  tuneGrid = expand.grid(alpha = 0:1, lambda = 0.0001, length = 20),
  trControl = trainControl(method = "cv",number = 10, verbose = TRUE,classProbs = TRUE,savePredictions = TRUE))


## Comparing Models

# Make a list of the models

model_list <- list(
  glmnet = model_glmnet,
  rf = model_rf
)
  
# Collect resamples from the CV folds

resamps <- resamples(model_list)
resamps

summary(resamps)

# More on resamples

bwplot(resamps, metric = "ROC")


