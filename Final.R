library(tidyverse)
library(caret)
library(glmnet)
library(GGally)
library(MASS)
library(splines)

# read data
set.seed(2022723)
df <- read.table('~/Desktop/ridge_lasso_stepwise/ridge_lasso_stepwise/data_tr.txt', header = TRUE, sep = '\t', dec = '.')[,-1]

# remove perfect collinear column
df <- subset(df, select = -c(hequity, smcol))
df

# train-test split
training_sample <- createDataPartition(df $ tw, p = 0.8, list = FALSE)
train <- df[training_sample, ]
test <- df[-training_sample, ]

# Create dataframe only for dependent variable 
x_train = subset(train, select = -c(tw))
y_train = train[['tw']]
x_test = subset(test, select = -c(tw))
y_test = test[['tw']]

# Create base train test dataframe that remain unchanged. Will be used later
base_train <- train
base_test <- test
base_xtrain <- x_train
base_xtest <- x_test
base_ytrain <- y_train
base_ytest <- y_test

# Baseline model
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 6, nrow = k, dimnames = list(1:k,c('null','full', 'ridge', 'lasso', 'forward', 'backward')))
for (j in 1:k) {
  tr <- (ii != j)
  vali <- (ii == j)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  # null model
  null <- lm(tw ~ 1, data = train[tr, ])
  mspe_null = log(mean((y_vali - predict(null, x_vali))^2))
  # full model
  full <- lm(tw ~. -tw, data = train[tr, ])
  mspe_full = log(mean((y_vali - predict(full, x_vali))^2))
  # ridge model
  ridge_hyper <- 10 ^ seq(3,-2, -0.01)
  cv_ridge = cv.glmnet(as.matrix(x_tr), 
                       y_tr, 
                       type.measure = 'mse',
                       nfold = 10,
                       alpha = 0, lambda = ridge_hyper)
  ridge <- glmnet(x = x_tr, y = y_tr, lambda = cv_ridge$lambda.min, alpha = 0)
  mspe_ridge = log(mean((y_vali - predict(ridge, as.matrix(x_vali)))^2))
  print(cv_ridge$lambda.min)
  # lasso model
  lasso_hyper <- 100 ^ seq(2,-1, -0.001)
  cv_lasso = cv.glmnet(as.matrix(x_tr), 
                       y_tr, 
                       type.measure = 'mse',
                       nfold = 10,
                       alpha = 1, lambda = lasso_hyper)
  lasso <- glmnet(x = x_tr, y = y_tr, lambda = cv_lasso$lambda.min, alpha = 1)
  mspe_lasso = log(mean((y_vali - predict(lasso, as.matrix(x_vali)))^2))
  print(cv_lasso$lambda.min)
  #forward
  forward <- stepAIC(null, scope = list(lower = null, upper = full), trace = F, direction = 'forward')
  mspe_forward <- log(mean((predict(forward, newdata = x_vali) - y_vali)^2))
  #backward
  backward <- stepAIC(full, scope = list(lower = null, upper = full), trace = F, direction = 'backward')
  mspe_backward <- log(mean((predict(backward, newdata = x_vali) - y_vali)^2))
  #add results into the matrix
  mse[j,'null'] = mspe_null
  mse[j,'full'] = mspe_full
  mse[j,'ridge'] = mspe_ridge
  mse[j,'lasso'] = mspe_lasso
  mse[j, 'forward'] = mspe_forward
  mse[j, 'backward'] = mspe_backward
}
baseline_mse = colMeans(mse)
baseline_mse

#--------------------------------------------------------------------

# Explorative data analysis

# Plot matrix using category e401. red:no e401. blue:have e401
df %>% mutate(e401 = factor(e401))  %>% ggpairs(
  columns = c('ira' , 'e401' , 'nifa' , 'inc' , 'hmort' , 'hval', 'tw'),
  aes(color = e401),
  upper = list(
    combo = wrap('box_no_facet', size = 0.1),
    discrete = 'count'
    ),
  lower = list(continuous = wrap('points', size = 0.1)),
  axisLabels = 'none'
)

#--------------------------------------------------------------------


# outlier
# the inc-tw scatter, no clear outlier
plot(df$inc, df$tw, pch = 20)
# there is a inc < 0, doesn't make sense
train <- base_train[base_train$inc >= 0,]
# outlier for tw
boxplot(tw ~ male,data =  df, pch = 20)
boxplot(tw ~ e401,data =  df, pch = 20)
boxplot(tw ~ twoearn,data =  df, pch = 20)
boxplot(tw ~ age,data =  df, pch = 20)
boxplot(tw ~ fsize,data =  df, pch = 20)
boxplot(tw ~ marr,data =  df, pch = 20)
train <- train[train$tw < 1700000 & (train$tw > -42000),]
# no clear outlier for hval and hmort
plot(df$hval, df$tw, pch = 20)
boxplot(df$hval, pch =20)

plot(df$hmort, df$tw, pch = 20)
boxplot(df$hmort, pch =20)

reg <- lm(tw ~ ira + inc + age + educ, data = df)
plot(reg)


# After cleaning outlier

# Cross validation
set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 6, nrow = k, dimnames = list(1:k,c('null','full', 'ridge', 'lasso', 'forward', 'backward')))
for (j in 1:k) {
  tr <- (ii != j)
  vali <- (ii == j)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  
  # lasso model
  lasso_hyper <- append(100 ^ seq(2,-1, -0.005),0)
  cv_lasso = cv.glmnet(as.matrix(x_tr), 
                       y_tr, 
                       type.measure = 'mse',
                       nfold = 10,
                       alpha = 1, lambda = lasso_hyper)
  lasso <- glmnet(x = x_tr, y = y_tr, lambda = cv_lasso$lambda.min, alpha = 1)
  mspe_lasso = log(mean((y_vali - predict(lasso, as.matrix(x_vali)))^2))

  mse[j,'lasso'] = mspe_lasso

}
outlier_mse = colMeans(mse)
outlier_mse

#--------------------------------------------------------------------


# Log transformation

# function used for log transformation
# some features have negative values which need to be treated separately
log_trans <-  function(x) {
  if (x > 0) {
    return(log(x))
  } else if (x == 0) {
    return(0)
  } else {
    return(-log(abs(x)))
  }
}

posi_log <-  function(x) {
  
  return(log(1+x+abs(min(x))))
  
}


# Plot the distribution of columns before and after log transformation
hist(train$nifa, breaks = 80)
hist(apply(train['nifa'], 1, log_trans), breaks =40)
hist(train$inc, breaks = 80)
hist(apply(train['inc'], 1, log_trans), breaks = 60)

# Log transform the columns
train[,'inc'] <- apply(train['inc'], 1, log1p)
train[,'nifa'] <- apply(train['nifa'], 1, log1p)

# keep track of the train x/y set
x_train = subset(train, select = -c(tw))
y_train = train[['tw']]

# ten fold cv to see the mse of log method
set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 6, nrow = k, dimnames = list(1:k,c('null','full', 'ridge', 'lasso', 'forward', 'backward')))
for (j in 1:k) {
  tr <- (ii != j)
  vali <- (ii == j)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  # lasso model
  lasso_hyper <- append(10 ^ seq(3,1, -0.001),0)
  cv_lasso = cv.glmnet(as.matrix(x_tr), 
                       y_tr, 
                       type.measure = 'mse',
                       nfold = 10,
                       alpha = 1, lambda = lasso_hyper)
  print(paste(j, cv_lasso$lambda.min))
  lasso <- glmnet(x = x_tr, y = y_tr, lambda = cv_lasso$lambda.min, alpha = 1)
  mspe_lasso = log(mean((y_vali - predict(lasso, as.matrix(x_vali)))^2))
  
  mse[j,'lasso'] = mspe_lasso
}
log_mse = colMeans(mse)
log_mse

#--------------------------------------------------------------------

# Binary transformation

# Reset training data after previous log transformation
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]
# Plot the distribution before and after transformation
hist(train$ira, breaks = 80)
hist(as.numeric(factor(ifelse(train$ira==0, 0,1))), breaks = 5)
hist(train$hmort, breaks = 80)
hist(as.numeric(factor(ifelse(train$hmort==0, 0,1))), breaks = 5)
hist(train$hval, breaks = 80)
hist(as.numeric(factor(ifelse(train$hval==0, 0,1))), breaks = 5)
# Transform the column
train[,'ira'] <- as.numeric(factor(ifelse(train$ira==0, 0,1)))
train[,'hmort'] <- as.numeric(factor(ifelse(train$ira==0, 0,1)))
train[,'hval'] <- as.numeric(factor(ifelse(train$ira==0, 0,1)))

# keep track of the train x/y set, test x/y set
x_train = subset(train, select = -c(tw))
y_train = train[['tw']]

# ten fold cv to see the mse of binarization method
set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 6, nrow = k, dimnames = list(1:k,c('null','full', 'ridge', 'lasso', 'forward', 'backward')))
for (j in 1:k) {
  tr <- (ii != j)
  vali <- (ii == j)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  # lasso model
  lasso_hyper <- append(10 ^ seq(3,1, -0.001),0)
  cv_lasso = cv.glmnet(as.matrix(x_tr), 
                       y_tr, 
                       type.measure = 'mse',
                       nfold = 10,
                       alpha = 1, lambda = lasso_hyper)
  print(paste(j, cv_lasso$lambda.min))
  lasso <- glmnet(x = x_tr, y = y_tr, lambda = cv_lasso$lambda.min, alpha = 1)
  mspe_lasso = log(mean((y_vali - predict(lasso, as.matrix(x_vali)))^2))
  
  mse[j,'lasso'] = mspe_lasso
}
bi_mse = colMeans(mse)
bi_mse

#--------------------------------------------------------------------

# Interaction terms

# Reset training data after previous log transformation
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]
# Using .*. for all interactions except nosh, hs, col
f <- as.formula(tw ~ .*(.-nohs-hs-col))

# create dataframe that include all variables and their interaction
y <- train$tw
train <- as.data.frame(model.matrix(f, train)[, -1])
train$tw <- y

# # keep track of the train x/y set
x_train = subset(train, select = -c(tw))
y_train = train[['tw']]

# ten fold cv to see the mse of interaction method
set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 6, nrow = k, dimnames = list(1:k,c('null','full', 'ridge', 'lasso', 'forward', 'backward')))
for (j in 1:k) {
  tr <- (ii != j)
  vali <- (ii == j)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  
  # lasso model
  lasso_hyper <- 10 ^ seq(4,2.5, -0.0005)
  cv_lasso = cv.glmnet(as.matrix(x_tr), 
                       y_tr, 
                       type.measure = 'mse',
                       nfold = 10,
                       alpha = 1, lambda = lasso_hyper)
  print(paste(j, cv_lasso$lambda.min))
  lasso <- glmnet(x = x_tr, y = y_tr, lambda = cv_lasso$lambda.min, alpha = 1)
  mspe_lasso = log(mean((y_vali - predict(lasso, as.matrix(x_vali)))^2))
  
  mse[j,'lasso'] = mspe_lasso
 
}
inter_mse = colMeans(mse)
inter_mse

#--------------------------------------------------------------------


# Reset training data from previous transformation
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]

# Interaction marr*male
train[,c( 'marr-male')] <- cbind(
  train$marr * train$male
)

set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 6, nrow = k, dimnames = list(1:k,c('null','full', 'ridge', 'lasso', 'forward', 'backward')))
for (j in 1:k) {
  tr <- (ii != j)
  vali <- (ii == j)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  
  # lasso model
  lasso_hyper <- append(10 ^ seq(3,-2, -0.005),0)
  cv_lasso = cv.glmnet(as.matrix(x_tr), 
                       y_tr, 
                       type.measure = 'mse',
                       nfold = 10,
                       alpha = 1, lambda = lasso_hyper)
  print(paste(j, cv_lasso$lambda.min))
  lasso <- glmnet(x = x_tr, y = y_tr, lambda = cv_lasso$lambda.min, alpha = 1)
  mspe_lasso = log(mean((y_vali - predict(lasso, as.matrix(x_vali)))^2))
  
  mse[j,'lasso'] = mspe_lasso
  
}
inter2_mse = colMeans(mse)
inter2_mse

#--------------------------------------------------------------------

# Reset training data from previous transformation
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]

# Interaction marr*male, eudc*male
train[,c( 'marr-male','educ-male')] <- cbind(
  train$marr * train$male,train$educ * train$male
)

set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 6, nrow = k, dimnames = list(1:k,c('null','full', 'ridge', 'lasso', 'forward', 'backward')))
for (j in 1:k) {
  tr <- (ii != j)
  vali <- (ii == j)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  
  # lasso model
  lasso_hyper <- append(10 ^ seq(3,-2, -0.005),0)
  cv_lasso = cv.glmnet(as.matrix(x_tr), 
                       y_tr, 
                       type.measure = 'mse',
                       nfold = 10,
                       alpha = 1, lambda = lasso_hyper)
  print(paste(j, cv_lasso$lambda.min))
  lasso <- glmnet(x = x_tr, y = y_tr, lambda = cv_lasso$lambda.min, alpha = 1)
  mspe_lasso = log(mean((y_vali - predict(lasso, as.matrix(x_vali)))^2))
  
  mse[j,'lasso'] = mspe_lasso
  
}
inter3_mse = colMeans(mse)
inter3_mse

#--------------------------------------------------------------------

# Reset training data from previous transformation
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]

# Interaction marr*male, educ*male, educ*age
train[,c( 'marr-male','educ-male', 'educ-age')] <- cbind(
  train$marr * train$male,train$educ * train$male,train$educ * train$age
)

set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 6, nrow = k, dimnames = list(1:k,c('null','full', 'ridge', 'lasso', 'forward', 'backward')))
for (j in 1:k) {
  tr <- (ii != j)
  vali <- (ii == j)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  
  # lasso model
  lasso_hyper <- append(10 ^ seq(3,-2, -0.005),0)
  cv_lasso = cv.glmnet(as.matrix(x_tr), 
                       y_tr, 
                       type.measure = 'mse',
                       nfold = 10,
                       alpha = 1, lambda = lasso_hyper)
  print(paste(j, cv_lasso$lambda.min))
  lasso <- glmnet(x = x_tr, y = y_tr, lambda = cv_lasso$lambda.min, alpha = 1)
  mspe_lasso = log(mean((y_vali - predict(lasso, as.matrix(x_vali)))^2))
  
  mse[j,'lasso'] = mspe_lasso
  
}
inter4_mse = colMeans(mse)
inter4_mse

#--------------------------------------------------------------------

# Reset training data from previous transformation
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]

# Interaction marr*male, educ*male, educ*age, male*age
train[,c( 'marr-male','educ-male', 'educ-age', 'male-age')] <- cbind(
  train$marr * train$male,train$educ * train$male,train$educ * train$age, train$male * train$age
)

set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 6, nrow = k, dimnames = list(1:k,c('null','full', 'ridge', 'lasso', 'forward', 'backward')))
for (j in 1:k) {
  tr <- (ii != j)
  vali <- (ii == j)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  
  # lasso model
  lasso_hyper <- append(10 ^ seq(3,-2, -0.005),0)
  cv_lasso = cv.glmnet(as.matrix(x_tr), 
                       y_tr, 
                       type.measure = 'mse',
                       nfold = 10,
                       alpha = 1, lambda = lasso_hyper)
  print(paste(j, cv_lasso$lambda.min))
  lasso <- glmnet(x = x_tr, y = y_tr, lambda = cv_lasso$lambda.min, alpha = 1)
  mspe_lasso = log(mean((y_vali - predict(lasso, as.matrix(x_vali)))^2))
  
  mse[j,'lasso'] = mspe_lasso
  
}
inter5_mse = colMeans(mse)
inter5_mse

#--------------------------------------------------------------------


# Natural spline using command ns()

# Reset training data after previous interaction method
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]

#--------------------------------------------------------------------

# Choose best degree of freedom for ira
feature <- 'ira'
deg_cv = 15  #range of degree of freedom
set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse_ns <- matrix(ncol = deg_cv-1, nrow = k, dimnames = list(1:k, 2:deg_cv))
lasso_hyper <- append(10 ^ seq(4,-2, -0.01),0)
for (j in 1:k){
  vali <- (ii == j)
  tr <- (ii!=j)
  for (i in 2:deg_cv) {
    # Choose the optimal lambda
    cv_lasso = cv.glmnet(x = as.matrix(ns(train[tr,feature], df = i)), 
                         y = train[tr,'tw'], 
                         type.measure = 'mse',
                         nfold = 10,
                         alpha = 1, lambda = lasso_hyper)
    print(paste('j = ' ,j))
    print(cv_lasso$lambda.min)
    # plug in lambda and test different degree of freedom
    lasso <- glmnet(x = ns(train[tr, feature], df = i), lambda = cv_lasso$lambda.min, y = train[tr,'tw'], alpha = 1)
    # calculate log mse
    mspe_lasso = log(
      mean(
        (
          train[vali, 'tw'] - predict(lasso, as.matrix(ns(train[vali,feature], df = i))))^2
         )
           )
    mse_ns[j,i-1] <- mspe_lasso
  }
}
mse_ns <- colMeans(mse_ns)
mse_ns
min(mse_ns)  # min mse
which.min(mse_ns) + 1 # best degree of freedom

# scatter plot of tw and the feature
plot(train[,feature], train$tw, pch = 20, xlab = feature)
# line of best fit generated from the best degree of freedom
lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ ns(ira,df = which.min(mse_ns) + 1), data=train), newdata = train[order(train[,feature]),])
), col = 'orange', lwd = 2)
# Best degree
# tw - ira: 10


#--------------------------------------------------------------------

# A comparison between the fitness of polynomial and splines
plot(train[,feature], train$tw, pch = 20, xlab = feature)
# lines generated from the splines
lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ ns(ira,df = 10), data=train), newdata = train[order(train[,feature]),])
), col = 'red', lwd = 2)

lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ ns(ira,df = 1), data=train), newdata = train[order(train[,feature]),])
), col = 'yellow', lwd = 2)

lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ ns(ira,df = 5), data=train), newdata = train[order(train[,feature]),])
), col = 'orange', lwd = 2)
legend(x = 'topright', legend = c('spline1', 'spline5','spline10'),
       fill = c('orange','yellow','red'), cex = 0.75)

# lines generated from the polynomials
plot(train[,feature], train$tw, pch = 20, xlab = feature)

lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ poly(ira, 1), data=train), newdata = train[order(train[,feature]),])
), col = 'light blue', lwd = 2)

lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ poly(ira, 3), data=train), newdata = train[order(train[,feature]),])
), col = 'blue', lwd = 2)

lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ poly(ira, 5), data=train), newdata = train[order(train[,feature]),])
), col = 'navy', lwd = 2)

lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ poly(ira, 7), data=train), newdata = train[order(train[,feature]),])
), col = 'purple', lwd = 2)

legend(x = 'topright', legend = c( 'poly1', 'poly3', 'poly5', 'poly7'),
       fill = c('light blue','blue','navy', 'purple'), cex = 0.75)

#--------------------------------------------------------------------

# Choose best degree of freedom for nifa spline
feature <- 'nifa'
deg_cv = 15  #range of degree of freedom
mse_ns <- matrix(ncol = deg_cv-1, nrow = k, dimnames = list(1:k, 2:deg_cv))
lasso_hyper <- append(10 ^ seq(4,-2, -0.01),0)
for (j in 1:k){
  vali <- (ii == j)
  tr <- (ii!=j)
  for (i in 2:deg_cv) {
    # Choose the optimal lambda
    cv_lasso = cv.glmnet(x = as.matrix(ns(train[tr,feature], df = i)), 
                         y = train[tr,'tw'], 
                         type.measure = 'mse',
                         nfold = 10,
                         alpha = 1, lambda = lasso_hyper)
    print(paste('j = ' ,j))
    print(cv_lasso$lambda.min)
    # plug in lambda and test different degree of freedom
    lasso <- glmnet(x = ns(train[tr, feature], df = i), lambda = cv_lasso$lambda.min, y = train[tr,'tw'], alpha = 1)
    # calculate log mse
    mspe_lasso = log(
      mean(
        (
          train[vali, 'tw'] - predict(lasso, as.matrix(ns(train[vali,feature], df = i))))^2
      )
    )
    mse_ns[j,i-1] <- mspe_lasso
  }
}
mse_ns <- colMeans(mse_ns)
mse_ns
min(mse_ns)  # min mse
which.min(mse_ns) + 1 # best degree of freedom

# scatter plot of tw and the feature
plot(train[,feature], train$tw, pch = 20, xlab = feature)
# line of best fit generated from the best degree of freedom
lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ ns(nifa,df = which.min(mse_ns) + 1), data=train), newdata = train[order(train[,feature]),])
), col = 'orange', lwd = 2)
# Best degree
# tw - nifa: 12

#--------------------------------------------------------------------

# Choose best degree of freedom for inc spline
feature <- 'inc'
deg_cv = 15  #range of degree of freedom
mse_ns <- matrix(ncol = deg_cv-1, nrow = k, dimnames = list(1:k, 2:deg_cv))
lasso_hyper <- append(10 ^ seq(4,-2, -0.01),0)
for (j in 1:k){
  vali <- (ii == j)
  tr <- (ii!=j)
  for (i in 2:deg_cv) {
    # Choose the optimal lambda
    cv_lasso = cv.glmnet(x = as.matrix(ns(train[tr,feature], df = i)), 
                         y = train[tr,'tw'], 
                         type.measure = 'mse',
                         nfold = 10,
                         alpha = 1, lambda = lasso_hyper)
    print(paste('j = ' ,j))
    print(cv_lasso$lambda.min)
    # plug in lambda and test different degree of freedom
    lasso <- glmnet(x = ns(train[tr, feature], df = i), lambda = cv_lasso$lambda.min, y = train[tr,'tw'], alpha = 1)
    # calculate log mse
    mspe_lasso = log(
      mean(
        (
          train[vali, 'tw'] - predict(lasso, as.matrix(ns(train[vali,feature], df = i))))^2
      )
    )
    mse_ns[j,i-1] <- mspe_lasso
  }
}
mse_ns <- colMeans(mse_ns)
mse_ns
min(mse_ns)  # min mse
which.min(mse_ns) + 1 # best degree of freedom

# scatter plot of tw and the feature
plot(train[,feature], train$tw, pch = 20, xlab = feature)
# line of best fit generated from the best degree of freedom
lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ ns(inc,df = which.min(mse_ns) + 1), data=train), newdata = train[order(train[,feature]),])
), col = 'orange', lwd = 2)
# Best degree
# tw - inc: 14

#--------------------------------------------------------------------

# Choose best degree of freedom for hval spline 
feature <- 'hval'
deg_cv = 15  #range of degree of freedom
mse_ns <- matrix(ncol = deg_cv-1, nrow = k, dimnames = list(1:k, 2:deg_cv))
lasso_hyper <- append(10 ^ seq(4,-2, -0.01),0)
for (j in 1:k){
  vali <- (ii == j)
  tr <- (ii!=j)
  for (i in 2:deg_cv) {
    # Choose the optimal lambda
    cv_lasso = cv.glmnet(x = as.matrix(ns(train[tr,feature], df = i)), 
                         y = train[tr,'tw'], 
                         type.measure = 'mse',
                         nfold = 10,
                         alpha = 1, lambda = lasso_hyper)
    print(paste('j = ' ,j))
    print(cv_lasso$lambda.min)
    # plug in lambda and test different degree of freedom
    lasso <- glmnet(x = ns(train[tr, feature], df = i), lambda = cv_lasso$lambda.min, y = train[tr,'tw'], alpha = 1)
    # calculate log mse
    mspe_lasso = log(
      mean(
        (
          train[vali, 'tw'] - predict(lasso, as.matrix(ns(train[vali,feature], df = i))))^2
      )
    )
    mse_ns[j,i-1] <- mspe_lasso
  }
}
mse_ns <- colMeans(mse_ns)
mse_ns
min(mse_ns)  # min mse
which.min(mse_ns) + 1 # best degree of freedom

# scatter plot of tw and the feature
plot(train[,feature], train$tw, pch = 20, xlab = feature)
# line of best fit generated from the best degree of freedom
lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ ns(hval,df = which.min(mse_ns) + 1), data=train), newdata = train[order(train[,feature]),])
), col = 'orange', lwd = 2)
# Best degree
# tw - hval: 2

#--------------------------------------------------------------------

# Choose best degree of freedom for hmort spline
feature <- 'hmort'
deg_cv = 15  #range of degree of freedom
mse_ns <- matrix(ncol = deg_cv-1, nrow = k, dimnames = list(1:k, 2:deg_cv))
lasso_hyper <- append(10 ^ seq(4,-2, -0.01),0)
for (j in 1:k){
  vali <- (ii == j)
  tr <- (ii!=j)
  for (i in 5:deg_cv) {
    # Choose the optimal lambda
    cv_lasso = cv.glmnet(x = as.matrix(ns(train[tr,feature], df = i)), 
                         y = train[tr,'tw'], 
                         type.measure = 'mse',
                         nfold = 10,
                         alpha = 1, lambda = lasso_hyper)
    print(paste('j = ' ,j))
    print(cv_lasso$lambda.min)
    # plug in lambda and test different degree of freedom
    lasso <- glmnet(x = ns(train[tr, feature], df = i), lambda = cv_lasso$lambda.min, y = train[tr,'tw'], alpha = 1)
    # calculate log mse
    mspe_lasso = log(
      mean(
        (
          train[vali, 'tw'] - predict(lasso, as.matrix(ns(train[vali,feature], df = i))))^2
      )
    )
    mse_ns[j,i-1] <- mspe_lasso
  }
}
mse_ns <- colMeans(mse_ns)
mse_ns
min(mse_ns)  # min mse
which.min(mse_ns) + 1 # best degree of freedom

# scatter plot of tw and the feature
plot(train[,feature], train$tw, pch = 20, xlab = feature)
# line of best fit generated from the best degree of freedom
lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ ns(hmort,df = which.min(mse_ns) + 1), data=train), newdata = train[order(train[,feature]),])
), col = 'orange', lwd = 2)
# Best degree
# tw - hmort: 9

#--------------------------------------------------------------------
# Choose best degree of freedom for age spline
feature <- 'age'
deg_cv = 10  #range of degree of freedom
mse_ns <- matrix(ncol = deg_cv-1, nrow = k, dimnames = list(1:k, 2:deg_cv))
lasso_hyper <- append(10 ^ seq(4,-2, -0.01),0)
for (j in 1:k){
  vali <- (ii == j)
  tr <- (ii!=j)
  for (i in 2:deg_cv) {
    # Choose the optimal lambda
    cv_lasso = cv.glmnet(x = as.matrix(ns(train[tr,feature], df = i)), 
                         y = train[tr,'tw'], 
                         type.measure = 'mse',
                         nfold = 10,
                         alpha = 1, lambda = lasso_hyper)
    print(paste('j = ' ,j))
    print(cv_lasso$lambda.min)
    # plug in lambda and test different degree of freedom
    lasso <- glmnet(x = ns(train[tr, feature], df = i), lambda = cv_lasso$lambda.min, y = train[tr,'tw'], alpha = 1)
    # calculate log mse
    mspe_lasso = log(
      mean(
        (
          train[vali, 'tw'] - predict(lasso, as.matrix(ns(train[vali,feature], df = i))))^2
      )
    )
    mse_ns[j,i-1] <- mspe_lasso
  }
}
mse_ns <- colMeans(mse_ns)
mse_ns
min(mse_ns)  # min mse
which.min(mse_ns) + 1 # best degree of freedom

# scatter plot of tw and the feature
plot(train[,feature], train$tw, pch = 20, xlab = feature)
# line of best fit generated from the best degree of freedom
lines(cbind(
  train[order(train[,feature]), feature],
  predict(
    lm(tw ~ ns(age,df = which.min(mse_ns) + 1), data=train), newdata = train[order(train[,feature]),])
), col = 'orange', lwd = 2)
# Best degree
# tw - age: 5


#--------------------------------------------------------------------

# Reset the training set 
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]

# Create data frame that include the spline for ira and nifa using the 
# best df we find above. First only spline for ira and nifa
f <- as.formula(tw ~ ns(ira, 10) + ns(nifa, 12) + .)
y <- train$tw
train <- as.data.frame(model.matrix(f, train)[, -1])
train$tw <- y

set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 1, nrow = k, dimnames = list(1:k,1))

for (fold in 1:k) {
  
  tr <- (ii != fold)
  vali <- (ii == fold)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  # Choose the optimal lambda
  cv_gam = cv.glmnet(x = as.matrix(x_tr), 
                       y =y_tr, 
                       type.measure = 'mse',
                       nfold = 10,
                       alpha = 1, lambda = lasso_hyper)
  print(cv_gam$lambda.min)
  
  gam <- glmnet(x = as.matrix(x_tr), y = y_tr, lambda = cv_gam$lambda.min,  alpha = 1)
  mspe_gam <- log(mean((y_vali - predict(gam, as.matrix(x_vali)))^2))
  mse[fold, 1] <- mspe_gam

}
spline1_mse <- mean(mse)
spline1_mse


#--------------------------------------------------------------------

# Reset the training set 
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]

# Create data frame that include the spline for ira, nifa and age using the 
# best df we find above.
f <- as.formula(tw ~ ns(ira, 10) + ns(nifa, 12) + ns(age, 5) + .)
y <- train$tw
train <- as.data.frame(model.matrix(f, train)[, -1])
train$tw <- y

set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 1, nrow = k, dimnames = list(1:k,1))

for (fold in 1:k) {
  
  tr <- (ii != fold)
  vali <- (ii == fold)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  # Choose the optimal lambda
  cv_gam = cv.glmnet(x = as.matrix(x_tr), 
                     y =y_tr, 
                     type.measure = 'mse',
                     nfold = 10,
                     alpha = 1, lambda = lasso_hyper)
  print(cv_gam$lambda.min)
  
  gam <- glmnet(x = as.matrix(x_tr), y = y_tr, lambda = cv_gam$lambda.min,  alpha = 1)
  mspe_gam <- log(mean((y_vali - predict(gam, as.matrix(x_vali)))^2))
  mse[fold, 1] <- mspe_gam
  
}
spline2_mse <- mean(mse)
spline2_mse

#--------------------------------------------------------------------

# Reset the training set 
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]

# Create data frame that include the spline for ira, nifa and hval using the 
# best df we find above.
f <- as.formula(tw ~ ns(ira, 10) + ns(nifa, 12) + ns(hval, 2) + .)
y <- train$tw
train <- as.data.frame(model.matrix(f, train)[, -1])
train$tw <- y

n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 1, nrow = k, dimnames = list(1:k,1))

for (fold in 1:k) {
  
  tr <- (ii != fold)
  vali <- (ii == fold)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  # Choose the optimal lambda
  cv_gam = cv.glmnet(x = as.matrix(x_tr), 
                     y =y_tr, 
                     type.measure = 'mse',
                     nfold = 10,
                     alpha = 1, lambda = lasso_hyper)
  print(cv_gam$lambda.min)
  
  gam <- glmnet(x = as.matrix(x_tr), y = y_tr, lambda = cv_gam$lambda.min,  alpha = 1)
  mspe_gam <- log(mean((y_vali - predict(gam, as.matrix(x_vali)))^2))
  mse[fold, 1] <- mspe_gam
  
}
spline3_mse <- mean(mse)
spline3_mse

#--------------------------------------------------------------------

# Reset the training set 
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]

# Create data frame that include the spline for ira, nifa and hmort using the 
# best df we find above.
f <- as.formula(tw ~ ns(ira, 10) + ns(nifa, 12) + ns(hmort, 9) + .)
y <- train$tw
train <- as.data.frame(model.matrix(f, train)[, -1])
train$tw <- y

set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 1, nrow = k, dimnames = list(1:k,1))

for (fold in 1:k) {
  
  tr <- (ii != fold)
  vali <- (ii == fold)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  # Choose the optimal lambda
  cv_gam = cv.glmnet(x = as.matrix(x_tr), 
                     y =y_tr, 
                     type.measure = 'mse',
                     nfold = 10,
                     alpha = 1, lambda = lasso_hyper)
  print(cv_gam$lambda.min)
  
  gam <- glmnet(x = as.matrix(x_tr), y = y_tr, lambda = cv_gam$lambda.min,  alpha = 1)
  mspe_gam <- log(mean((y_vali - predict(gam, as.matrix(x_vali)))^2))
  mse[fold, 1] <- mspe_gam
  
}
spline4_mse <- mean(mse)
spline4_mse


#--------------------------------------------------------------------

# Reset the training set 
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]

# Create data frame that include the spline for ira, nifa and inc using the 
# best df we find above.
f <- as.formula(tw ~ ns(ira, 10) + ns(nifa, 12) + ns(inc, 15) + .)
y <- train$tw
train <- as.data.frame(model.matrix(f, train)[, -1])
train$tw <- y

set.seed(2022723)
n <- length(train$tw)
k <- 10
ii <- sample(rep(1:k, length = n))
mse <- matrix(ncol = 1, nrow = k, dimnames = list(1:k,1))

for (fold in 1:k) {
  
  tr <- (ii != fold)
  vali <- (ii == fold)
  x_tr <- subset(train[tr, ], select = -c(tw))
  y_tr = train[tr, 'tw']
  x_vali = subset(train[vali, ], select = -c(tw))
  y_vali = train[vali, 'tw']
  # Choose the optimal lambda
  cv_gam = cv.glmnet(x = as.matrix(x_tr), 
                     y =y_tr, 
                     type.measure = 'mse',
                     nfold = 10,
                     alpha = 1, lambda = lasso_hyper)
  print(cv_gam$lambda.min)
  
  gam <- glmnet(x = as.matrix(x_tr), y = y_tr, lambda = cv_gam$lambda.min,  alpha = 1)
  mspe_gam <- log(mean((y_vali - predict(gam, as.matrix(x_vali)))^2))
  mse[fold, 1] <- mspe_gam
  
}
spline5_mse <- mean(mse)
spline5_mse


#--------------------------------------------------------------------

# Final Model testing on test set
# Reset the training set 
train <- base_train[base_train$inc >= 0,]
train <- train[train$tw < 1700000 & (train$tw > -42000),]
test <- base_test

# Interaction marr*male, educ*male, educ*age, male*age
train[,c( 'marr-male','educ-male', 'educ-age', 'male-age')] <- cbind(
  train$marr * train$male,train$educ * train$male,train$educ * train$age, train$male * train$age
)

test[,c( 'marr-male','educ-male', 'educ-age', 'male-age')] <- cbind(
  test$marr * test$male, test$educ * test$male,test$educ * test$age, test$male * test$age
)


# Create data frame that include the spline for ira, nifa, hmort using the 
# best df we find above.

f <- as.formula(tw ~ ns(ira, 10) + ns(nifa, 12) + ns(hmort, 9) +.)
# train dataframe
y <- train$tw
train <- as.data.frame(model.matrix(f, train)[, -1])
train$tw <- y
# test data frame
y <- test$tw
test <- as.data.frame(model.matrix(f, test)[, -1])
test$tw <- y

# keep track of the train x/y set, test x/y set
x_train = subset(train, select = -c(tw))
y_train = train[['tw']]
x_test = subset(test, select = -c(tw))
y_test = test[['tw']]

# Choose the best lambda
lasso_hyper <- append(10^seq(3,-1,-0.005), 0)
cv_gam = cv.glmnet(x = as.matrix(x_train), 
                   y = y_train, 
                   type.measure = 'mse',
                   nfold = 10,
                   alpha = 1, lambda = lasso_hyper)
print(cv_gam$lambda.min)

gam <- glmnet(x = as.matrix(x_train), y = y_train, lambda = cv_gam$lambda.min,  alpha = 1)
mspe_gam <- log(mean((y_test - predict(gam, as.matrix(x_test)))^2))
mspe_gam

mspe_gam_tr <- log(mean((y_train - predict(gam, as.matrix(x_train)))^2))
mspe_gam_tr


# Random forest regression
grid <- expand.grid(.mtry = c( 9, 10, 11, 12))
control <- trainControl(method = 'cv', number = 5)

rf <- train(tw ~., data = df, method = 'rf', metric = 'RMSE', tuneGrid = grid, trControl = control)

print(rf)
# mtry  RMSE      Rsquared   MAE
# 9    45060.03  0.8268374  18913.54
# 10    44905.68  0.8276314  18913.10
# 11    44829.37  0.8277689  18899.09
# 12    44848.08  0.8274488  18944.73

# > log(44829.37^2)
# [1] 21.42124

#--------------------------------------------------------------------

# Predict hidden data
# get unhidden data
hidden <- read.table('Desktop/econ178/data_for_prediction.txt', head = T)[,-1]
hidden <- subset(hidden, select = -c(hequity, smcol))
hidden

# build model using all data we have
unhidden <- df
unhidden <- unhidden[unhidden$inc >= 0,]
unhidden <- unhidden[unhidden$tw < 1700000 & (unhidden$tw > -42000),]

# Interaction marr*male, educ*male, educ*age, male*age
unhidden[,c( 'marr-male','educ-male', 'educ-age', 'male-age')] <- cbind(
  unhidden$marr * unhidden$male,unhidden$educ * unhidden$male,
  unhidden$educ * unhidden$age, unhidden$male * unhidden$age
)

hidden[,c( 'marr-male','educ-male', 'educ-age', 'male-age')] <- cbind(
  hidden$marr * hidden$male, hidden$educ * hidden$male,
  hidden$educ * hidden$age, hidden$male * hidden$age
)


# Create data frame that include the spline for ira, nifa, hmort using the 
# best df we find above.

f <- as.formula(tw ~ ns(ira, 10) + ns(nifa, 12) + ns(hmort, 9) +.)
# unhidden dataframe
y <- unhidden$tw
unhidden <- as.data.frame(model.matrix(f, unhidden)[, -1])
unhidden$tw <- y
# hidden data frame
hidden['tw'] <- 1

hidden <- as.data.frame(model.matrix(f, hidden)[, -1])

# keep track of the train x/y set, test x/y set
x_train = subset(unhidden, select = -c(tw))
y_train = unhidden[['tw']]
x_test <- hidden

# Choose the best lambda
lasso_hyper <- append(10^seq(3,1,-0.0001), 0)
cv_gam = cv.glmnet(x = as.matrix(x_train), 
                   y = y_train, 
                   type.measure = 'mse',
                   nfold = 10,
                   alpha = 1, lambda = lasso_hyper)
print(cv_gam$lambda.min)

gam <- glmnet(x = as.matrix(x_train), y = y_train, lambda = cv_gam$lambda.min,  alpha = 1)
prediction <- predict(gam, as.matrix(x_test))

write.table(prediction, file = 'Desktop/econ178/my_predictions.txt')



