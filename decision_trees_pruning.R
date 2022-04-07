# Decision Trees and Pruning in R - DZone
# Source: https://dzone.com/articles/decision-trees-and-pruning-in-r

# load packages
library(dplyr)
library(ggplot2)
library(ISLR)
library(rpart)
library(rpart.plot)

# read dataset
data("Carseats")

head(Carseats)

skimr::skim(Carseats)

# convert `Sales` to binary (`High` for `Sales` > 7.5 - median, "Yes", "No")
carseats <- as_tibble(Carseats) %>%
     mutate(High = factor(if_else(Sales <= 7.5, "No", "Yes"))) %>%
     select(-Sales)

carseats

# count `High`
carseats %>% 
     count(High)

# `High` bar plot
carseats %>% 
     ggplot(aes(High, fill = High)) + 
     geom_bar() + 
     theme(legend.position = 'none')

# create train/test datasets
set.seed(11111)
sample_ind <- sample(nrow(carseats), nrow(carseats) * 0.70) # 70% - train, 30% - test
train <- carseats[sample_ind,]
test <- carseats[-sample_ind,]

# baseline
carseats_base_model <- rpart(High ~ ., data = train, method = "class",
                       control = rpart.control(cp = 0))

summary(carseats_base_model)

# plot decision tree
rpart.plot(carseats_base_model)

# examine the complexity plot
printcp(carseats_base_model)

plotcp(carseats_base_model)

# compute the accuracy of the pruned tree
compute_accuracy <- function(model, tbl) {
     
     tbl$pred <- predict(model, tbl, type = "class")
     base_accuracy <- mean(tbl$pred == tbl$High)
     
     return(base_accuracy)
}

train_base_accuracy <- compute_accuracy(model = carseats_base_model, train)
train_base_accuracy

test_base_accuracy <- compute_accuracy(model = carseats_base_model, test)
test_base_accuracy

# Preprunning (early stopping)
# Grow a tree with minsplit of 100 and max depth of 4
train$pred <- NULL
carseats_model_preprun <- rpart(High ~ ., data = train, method = "class", 
                          control = rpart.control(
                               cp = 0, 
                               maxdepth = 4, 
                               minsplit = 12)
                          )

rpart.plot(carseats_model_preprun)

# compute the accuracy of the pruned tree
train_acc_preprun <- compute_accuracy(carseats_model_preprun, train)
train_acc_preprun

# remove `pred` from test
test$pred <- NULL
test_acc_preprun <- compute_accuracy(carseats_model_preprun, test)
test_acc_preprun

# Postprunning
# Prune the carseats_base_model based on the optimal cp value
carseats_model_pruned <- prune(carseats_base_model, cp = 0.0227273)

rpart.plot(carseats_model_pruned)

# compute the accuracy of the pruned tree
test$pred <- NULL
test_acc_postprun <- compute_accuracy(carseats_model_pruned, test)
data.frame(test_base_accuracy, test_acc_preprun, test_acc_postprun)
