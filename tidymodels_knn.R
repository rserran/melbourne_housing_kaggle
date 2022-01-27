# Melbourne Housing Dataset - Kaggle
# Applying knn regression
# Source: https://ubc-dsci.github.io/introduction-to-datascience/regression1.html

# load packages
suppressMessages(library(tidyverse))
suppressMessages(library(tidymodels))

# read dataset
housing <- readr::read_csv('./melbourne_housing_imp.csv') %>% 
     janitor::clean_names()

head(housing)

skimr::skim(housing)

# split dataset
set.seed(2022)
split <- initial_split(housing, prop = 0.8, strata = price)
train <- training(split)
test <- testing(split)

# create k-folds
set.seed(1048)
folds <- vfold_cv(train, v = 10, strata = price)

# create recipe
rec <- recipe(price ~ distance, data = train) %>% 
     step_normalize(distance)

rec2 <- recipe(price ~ distance + rooms, data = train) %>% 
     step_normalize(all_predictors())

knn_spec <- nearest_neighbor(
     weight_func = 'gaussian', 
     neighbors = tune()
) %>% 
     set_engine('kknn') %>% 
     set_mode('regression')

# create workflow
wkflw <- workflow() %>% 
     add_recipe(rec) %>% 
     add_model(knn_spec)

wkflw

# create workflow with rec2
wkflw2 <- workflow() %>% 
     add_recipe(rec2) %>% 
     add_model(knn_spec)

wkflw2

# create tuning grid
grid <- tibble(
     neighbors = seq(1, 400, by = 5)
)

# setup parallel processing
set.seed(1883)
doParallel::registerDoParallel(9)
foreach::getDoParWorkers()

## wkflw (price ~ distance)
# execute hyperparameter tuning
tictoc::tic()
results <- wkflw %>% 
     tune_grid(
          resamples = folds, 
          grid = grid
     )
tictoc::toc()

results

# show best model
show_best(results, metric = 'rmse')

# plot results
autoplot(results)

# evaluating on test set
final_knn <- wkflw %>%
     finalize_workflow(select_best(results))

final_knn

# last fit
knn_fit <- last_fit(final_knn, split)
knn_fit

# show metrics
collect_metrics(knn_fit)

# plot predictions v. observed values
collect_predictions(knn_fit) %>%
     ggplot(aes(price, .pred)) +
     geom_abline(lty = 2, color = "red", size = 1) +
     geom_point(alpha = 0.5, color = "steelblue")

## wkflw2 (price ~ distance + rooms)
# execute hyperparameter tuning
tictoc::tic()
results2 <- wkflw2 %>% 
     tune_grid(
          resamples = folds, 
          grid = grid
     )
tictoc::toc()

results2

# show best model
show_best(results2, metric = 'rmse')

# plot results
autoplot(results2)

# evaluating on test set
final_knn2 <- wkflw2 %>%
     finalize_workflow(select_best(results2))

final_knn2

# last fit
knn_fit2 <- last_fit(final_knn2, split)
knn_fit2

# show metrics
collect_metrics(knn_fit2)

# plot predictions v. observed values
collect_predictions(knn_fit2) %>%
     ggplot(aes(price, .pred)) +
     geom_abline(lty = 2, color = "red", size = 1) +
     geom_point(alpha = 0.5, color = "steelblue")

doParallel::stopImplicitCluster()