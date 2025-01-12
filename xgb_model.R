###############################################################################
# Install any missing packages
###############################################################################
# install.packages(c("dplyr", "caret", "fastDummies", "xgboost", "Matrix", 
#                    "doParallel", "ggplot2"))

###############################################################################
# Load Libraries
###############################################################################
library(dplyr)         # For data manipulation
library(caret)         # Machine learning utilities
library(fastDummies)   # One-hot encoding for categorical variables
library(xgboost)       # XGBoost modeling framework
library(Matrix)        # Sparse matrix operations
library(doParallel)    # Parallel processing
library(ggplot2)       # Visualization

###############################################################################
# Parallel Processing Setup
###############################################################################
# Create a parallel cluster using all but one core
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

###############################################################################
# Load Data
###############################################################################
train_data <- read.csv("fifa_data_train.csv")
test_data  <- read.csv("fifa_data_test.csv")

###############################################################################
# Merge Train & Test for Consistent Encoding
###############################################################################
train_data$dataset <- "train"
test_data$dataset  <- "test"

combined_data <- bind_rows(train_data, test_data)

###############################################################################
# One-Hot Encoding of Selected Categorical Columns
###############################################################################
categorical_cols <- c("work_rate", "body_type")

combined_data <- combined_data %>%
  dummy_cols(select_columns = categorical_cols,
             remove_first_dummy = FALSE)

# Remove original columns (already replaced by dummies) & other columns not needed
combined_data <- combined_data %>%
  select(
    -body_type,
    -work_rate,
    -long_name,
    -Value20
  )

# Move 'value_eur' column to the end, ensuring a neat column order
target_col  <- "value_eur"
all_cols    <- colnames(combined_data)
cols_reorder <- c(setdiff(all_cols, target_col), target_col)
combined_data <- combined_data[, cols_reorder]

# Split the combined data back into train & test
train_data <- combined_data %>% filter(dataset == "train") %>% select(-dataset)
test_data  <- combined_data %>% filter(dataset == "test")  %>% select(-dataset)

###############################################################################
# Log-Transform Target
###############################################################################
train_data$value_eur_log <- log(train_data$value_eur + 1)
test_data$value_eur_log  <- log(test_data$value_eur + 1)

###############################################################################
# Create Train/Validation Split
###############################################################################
set.seed(123)  
train_prop   <- 0.7
train_indices <- sample(seq_len(nrow(train_data)), size = train_prop * nrow(train_data))

train_split <- train_data[train_indices, ]
val_split   <- train_data[-train_indices, ]

###############################################################################
# Identify Columns to Scale vs. Skip
###############################################################################
emb_patterns <- c("^work_rate_", "^body_type_")
emb_columns  <- unlist(
  lapply(emb_patterns, function(pattern) {
    grep(pattern, names(train_split), value = TRUE)
  })
)

# binary colums should not be scalled
binary_columns <- c("preferred_foot")

target_col   <- "value_eur_log"
all_features <- setdiff(names(train_split), target_col)
cols_to_scale <- setdiff(all_features, c(emb_columns, binary_columns))

cat("Columns to scale:\n",    paste(cols_to_scale, collapse = ", "), "\n\n")
cat("Embedding columns:\n",   paste(emb_columns, collapse = ", "),   "\n\n")
cat("Binary columns:\n",      paste(binary_columns, collapse = ", "),"\n\n")

###############################################################################
# Separate Features & Targets (Train, Val, Test)
###############################################################################
x_train_num  <- train_split[, cols_to_scale, drop = FALSE]
x_train_emb  <- train_split[, emb_columns,   drop = FALSE]
x_train_bin  <- train_split[, binary_columns, drop = FALSE]
y_train      <- train_split[, target_col]

x_val_num  <- val_split[, cols_to_scale, drop = FALSE]
x_val_emb  <- val_split[, emb_columns,   drop = FALSE]
x_val_bin  <- val_split[, binary_columns, drop = FALSE]
y_val      <- val_split[, target_col]

x_test_num <- test_data[, cols_to_scale, drop = FALSE]
x_test_emb <- test_data[, emb_columns,   drop = FALSE]
x_test_bin <- test_data[, binary_columns, drop = FALSE]
y_test     <- test_data[, target_col]

###############################################################################
# Scale Numeric Columns
###############################################################################
preproc <- preProcess(x_train_num, method = c("center", "scale"))

x_train_num_scaled <- predict(preproc, x_train_num)
x_val_num_scaled   <- predict(preproc, x_val_num)
x_test_num_scaled  <- predict(preproc, x_test_num)

y_train_vector <- as.numeric(y_train)
y_val_vector   <- as.numeric(y_val)
y_test_vector  <- as.numeric(y_test)

###############################################################################
# Combine Scaled & Embedding Columns
###############################################################################
train_final <- cbind(x_train_num_scaled, x_train_bin, x_train_emb)
val_final   <- cbind(x_val_num_scaled,   x_val_bin,   x_val_emb)
test_final  <- cbind(x_test_num_scaled,  x_test_bin,  x_test_emb)

# Remove 'value_eur' to avoid leakage
train_final <- train_final[, !(colnames(train_final) %in% "value_eur")]
val_final   <- val_final[, !(colnames(val_final)   %in% "value_eur")]
test_final  <- test_final[, !(colnames(test_final) %in% "value_eur")]

###############################################################################
# Sanitize Column Names for Sparse Model Matrix
###############################################################################
colnames(train_final) <- gsub("/", ".", colnames(train_final))
colnames(val_final)   <- gsub("/", ".", colnames(val_final))
colnames(test_final)  <- gsub("/", ".", colnames(test_final))

colnames(train_final) <- make.names(colnames(train_final), unique = TRUE)
colnames(val_final)   <- make.names(colnames(val_final),   unique = TRUE)
colnames(test_final)  <- make.names(colnames(test_final),  unique = TRUE)

cat("Columns in train_final:\n")
print(colnames(train_final))

###############################################################################
# Convert to Sparse Matrices & DMatrix
###############################################################################
sparse_train <- sparse.model.matrix(~ . - 1, data = train_final)
sparse_val   <- sparse.model.matrix(~ . - 1, data = val_final)
sparse_test  <- sparse.model.matrix(~ . - 1, data = test_final)

dtrain <- xgb.DMatrix(data = sparse_train, label = y_train_vector)
dval   <- xgb.DMatrix(data = sparse_val,   label = y_val_vector)
dtest  <- xgb.DMatrix(data = sparse_test,  label = y_test_vector)

###############################################################################
# XGBoost Training with Early Stopping
###############################################################################
params <- list(
  objective        = "reg:squarederror",
  eval_metric      = "mae",
  eta              = 0.1,   # Learning rate
  max_depth        = 6,     # Tree depth
  subsample        = 0.8,   # Row sampling
  colsample_bytree = 0.8,   # Column sampling
  lambda           = 1,     # L2 regularization
  alpha            = 0      # L1 regularization
)

watchlist <- list(train = dtrain, eval = dval)

xgb_model <- xgb.train(
  params               = params,
  data                 = dtrain,
  nrounds             = 1000,   # Max number of boosting iterations
  watchlist           = watchlist,
  early_stopping_rounds = 10,
  print_every_n       = 10,
  maximize            = FALSE   # Minimizing MAE
)

###############################################################################
# Predictions & Evaluation
###############################################################################
xgb_pred_log <- predict(xgb_model, dtest)

# Convert log-scale predictions back to EUR
xgb_pred_eur    <- exp(xgb_pred_log) - 1
xgb_actual_eur  <- exp(y_test_vector) - 1
xgb_errors_eur  <- xgb_pred_eur - xgb_actual_eur

# Metrics in EUR
xgb_mae_eur  <- mean(abs(xgb_errors_eur))
xgb_mse_eur  <- mean(xgb_errors_eur^2)
xgb_rmse_eur <- sqrt(xgb_mse_eur)

# Metrics in Log Space
xgb_mae_log  <- mean(abs(xgb_pred_log - y_test_vector))
xgb_mse_log  <- mean((xgb_pred_log - y_test_vector)^2)
xgb_rmse_log <- sqrt(xgb_mse_log)

# R² in Log Space
xgb_r2 <- 1 - sum((xgb_pred_log - y_test_vector)^2) /
  sum((y_test_vector - mean(y_test_vector))^2)

cat("\n------------------- XGBoost Evaluation -------------------\n")
cat("Test MAE (log scale):", xgb_mae_log,  "\n")
cat("Test MAE (EUR):      ", xgb_mae_eur,  "\n")
cat("Test MSE (EUR):      ", xgb_mse_eur,  "\n")
cat("Test RMSE (EUR):     ", xgb_rmse_eur, "\n")
cat("Test R²:             ", xgb_r2,       "\n")
cat("----------------------------------------------------------\n\n")

###############################################################################
# Feature Importance
###############################################################################
importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix)

# Top 10 important features
xgb.plot.importance(importance_matrix[1:10, ])

###############################################################################
# Additional Advanced Visualizations
###############################################################################
# 1. Predicted vs Actual with residual-based color gradient
results_df <- data.frame(
  ActualValue     = xgb_actual_eur,
  PredictedValue  = xgb_pred_eur,
  Residual        = xgb_errors_eur
)

ggplot(results_df, aes(x = ActualValue, y = PredictedValue, color = Residual)) +
  geom_point(alpha = 0.7) +
  scale_color_gradient2(midpoint = 0, low = "blue", high = "red", mid = "green") +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed") +
  labs(
    title = "Predicted vs. Actual Market Values",
    x     = "Actual Value (EUR)",
    y     = "Predicted Value (EUR)",
    color = "Residuals"
  ) +
  theme_minimal()

# 2. Residual Distribution
ggplot(results_df, aes(x = Residual)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
  labs(
    title = "Residual Distribution",
    x     = "Residuals (EUR)",
    y     = "Frequency"
  ) +
  theme_minimal()

###############################################################################
# Stop Parallel Cluster
###############################################################################
stopCluster(cl)
registerDoSEQ()

