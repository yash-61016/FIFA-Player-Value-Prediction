# Load Required Libraries
# Install any missing packages first using install.packages("packageName")
library(tidyverse)
library(lubridate)
library(modeest)
library(fastDummies)
library(caret)
library(corrplot)
library(keras)

# 1. Load the Data
train_data <- read.csv("/Users/yashpatel/Documents/dev/R_projects/ica/fifa_data_train.csv")
test_data  <- read.csv("/Users/yashpatel/Documents/dev/R_projects/ica/fifa_data_test.csv")

train_data$value_eur_log <- log(train_data$value_eur + 1)
test_data$value_eur_log  <- log(test_data$value_eur + 1)

# 2. Split the Training Data into Training and Validation Sets
set.seed(123)  # For reproducibility

# Define the proportion for training vs. validation
train_prop <- 0.7
train_indices <- sample(seq_len(nrow(train_data)), size = train_prop * nrow(train_data))

train_split <- train_data[train_indices, ]
val_split   <- train_data[-train_indices, ]

# 3. Identify Columns to Scale and Columns to Skip
# Define binary columns that should not be scaled
binary_columns <- c("has_release_clause", "is_goalkeeper")

# Define patterns for one-hot encoded columns
one_hot_patterns <- c("^preferred_foot_", "^work_rate_", "^body_type_", "^team_position_")

# Use grep to find all one-hot encoded columns based on their prefixes
one_hot_columns <- unlist(lapply(one_hot_patterns, function(pattern) {
  grep(pattern, names(train_split), value = TRUE)
}))

# Combine all columns to skip scaling
cols_to_skip <- c(binary_columns, one_hot_columns)

# Define the target variable
target_col <- "value_eur_log"

# All feature columns excluding the target
all_features <- setdiff(names(train_split), target_col)

# Columns to scale: all numeric columns not in cols_to_skip
cols_to_scale <- setdiff(all_features, cols_to_skip)

# Verify the columns
cat("Columns to scale:\n")
print(cols_to_scale)
cat("\nColumns to skip scaling:\n")
print(cols_to_skip)

# 4. Separate Features and Targets
# Training Set
x_train <- train_split[, cols_to_scale]
y_train <- train_split[, target_col]

x_train_skip <- train_split[, cols_to_skip]

# Validation Set
x_val <- val_split[, cols_to_scale]
y_val <- val_split[, target_col]

x_val_skip <- val_split[, cols_to_skip]

# Test Set
x_test <- test_data[, cols_to_scale]
y_test <- test_data[, target_col]

x_test_skip <- test_data[, cols_to_skip]

# 5. Scale the Data Using caret's preProcess
# Fit the scaler on the training set's columns to scale
preproc <- preProcess(x_train, method = c("center", "scale"))

# Apply the scaler to training, validation, and test sets
x_train_scaled <- predict(preproc, x_train)
x_val_scaled   <- predict(preproc, x_val)
x_test_scaled  <- predict(preproc, x_test)

# 6. Combine Scaled and Skipped Columns Back Together
# For Training Set
train_final <- cbind(x_train_scaled, x_train_skip)

# For Validation Set
val_final <- cbind(x_val_scaled, x_val_skip)

# For Test Set
test_final <- cbind(x_test_scaled, x_test_skip)

# Ensure that the column order is consistent across all datasets
common_columns <- names(train_final)
val_final <- val_final[, common_columns]
test_final <- test_final[, common_columns]

# 7. Convert Data to Matrix Format for Keras
x_train_matrix <- as.matrix(train_final)
x_val_matrix   <- as.matrix(val_final)
x_test_matrix  <- as.matrix(test_final)

y_train_vector <- as.numeric(y_train)
y_val_vector   <- as.numeric(y_val)
y_test_vector  <- as.numeric(y_test)

# Clear any old session
k_clear_session()

# Build a deeper model
model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu", input_shape = ncol(x_train_matrix)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")  # For regression

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "mse",
  metrics = c("mae")
)

# Early stopping callback: stops training if val_loss doesn't improve after 5 epochs
early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 5,
  restore_best_weights = TRUE  # This will roll back to the best model weights
)

history <- model %>% fit(
  x_train_matrix, y_train,
  validation_data = list(x_val_matrix, y_val),
  epochs = 100,         # We set a high max; might stop earlier
  batch_size = 32,
  verbose = 1
)


# 11. Visualize Training and Validation Loss and Metrics
# Plot Training & Validation Loss (MSE)
plot(history)  # Keras built-in summary

# Alternatively, ggplot approach:
history_df <- data.frame(
  epoch = seq_along(history$metrics$loss),
  loss = history$metrics$loss,
  val_loss = history$metrics$val_loss,
  mae = history$metrics$mean_absolute_error,
  val_mae = history$metrics$val_mean_absolute_error
)

library(ggplot2)
ggplot(history_df, aes(x = epoch)) +
  geom_line(aes(y = loss, color = "Train Loss")) +
  geom_line(aes(y = val_loss, color = "Val Loss")) +
  theme_minimal() +
  ggtitle("Training vs. Validation Loss") +
  ylab("MSE (log-scale)") +
  scale_color_manual(values = c("blue", "red"))


# 12. Evaluate the Model on Validation and Test Sets
# Evaluate on the test set in log-scale
test_scores_log <- model %>% evaluate(x_test_matrix, y_test, verbose = 0)
cat("Test MSE (log scale):", test_scores_log["loss"], "\n")
cat("Test MAE (log scale):", test_scores_log["mae"], "\n\n")

# Predictions in log scale
test_pred_log <- model %>% predict(x_test_matrix)

# Convert from log(value+1) back to original scale
test_pred_value <- exp(test_pred_log) - 1
actual_test_value <- exp(y_test) - 1

# Let's compute MAE and MSE in the original euro scale:
errors <- test_pred_value - actual_test_value
mae_eur <- mean(abs(errors))
mse_eur <- mean(errors^2)
cat("Test MAE in euros:", mae_eur, "\n")
cat("Test MSE in euros:", mse_eur, "\n")

# (Optional) A quick percentage error if you want a sense of relative error:
mape <- mean(abs(errors) / (actual_test_value + 1e-9)) * 100
cat("Test MAPE (%):", mape, "\n")


# 13. Predict and Visualize Predicted vs. Actual Values on Test Set
predictions <- model %>% predict(x_test_matrix)

# Convert predictions and actuals to vectors
predictions_vector <- as.vector(predictions)
actuals_vector     <- y_test_vector

# Scatter Plot: Predicted vs. Actual Values
ggplot(data = NULL, aes(x = actuals_vector, y = predictions_vector)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs. Actual Player Market Values",
    x = "Actual Value (EUR)",
    y = "Predicted Value (EUR)"
  ) +
  theme_minimal()

# Optional: Residual Plot
residuals <- actuals_vector - predictions_vector
ggplot(data = NULL, aes(x = predictions_vector, y = residuals)) +
  geom_point(alpha = 0.5, color = "purple") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Residuals Plot",
    x = "Predicted Value (EUR)",
    y = "Residuals (Actual - Predicted)"
  ) +
  theme_minimal()




# For a simpler approach, we'll do everything in one go with caret or randomForest.
# We'll compare the same train_split vs. val_split logic, or sometimes you do cross-validation.

# Let's do a small random forest on the training split
# We'll combine scaled numeric + unscaled categorical for the RF as well
# (RF typically handles the scale less critically, but let's stay consistent).

library(randomForest) 
df_rf_train <- data.frame(train_final, value_eur_log = y_train)
df_rf_val   <- data.frame(val_final,   value_eur_log = y_val)

# Train a random forest (basic example)
set.seed(123)
rf_model <- randomForest(
  value_eur_log ~ ., 
  data = df_rf_train,
  ntree = 100,
  mtry = 10,  # tune as needed
  importance = TRUE
)

# Evaluate on validation set
val_pred_log_rf <- predict(rf_model, newdata = df_rf_val)
val_error_log_rf <- val_pred_log_rf - df_rf_val$value_eur_log
val_mae_log_rf   <- mean(abs(val_error_log_rf))
cat("RF Validation MAE (log scale):", val_mae_log_rf, "\n")

# Convert from log to EUR
val_pred_rf_eur  <- exp(val_pred_log_rf) - 1
val_actual_eur   <- exp(df_rf_val$value_eur_log) - 1
val_errors_eur   <- val_pred_rf_eur - val_actual_eur
val_mae_eur_rf   <- mean(abs(val_errors_eur))
cat("RF Validation MAE (EUR):", val_mae_eur_rf, "\n")

# Then do the same for test_data if you like
df_rf_test <- data.frame(test_final, value_eur_log = y_test)
test_pred_log_rf <- predict(rf_model, newdata = df_rf_test)
test_pred_rf_eur <- exp(test_pred_log_rf) - 1
test_actual_eur  <- exp(df_rf_test$value_eur_log) - 1
test_mae_eur_rf  <- mean(abs(test_pred_rf_eur - test_actual_eur))
cat("RF Test MAE (EUR):", test_mae_eur_rf, "\n")





# 14. Save the Model and Preprocessing Object (Optional)
# This is useful for deploying the model later
save_model_hdf5(model, "fifa_player_value_model.h5")
saveRDS(preproc, "scaling_parameters.rds")

# To load the model and preproc later:
# loaded_model <- load_model_hdf5("fifa_player_value_model.h5")
# loaded_preproc <- readRDS("scaling_parameters.rds")
