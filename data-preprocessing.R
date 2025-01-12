# install.packages(c("tidyverse", "lubridate", "modeest","fastDummies", "caret", "corrplot"))

library(tidyverse)
library(lubridate)
library(modeest)
library(fastDummies)
library(caret)
library(corrplot)

fifa_data <- read_csv("/Users/yashpatel/Documents/dev/R_projects/ica/players_19.csv")
#fifa_data <- read_csv("C:/Users/Yash/Desktop/Development/fifa AML/dataset/players_19.csv")

# View the first few rows
head(fifa_data)

# Check the structure of the data
str(fifa_data)

# Get a summary of numerical columns
summary(fifa_data)

# Calculate total missing values per column
missing_values <- colSums(is.na(fifa_data))

# Calculate percentage of missing values per column
missing_percentage <- (missing_values / nrow(fifa_data)) * 100

# Create a data frame of missing value statistics
missing_data <- data.frame(
  Column = names(missing_values),
  TotalMissing = missing_values,
  PercentMissing = missing_percentage
)

# View columns with missing data
missing_data_filtered <- missing_data %>% 
  filter(TotalMissing > 0)

print(missing_data_filtered)

# Drop selected columns
fifa_data <- fifa_data %>% 
  select(
    -c(
      player_tags,
      team_jersey_number,
      loaned_from,
      nation_position,
      nation_jersey_number,
      player_traits
      )
    )

# Function to calculate mode
get_mode <- function(v) {
  # Remove NA values
  v <- na.omit(v)
  # Return the most frequent value
  mfv(v, method = "mfv")
}

# Calculate median of 'release_clause_eur' excluding NA
median_release_clause <- median(fifa_data$release_clause_eur, na.rm=TRUE)
# Impute missing 'release_clause_eur' with median
fifa_data <- fifa_data %>%
  mutate(
    release_clause_eur = if_else(
      is.na(release_clause_eur), 
      median_release_clause, 
      release_clause_eur
    ),
    has_release_clause = if_else(!is.na(release_clause_eur), 1, 0)
  )

# Calculate mode of 'contract_valid_until'
mode_contract_valid_until <- get_mode(fifa_data$contract_valid_until)

# Impute missing 'contract_valid_until' with mode
fifa_data <- fifa_data %>%
  mutate(
    contract_valid_until = if_else(
      is.na(contract_valid_until),
      mode_contract_valid_until,
      contract_valid_until
    )
  )

# Impute 'team_position' by extracting the first position from 'player_positions'
fifa_data <- fifa_data %>%
  mutate(
    team_position = if_else(
      is.na(team_position),
      str_split(player_positions, pattern = ",") %>%
        map_chr(~ .x[1] %>% str_trim()),
      team_position
    )
  )

# Checking date type
head(fifa_data$joined, 20)

# Calculate the median join date
median_join_date <- median(fifa_data$joined, na.rm=TRUE)

fifa_data <- fifa_data %>%
  mutate(
    joined = if_else(
      is.na(joined),
      median_join_date,
      joined
    )
  )

# Create 'days_at_club' by subtracting 'joined' from the reference date
reference_date <- ymd("2024-12-31")

fifa_data <- fifa_data %>%
  mutate(
    days_at_club = as.numeric(difftime(reference_date, joined, units="days"))
  )

# Create 'is_goalkeeper' as a binary variable
fifa_data <- fifa_data %>%
  mutate(
    is_goalkeeper = if_else(
      team_position %in% c("GK", "Goalkeeper"),
      1,
      0
    )
  )

# List of outfield attributes
outfield_attributes <- c("pace", "shooting", "passing", "dribbling", "defending", "physic")

# Assign zero to outfield attributes for goalkeeper
fifa_data <- fifa_data %>%
  mutate(
    across(all_of(outfield_attributes), ~ if_else(is_goalkeeper == 1, 0, .))
  )

# Removing gk_* fields because of redundancy
fifa_data <- fifa_data %>%
  select(-c(gk_diving, gk_handling, gk_kicking, gk_reflexes, gk_speed, gk_positioning))


# List of position ratings
position_ratings <- c(
  "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw",
  "lam", "cam", "ram", "lm", "lcm", "cm", "rcm",
  "rm", "lwb", "ldm", "cdm", "rdm", "rwb",
  "lb", "lcb", "cb", "rcb", "rb"
)

# Function to extract numeric rating
extract_numeric_rating <- function(x) {
  x <- trimws(x)
  if (x == "" || x == " " || x == "--" || is.na(x)) {
    return(NA)
  }
  num <- as.numeric(sub("^(\\d+).*", "\\1", x))
  return(num)
}

# Apply the function to position rating columns
fifa_data <- fifa_data %>%
  mutate(across(all_of(position_ratings), ~ sapply(., extract_numeric_rating)))

# Replace NA with zeros
fifa_data <- fifa_data %>%
  mutate(across(all_of(position_ratings), ~ replace_na(., 0)))

fifa_data <- fifa_data %>%
  mutate(is_goalkeeper = as.numeric(as.character(is_goalkeeper)))

# Assign zero to position ratings for goalkeepers
fifa_data <- fifa_data %>%
  mutate(
    across(all_of(position_ratings), ~ if_else(is_goalkeeper == 1, 0, .))
  )

sapply(fifa_data[position_ratings], function(x) sum(is.na(x)))

# Impute remaining NAs in outfield attributes with median
fifa_data <- fifa_data %>%
  mutate(
    pace = if_else(is.na(pace), median(pace, na.rm = TRUE), pace),
    shooting = if_else(is.na(shooting), median(shooting, na.rm = TRUE), shooting), 
    passing = if_else(is.na(passing), median(passing, na.rm = TRUE), passing), 
    dribbling = if_else(is.na(dribbling), median(dribbling, na.rm = TRUE), dribbling), 
    defending = if_else(is.na(defending), median(defending, na.rm = TRUE), defending), 
    physic = if_else(is.na(physic), median(physic, na.rm = TRUE), physic), 
  )

## Feature Engineering
# Define position categories
forward_positions <- c("ls", "st", "rs", "lw", "lf", "cf", "rf", "rw")
midfielder_positions <- c("lam", "cam", "ram", "lm", "lcm", "cm", "rcm", "rm")
defender_positions <-c("lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb", "cb", "rcb", "rb")

fifa_data <- fifa_data %>%
  mutate(
    forward_positions = rowSums(select(., all_of(forward_positions)), na.rm = TRUE),
    midfielder_positions = rowSums(select(., all_of(midfielder_positions)), na.rm = TRUE),
    defender_positions = rowSums(select(., all_of(defender_positions)), na.rm = TRUE),
  )

# Cleaning body type
unique(fifa_data$body_type)

fifa_data <- fifa_data %>%
  mutate(
    body_type = case_when(
      body_type == "Akinfenwa" ~ "Stocky",
      body_type == "C. Ronaldo" ~ "Lean",
      body_type == "Courtois" ~ "Normal",
      body_type == "Messi" ~ "Lean",
      body_type == "Neymar" ~ "Lean",
      body_type == "Shaqiri" ~ "Stocky",
      body_type == "PLAYER_BODY_TYPE_25" ~ "Normal",
      TRUE ~ body_type
    )
  )


# List of categorical columns to convert
categorical_columns <- c("preferred_foot", "team_position", "work_rate", "body_type", 
                         "player_positions", "is_goalkeeper")

# Convert to factors
fifa_data <- fifa_data %>%
  mutate(
    across(all_of(categorical_columns), as.factor)
  )

# Check for remaining missing values
remaining_missing <- colSums(is.na(fifa_data))
print(remaining_missing[remaining_missing > 0])

# List of columns to remove
columns_to_remove <- c(
  "sofifa_id", "player_url", "short_name", "long_name", "dob",
  "real_face", "joined", "player_positions"
)

# Remove the columns
fifa_data <- fifa_data %>%
  select(-all_of(columns_to_remove))

# Calculate club frequencies
club_frequency <- fifa_data %>%
  group_by(club) %>%
  summarise(freq = n()) %>%
  ungroup()

# Merge frequencies back into the dataset
fifa_data <- fifa_data %>%
  left_join(club_frequency, by = "club") %>%
  rename(club_frequency = freq)

# Remove the 'club' column
fifa_data <- fifa_data %>%
  select(-club)

# Calculate nationality frequencies
nationality_frequency <- fifa_data %>%
  group_by(nationality) %>%
  summarise(freq = n()) %>%
  ungroup()

# Merge frequencies back into the dataset
fifa_data <- fifa_data %>%
  left_join(nationality_frequency, by = "nationality") %>%
  rename(nationality_frequency = freq)

# Remove the 'nationality' column
fifa_data <- fifa_data %>%
  select(-nationality)

# Define the reference year
current_year <- 2024

# Create 'contract_remaining_years'
fifa_data <- fifa_data %>%
  mutate(
    contract_remaining_years = if_else(
      contract_valid_until >= current_year,
      contract_valid_until - current_year,
      0  # Set to 0 if the year is in the past
    )
  )

# Remove 'contract_valid_until'
fifa_data <- fifa_data %>%
  select(-contract_valid_until)


# One-hot encode categorical variables
fifa_data <- fifa_data %>%
 dummy_cols(select_columns = c("preferred_foot", "work_rate", "body_type", "team_position"),
            remove_first_dummy = FALSE)

# Ensure binary variables are numeric
fifa_data <- fifa_data %>%
  mutate(
    is_goalkeeper = as.numeric(as.character(is_goalkeeper)),
    has_release_clause = as.numeric(as.character(has_release_clause))
  )


# Select numerical features
numerical_features <- fifa_data %>%
  select(where(is.numeric)) %>%
  select(-value_eur)  # Exclude target variable for now


# Identify zero-variance predictors
nzv_cols <- nearZeroVar(numerical_features, saveMetrics = TRUE)

# Print the summary of these features
print(nzv_cols)

# Filter out zero or near-zero variance features
numerical_features_cleaned <- numerical_features[, !nzv_cols$nzv]

# Compute the correlation matrix without zero-variance features
correlation_matrix <- cor(numerical_features_cleaned, use = "pairwise.complete.obs")

# Visualize correlation matrix
corrplot(correlation_matrix, method = "color", tl.cex = 0.5)

# Attempt findCorrelation
high_corr_pairs <- findCorrelation(correlation_matrix, cutoff = 0.9, names = TRUE)
print(high_corr_pairs)

# keeping essential feature
essential_features <- c("passing")
columns_to_remove <- setdiff(high_corr_pairs, essential_features)

# Removing aggregated colums to avoid confusing the model, increase training time, or risk of overfitting.
fifa_data <- fifa_data %>%
  select(
    -all_of(forward_positions),
    -all_of(midfielder_positions),
    -all_of(defender_positions)
  )

str(fifa_data[, sort(names(fifa_data))], list.len = ncol(fifa_data))

# Remove original factor columns after one-hot encoding
fifa_data <- fifa_data %>%
  select(-body_type, -preferred_foot, -work_rate, -team_position)

# Check missing values
colSums(is.na(fifa_data))


columns_to_clean <- c(
  "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy", 
  "attacking_short_passing", "attacking_volleys", "defending_marking", 
  "defending_standing_tackle", "defending_sliding_tackle", 
  "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking", 
  "goalkeeping_positioning", "goalkeeping_reflexes", "mentality_aggression", 
  "mentality_composure", "mentality_interceptions", "mentality_penalties", 
  "mentality_positioning", "mentality_vision", "movement_acceleration", 
  "movement_agility", "movement_balance", "movement_reactions", 
  "movement_sprint_speed", "power_jumping", "power_long_shots", 
  "power_shot_power", "power_stamina", "power_strength", 
  "skill_ball_control", "skill_curve", "skill_dribbling", 
  "skill_fk_accuracy", "skill_long_passing"
)

# Get unique values for each column
unique_values <- lapply(columns_to_clean, function(col) {
  unique(fifa_data[[col]])
})

# Name the list with the column names
names(unique_values) <- columns_to_clean

# Print unique values for each column
unique_values

extract_rating <- function(x) {
  # Trim whitespace
  x <- trimws(x)
  
  # If the field is empty or just "--", return NA
  if (x == "" || x == "--") {
    return(NA_real_)
  }
  
  # Check if there's a plus or minus
  if (grepl("\\+", x)) {
    # Split at '+' sign
    parts <- strsplit(x, "\\+")[[1]]
    base_val <- as.numeric(parts[1])
    increment <- as.numeric(parts[2])
    return(base_val + increment)
  } else if (grepl("-", x)) {
    # Split at '-' sign
    parts <- strsplit(x, "-")[[1]]
    base_val <- as.numeric(parts[1])
    decrement <- as.numeric(parts[2])
    return(base_val - decrement)
  } else {
    # No plus or minus, just convert directly
    return(as.numeric(x))
  }
}

# Example usage on a vector:
test_values <- c("82+2", "82-2", "90", "", "--")
sapply(test_values, extract_rating)
# [1] 84 80 90 NA NA

fifa_data <- fifa_data %>%
  mutate(across(all_of(columns_to_clean), ~ sapply(., extract_rating)))

fifa_data <- fifa_data %>%
  mutate(across(where(is.numeric), as.numeric))


str(fifa_data[, sort(names(fifa_data))], list.len = ncol(fifa_data))

# Set a seed for reproducibility
set.seed(123)

# Determine the indices for training
train_indices <- sample(seq_len(nrow(fifa_data)), size = 0.8 * nrow(fifa_data))

# Split into training and testing sets
train_data <- fifa_data[train_indices, ]
test_data <- fifa_data[-train_indices, ]

# Ensure 'value_eur' is the last column
target_col <- "value_eur"
all_cols <- colnames(train_data)
cols_reordered <- c(setdiff(all_cols, target_col), target_col)

train_data <- train_data[, cols_reordered]
test_data <- test_data[, cols_reordered]

# Save the training and testing sets to CSV
write.csv(train_data, "fifa_data_train.csv", row.names = FALSE)
write.csv(test_data, "fifa_data_test.csv", row.names = FALSE)
