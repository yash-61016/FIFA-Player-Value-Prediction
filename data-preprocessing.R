install.packages(c("tidyverse", "lubridate", "modeest"))

library(tidyverse)
library(lubridate)
library(modeest)

fifa_data <- read_csv("/Users/yashpatel/Documents/dev/R_projects/ica/players_20.csv")

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

# Create 'years_at_club' by subtracting 'joined' from the reference date
reference_date <- ymd("2024-12-31")

fifa_data <- fifa_data %>%
  mutate(
    years_at_club = as.numeric(difftime(reference_date, joined, units="days"))
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

# List of goalkeeping attributes
goalkeeping_attributes <- c("gk_diving", "gk_handling", "gk_kicking", "gk_reflexes", "gk_speed", "gk_positioning")

# Assign zero to goalkeeping attributes for outfield players
fifa_data <- fifa_data %>%
  mutate(
    across(all_of(goalkeeping_attributes), ~ if_else(is_goalkeeper == 0,0,.))
  )

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
    forward_rating = rowSums(select(., all_of(forward_positions)), na.rm = TRUE),
    midfielder_positions = rowSums(select(., all_of(midfielder_positions)), na.rm = TRUE),
    defender_positions = rowSums(select(., all_of(defender_positions)), na.rm = TRUE),
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

