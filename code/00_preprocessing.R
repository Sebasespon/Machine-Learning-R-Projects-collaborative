# title: "Machine Learning End to End"
# authors: Gabriel Blanco, Sebastian Esponda
# date: 04 december 2021
# output: html_document

# Libraries ------------------------------------------------------------------
library(dplyr)

# Read and sanity check
datapath <- "data/raw/bank-full.csv"

raw_data <- read.csv(
    datapath,
    sep = ";"
)

glimpse(raw_data)

# Make target name meaningful
raw_data <- raw_data  %>%
  rename("target" = "y")

# Check for NA and duplicated data
funModeling::df_status(raw_data)

# Everything seems quite in order. Now we split the data before
# going to the EDA stage
set.seed(123)
n_observations <- nrow(raw_data)

train_size <- floor(0.75 * n_observations)
train_index <- sample(
    seq_len(n_observations),
    size = train_size,
    replace = FALSE
    )

train <- raw_data[train_index, ]
test <- raw_data[-train_index, ]

write.csv(train, "data/clean/train.csv", row.names = FALSE)
write.csv(test, "data/clean/test.csv", row.names = FALSE)
