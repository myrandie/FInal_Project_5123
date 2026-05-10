Final Project
================
miranda_moe
2026-05-09

- [Packages](#packages)
- [Data Frames](#data-frames)

In this project, we will if we can build a classification model to learn
features of model descriptions to predict genre it represents. We will
use a dataset from Kaggle that has various characteristics of movies
grouped in different genres. Each genre has its own csv, and there are
16 genres in the dataset in total. The following is the link to original
dataset:

<https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre/data>

## Packages

Load the packages we need for this project.

``` r
knitr::opts_chunk$set(echo = TRUE)
library(tidymodels)
library(tm)
library(ggplot2)
library(tidyverse)
library(modeldata)
library(rpart.plot)
library(tidytext)
library(stringr)
library(tidyr)
library(caret)
library(tidytext)
library(lubridate)
library(tibble)
```

## Data Frames

Now, we will load the csvs. First, we would need to load every file that
ends with csv.

``` r
file_list <- list.files(pattern = "*.csv")
file_list
```

    ##  [1] "action.csv"    "adventure.csv" "animation.csv" "biography.csv"
    ##  [5] "crime.csv"     "family.csv"    "fantasy.csv"   "film-noir.csv"
    ##  [9] "history.csv"   "horror.csv"    "mystery.csv"   "romance.csv"  
    ## [13] "scifi.csv"     "sports.csv"    "thriller.csv"  "war.csv"

From raw data exploration, I have noticed that some movies have multiple
genre listed. To resolve the issue of multiple labeling, I only kept
movies where the CSV’s genre appeared somewhere in the movie’s genre
list. For example, a movie with genres listed as “Animation, Adventure,
Comedy” would be kept in animation.csv but excluded from adventure.csv
if “Adventure” was not the intended label. This ensures each movie is
only included when it genuinely belongs to that genre category. To
combine all 16 CSVs into a single dataframe, I used lapply() to loop
through each file. For each file, I extracted the genre label from the
filename by removing the .csv extension using gsub(), filtered rows
using str_detect() to keep only movies matching that genre, added a new
column final_genre to store the label, and selected only the relevant
columns (movie_id, movie_name, description, final_genre). All the movies
are now binded into one dataframe.

``` r
movies_list <- lapply(file_list, function(f) {
  genre_name <- gsub(".csv", "", f)
  df <- read_csv(f) |>
    filter(str_detect(str_to_lower(genre), genre_name)) |>
    mutate(final_genre = genre_name) |>
    select(movie_id, movie_name, description, final_genre)
  df
})
```

    ## Rows: 52452 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 25664 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 8419 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 8289 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 35852 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 17095 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 17163 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 986 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (10): movie_id, movie_name, certificate, runtime, genre, description, di...
    ## dbl  (4): year, rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 8996 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 36682 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 18960 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 52617 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 16557 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 5292 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 53365 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
    ## Rows: 9911 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (11): movie_id, movie_name, year, certificate, runtime, genre, descripti...
    ## dbl  (3): rating, votes, gross(in $)
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
movies <- bind_rows(movies_list)
```
