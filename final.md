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

movies <- bind_rows(movies_list)
head(movies, 10)
```

    ## # A tibble: 10 × 4
    ##    movie_id   movie_name                        description          final_genre
    ##    <chr>      <chr>                             <chr>                <chr>      
    ##  1 tt9114286  Black Panther: Wakanda Forever    The people of Wakan… action     
    ##  2 tt1630029  Avatar: The Way of Water          Jake Sully lives wi… action     
    ##  3 tt5884796  Plane                             A pilot finds himse… action     
    ##  4 tt6710474  Everything Everywhere All at Once A middle-aged Chine… action     
    ##  5 tt5433140  Fast X                            Dom Toretto and his… action     
    ##  6 tt10954600 Ant-Man and the Wasp: Quantumania Scott Lang and Hope… action     
    ##  7 tt9686790  Shotgun Wedding                   Darcy and Tom gathe… action     
    ##  8 tt12593682 Bullet Train                      Five assassins aboa… action     
    ##  9 tt1016150  All Quiet on the Western Front    A young German sold… action     
    ## 10 tt1745960  Top Gun: Maverick                 After thirty years,… action

When I look through the new dataframe “movies”, I noticed that some
movies are missing descriptions (which happens across genres.) Those
missing movie descriptions are indicated as “Add a Plot” rather than NA
or blank cells. Therefore, we will remove those movies so it wouldn’t
mess up the classification.

``` r
filtered_movies <- movies %>%
  filter(!grepl("Add a Plot", description))
```

Here, we will do some basic text analysis, such as removing stopwords,
punctuations, tokenizations, changing to lowercases

Before starting any text operations, I have noticed that some
descriptions don’t have the full descriptions in the original data and
at the end, it would say ” See full summary »“. Therefore, we will
remove them first.

``` r
filtered_movies %>%
  mutate(description = str_remove(description, "See full summary.*$"))
```

    ## # A tibble: 256,650 × 4
    ##    movie_id   movie_name                        description          final_genre
    ##    <chr>      <chr>                             <chr>                <chr>      
    ##  1 tt9114286  Black Panther: Wakanda Forever    The people of Wakan… action     
    ##  2 tt1630029  Avatar: The Way of Water          Jake Sully lives wi… action     
    ##  3 tt5884796  Plane                             A pilot finds himse… action     
    ##  4 tt6710474  Everything Everywhere All at Once A middle-aged Chine… action     
    ##  5 tt5433140  Fast X                            Dom Toretto and his… action     
    ##  6 tt10954600 Ant-Man and the Wasp: Quantumania Scott Lang and Hope… action     
    ##  7 tt9686790  Shotgun Wedding                   Darcy and Tom gathe… action     
    ##  8 tt12593682 Bullet Train                      Five assassins aboa… action     
    ##  9 tt1016150  All Quiet on the Western Front    A young German sold… action     
    ## 10 tt1745960  Top Gun: Maverick                 After thirty years,… action     
    ## # ℹ 256,640 more rows
