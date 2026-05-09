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

Now, we will load the csvs.

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
