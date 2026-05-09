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
library(tidymodels)
```

    ## ── Attaching packages ────────────────────────────────────── tidymodels 1.4.1 ──

    ## ✔ broom        1.0.12     ✔ recipes      1.3.1 
    ## ✔ dials        1.4.2      ✔ rsample      1.3.2 
    ## ✔ dplyr        1.1.4      ✔ tailor       0.1.0 
    ## ✔ ggplot2      3.5.2      ✔ tidyr        1.3.1 
    ## ✔ infer        1.1.0      ✔ tune         2.0.1 
    ## ✔ modeldata    1.5.1      ✔ workflows    1.3.0 
    ## ✔ parsnip      1.4.1      ✔ workflowsets 1.1.1 
    ## ✔ purrr        1.2.1      ✔ yardstick    1.3.2

    ## Warning: package 'broom' was built under R version 4.4.3

    ## Warning: package 'infer' was built under R version 4.4.3

    ## Warning: package 'parsnip' was built under R version 4.4.3

    ## Warning: package 'purrr' was built under R version 4.4.3

    ## Warning: package 'rsample' was built under R version 4.4.3

    ## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
    ## ✖ purrr::discard() masks scales::discard()
    ## ✖ dplyr::filter()  masks stats::filter()
    ## ✖ dplyr::lag()     masks stats::lag()
    ## ✖ recipes::step()  masks stats::step()

``` r
library(tm)
```

    ## Loading required package: NLP

    ## 
    ## Attaching package: 'NLP'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     annotate

``` r
library(ggplot2)
library(tidyverse)
```

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ## ✔ lubridate 1.9.3     ✔ tibble    3.2.1
    ## ✔ readr     2.1.5

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ NLP::annotate()     masks ggplot2::annotate()
    ## ✖ readr::col_factor() masks scales::col_factor()
    ## ✖ purrr::discard()    masks scales::discard()
    ## ✖ dplyr::filter()     masks stats::filter()
    ## ✖ stringr::fixed()    masks recipes::fixed()
    ## ✖ dplyr::lag()        masks stats::lag()
    ## ✖ readr::spec()       masks yardstick::spec()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(modeldata)
library(rpart.plot)
```

    ## Warning: package 'rpart.plot' was built under R version 4.4.3

    ## Loading required package: rpart
    ## 
    ## Attaching package: 'rpart'
    ## 
    ## The following object is masked from 'package:dials':
    ## 
    ##     prune

``` r
library(tidytext)
library(stringr)
library(tidyr)
library(caret)
```

    ## Loading required package: lattice
    ## 
    ## Attaching package: 'caret'
    ## 
    ## The following objects are masked from 'package:yardstick':
    ## 
    ##     precision, recall, sensitivity, specificity
    ## 
    ## The following object is masked from 'package:rsample':
    ## 
    ##     calibration
    ## 
    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
library(tidytext)
library(lubridate)
library(tibble)
```

## Data Frames

Now, we will load the csvs.
![](final_files/figure-gfm/pressure-1.png)<!-- -->

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
