Final Project
================
miranda_moe
2026-05-09

- [Packages](#packages)
- [Data Frames](#data-frames)
- [Text cleaning](#text-cleaning)
- [Prepping Data Frames to build classification
  model](#prepping-data-frames-to-build-classification-model)
- [Building Classification Model](#building-classification-model)

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
library(textstem)
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
# In scifi.csv, the genres are listed as "Sci-Fi". In sports.csv, they are listed as "Sport". so we would need to map that. 
genre_map <- c(
  "scifi" = "sci-fi",
  "sports" = "sport"
)
movies_list <- lapply(file_list, function(f) {
  genre_name <- gsub(".csv", "", f)
  search_term <- ifelse(genre_name %in% names(genre_map), genre_map[genre_name], genre_name)
  df <- read_csv(f) |>
    filter(str_detect(str_to_lower(genre), search_term)) |>
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

## Text cleaning

When I look through the new dataframe “movies”, I noticed that some
movies are missing descriptions (which happens across genres.) Those
missing movie descriptions are indicated as “Add a Plot” rather than NA
or blank cells. Therefore, we will remove those movies so it wouldn’t
mess up the classification.

``` r
filtered_movies <- movies %>%
  filter(!grepl("Add a Plot", description))
```

Here, we will do some basic text analysis, we will do that in the
following order:

- tokenize
- remove numbers (mostly just years, such as 2018, 2019 etc)
- remove stopwords
- lemmatize (reduce words to stem. for example, ‘playing’ would become
  ‘play’)
- TF-IDF (high tf-idf means words that appear a lot in one movide but
  rarely across others)
- Document-Term Matrix (DTM) that will transform into a numeric matrix

Before starting any text operations, I have noticed that some
descriptions don’t have the full descriptions in the original data and
at the end, it would say ” See full summary »“. And, we also have
parantheses in descriptions that would include years (for example,
spinoff from a movie name (2018), or a character name with an action
name in parantheses). I assumed those weren’t super necessary for the
model to learn the features. And, there are periods between words such
as L.A. or C.I.A, and tokenization might split them, which might not
result in a meaningful result. There are also single and double
quotation marks around names, which we will also get rid of. Therefore,
in the following steps we will remove those by using stringr functions.

``` r
movies_cleaned_descriptions <- filtered_movies %>%
  mutate(description = str_remove(description, "See full summary.*$"),
         description = str_remove_all(description, "\\s*\\([^\\)]+\\)"), #remove paratheses
         description = str_remove_all(description, "(?<=[A-Za-z])\\.(?=[A-Za-z])"), #remove periods
         description = str_remove_all(description, "[\"']")) #remove the single and double quotations
```

I’ve noticed that some movies still appeared across genres, since a
movie that is labeled as ‘Action, Adventure.’ would appear in both csvs.
And, since I will be using single label classification, I decided to
only keep the first occurance of each movie.

``` r
movies_cleaned_descriptions <- movies_cleaned_descriptions %>%
  distinct(movie_id, .keep_all = TRUE)
```

Now that we’ve cleaned those, we will start text cleaning by using
tidytext.

First step is tokenizing, removing stopwords and numbers. We will also
convert the dataframe to tibbles before lemmatizing because tidyverse
handles tibbles better and it will give us one token per row.

``` r
text_movies <- movies_cleaned_descriptions %>%
  unnest_tokens(word, description) %>%
  filter(!grepl('[0-9]', word)) %>%
  anti_join(stop_words) 
  
lemmatized_movies <- tibble(text_movies) %>%
  mutate(word = lemmatize_words(word)) %>% #now we will lemmatize the words
  count(movie_id, movie_name, word, final_genre, sort = TRUE) %>%
  bind_tf_idf(movie_id, word, n) %>%
  arrange(desc(tf_idf))
lemmatized_movies
```

    ## # A tibble: 2,359,116 × 8
    ##    movie_id   movie_name          word      final_genre     n    tf   idf tf_idf
    ##    <chr>      <chr>               <chr>     <chr>       <int> <dbl> <dbl>  <dbl>
    ##  1 tt0357912  Mohar               kxdhjdix… family          1     1  11.5   11.5
    ##  2 tt11690880 Adventure Night     tbd       adventure       1     1  11.5   11.5
    ##  3 tt1207994  Tanikalang dugo     sssshhhh… horror          1     1  11.5   11.5
    ##  4 tt14246472 Sommerland          sommerla… horror          1     1  11.5   11.5
    ##  5 tt2445788  Arkadas             senderöe  thriller        1     1  11.5   11.5
    ##  6 tt3395582  Sun Choke           janies    horror          1     1  11.5   11.5
    ##  7 tt7472110  a new test          alksjdhf… war             1     1  11.5   11.5
    ##  8 tt0053941  The White Horse Inn rössl     romance         1     1  10.8   10.8
    ##  9 tt0085789  Kiez                kiez      crime           1     1  10.8   10.8
    ## 10 tt0270224  Bang-bang boomerang duckhunt… animation       1     1  10.8   10.8
    ## # ℹ 2,359,106 more rows

As a final step before building a model, we coverted the TF-IDF weighted
tokens into a Document-Term Matrix using cast_dm(). A DTM represents
each movie as a row and each unique word as a column, where values will
be TF-IDF we calculated above. As mentioned before, DMF will covert the
current one-token-per-row structure to numerical matrix. TF-IDF
weighting was used instead of raw word counts so that common words
appearing across many movies are already weighted, allowing the model to
focus on words that are more distinctive and informative for predicting
genre.

``` r
document_term_matrix_movies <- lemmatized_movies %>%
  cast_dtm(movie_id, word, tf_idf)
document_term_matrix_movies
```

    ## <<DocumentTermMatrix (documents: 188093, terms: 96045)>>
    ## Non-/sparse entries: 2359116/18063033069
    ## Sparsity           : 100%
    ## Maximal term length: 44
    ## Weighting          : term frequency (tf)

We have currently 96045 unique words as columns, which will cause a
problem when we convert to dataframe to train data. Therefore, we will
remove the words that appear in very few movies.

``` r
dtm_reduced <- removeSparseTerms(document_term_matrix_movies, 0.99)
dtm_reduced
```

    ## <<DocumentTermMatrix (documents: 188093, terms: 171)>>
    ## Non-/sparse entries: 701341/31462562
    ## Sparsity           : 98%
    ## Maximal term length: 12
    ## Weighting          : term frequency (tf)

We removed terms that are missing from more than 99% of documents. As a
result, our novel terms came down to 171.

Now that we have a summary of DocumentTerm Matrix, we would need the
categorical y (the genres) that we want to predict by using the DTM
values. Therefore, we will create the outcome labels.

``` r
genre_labels <- lemmatized_movies %>%
  distinct(movie_id, final_genre)
head(genre_labels)
```

    ## # A tibble: 6 × 2
    ##   movie_id   final_genre
    ##   <chr>      <chr>      
    ## 1 tt0357912  family     
    ## 2 tt11690880 adventure  
    ## 3 tt1207994  horror     
    ## 4 tt14246472 horror     
    ## 5 tt2445788  thriller   
    ## 6 tt3395582  horror

## Prepping Data Frames to build classification model

We would first need the dtm_reduce to be in a dataframe where we can
join the outcome categories (labels).

``` r
dtm_df <- as.data.frame(as.matrix(dtm_reduced))
head(dtm_df, 2)
```

    ##            game play island horror black hope bad land soldier hes cop accident
    ## tt0357912     0    0      0      0     0    0   0    0       0   0   0        0
    ## tt11690880    0    0      0      0     0    0   0    0       0   0   0        0
    ##            deal middle spend serial attack real hunt comedy boy hire street die
    ## tt0357912     0      0     0      0      0    0    0      0   0    0      0   0
    ## tt11690880    0      0     0      0      0    0    0      0   0    0      0   0
    ##            win trip marriage village army car power drug rich tale mission
    ## tt0357912    0    0        0       0    0   0     0    0    0    0       0
    ## tt11690880   0    0        0       0    0   0     0    0    0    0       0
    ##            lover catch earth call prison future head miss age dead parent
    ## tt0357912      0     0     0    0      0      0    0    0   0    0      0
    ## tt11690880     0     0     0    0      0      0    0    0   0    0      0
    ##            century dream hide action haunt change country money detective
    ## tt0357912        0     0    0      0     0      0       0     0         0
    ## tt11690880       0     0    0      0     0      0       0     0         0
    ##            dangerous officer college york human deadly encounter kidnap stop
    ## tt0357912          0       0       0    0     0      0         0      0    0
    ## tt11690880         0       0       0    0     0      0         0      0    0
    ##            battle visit child drama job sister adventure gang series seek
    ## tt0357912       0     0     0     0   0      0         0    0      0    0
    ## tt11690880      0     0     0     0   0      0         0    0      0    0
    ##            female husband team star arrive name thriller revenge agent move
    ## tt0357912       0       0    0    0      0    0        0       0     0    0
    ## tt11690880      0       0    0    0      0    0        0       0     0    0
    ##            trouble movie attempt learn couple criminal student wealthy tell
    ## tt0357912        0     0       0     0      0        0       0       0    0
    ## tt11690880       0     0       0     0      0        0       0       0    0
    ##            evil beautiful break girl night relationship crime journey join
    ## tt0357912     0         0     0    0     0            0     0       0    0
    ## tt11690880    0         0     0    0     0            0     0       0    0
    ##            local american survive event girlfriend brother house involve find
    ## tt0357912      0        0       0     0          0       0     0       0    0
    ## tt11690880     0        0       0     0          0       0     0       0    0
    ##            plan investigate police school lead run steal true strange mother
    ## tt0357912     0           0      0      0    0   0     0    0       0      0
    ## tt11690880    0           0      0      0    0   0     0    0       0      0
    ##            day dark past fight people son secret start killer lose kill save
    ## tt0357912    0    0    0     0      0   0      0     0      0    0    0    0
    ## tt11690880   0    0    0     0      0   0      0     0      0    0    0    0
    ##            daughter travel bring marry return city search film town plot war
    ## tt0357912         0      0     0     0      0    0      0    0    0    0   0
    ## tt11690880        0      0     0     0      0    0      0    0    0    0   0
    ##            meet escape struggle leave home time base live wife death take set
    ## tt0357912     0      0        0     0    0    0    0    0    0     0    0   0
    ## tt11690880    0      0        0     0    0    0    0    0    0     0    0   0
    ##            decide discover begin force story woman mysterious fall world friend
    ## tt0357912       0        0     0     0     0     0          0    0     0      0
    ## tt11690880      0        0     0     0     0     0          0    0     0      0
    ##            father family murder love life
    ## tt0357912       0      0      0    0    0
    ## tt11690880      0      0      0    0    0

As we can see above, movie ids are not a column and they appear as row
names. We will first convert them to column called movie_id before we
join the labels.

``` r
dtm_df$movie_id <- rownames(dtm_df)
head(dtm_df$movie_id)
```

    ## [1] "tt0357912"  "tt11690880" "tt1207994"  "tt14246472" "tt2445788" 
    ## [6] "tt3395582"

``` r
dtm_df <- dtm_df %>%
  left_join(genre_labels, by = "movie_id")
head(dtm_df, 2)
```

    ##   game play island horror black hope bad land soldier hes cop accident deal
    ## 1    0    0      0      0     0    0   0    0       0   0   0        0    0
    ## 2    0    0      0      0     0    0   0    0       0   0   0        0    0
    ##   middle spend serial attack real hunt comedy boy hire street die win trip
    ## 1      0     0      0      0    0    0      0   0    0      0   0   0    0
    ## 2      0     0      0      0    0    0      0   0    0      0   0   0    0
    ##   marriage village army car power drug rich tale mission lover catch earth call
    ## 1        0       0    0   0     0    0    0    0       0     0     0     0    0
    ## 2        0       0    0   0     0    0    0    0       0     0     0     0    0
    ##   prison future head miss age dead parent century dream hide action haunt
    ## 1      0      0    0    0   0    0      0       0     0    0      0     0
    ## 2      0      0    0    0   0    0      0       0     0    0      0     0
    ##   change country money detective dangerous officer college york human deadly
    ## 1      0       0     0         0         0       0       0    0     0      0
    ## 2      0       0     0         0         0       0       0    0     0      0
    ##   encounter kidnap stop battle visit child drama job sister adventure gang
    ## 1         0      0    0      0     0     0     0   0      0         0    0
    ## 2         0      0    0      0     0     0     0   0      0         0    0
    ##   series seek female husband team star arrive name thriller revenge agent move
    ## 1      0    0      0       0    0    0      0    0        0       0     0    0
    ## 2      0    0      0       0    0    0      0    0        0       0     0    0
    ##   trouble movie attempt learn couple criminal student wealthy tell evil
    ## 1       0     0       0     0      0        0       0       0    0    0
    ## 2       0     0       0     0      0        0       0       0    0    0
    ##   beautiful break girl night relationship crime journey join local american
    ## 1         0     0    0     0            0     0       0    0     0        0
    ## 2         0     0    0     0            0     0       0    0     0        0
    ##   survive event girlfriend brother house involve find plan investigate police
    ## 1       0     0          0       0     0       0    0    0           0      0
    ## 2       0     0          0       0     0       0    0    0           0      0
    ##   school lead run steal true strange mother day dark past fight people son
    ## 1      0    0   0     0    0       0      0   0    0    0     0      0   0
    ## 2      0    0   0     0    0       0      0   0    0    0     0      0   0
    ##   secret start killer lose kill save daughter travel bring marry return city
    ## 1      0     0      0    0    0    0        0      0     0     0      0    0
    ## 2      0     0      0    0    0    0        0      0     0     0      0    0
    ##   search film town plot war meet escape struggle leave home time base live wife
    ## 1      0    0    0    0   0    0      0        0     0    0    0    0    0    0
    ## 2      0    0    0    0   0    0      0        0     0    0    0    0    0    0
    ##   death take set decide discover begin force story woman mysterious fall world
    ## 1     0    0   0      0        0     0     0     0     0          0    0     0
    ## 2     0    0   0      0        0     0     0     0     0          0    0     0
    ##   friend father family murder love life   movie_id final_genre
    ## 1      0      0      0      0    0    0  tt0357912      family
    ## 2      0      0      0      0    0    0 tt11690880   adventure

Now that we’ve prepared the dataframe, we will proceed with building the
model.

## Building Classification Model

First, we will change the outcome variable into a factor

``` r
str(dtm_df$final_genre)
```

    ##  chr [1:188093] "family" "adventure" "horror" "horror" "thriller" "horror" ...

``` r
dtm_df$final_genre <- factor(dtm_df$final_genre)
levels(dtm_df$final_genre)
```

    ##  [1] "action"    "adventure" "animation" "biography" "crime"     "family"   
    ##  [7] "fantasy"   "film-noir" "history"   "horror"    "mystery"   "romance"  
    ## [13] "scifi"     "sports"    "thriller"  "war"
