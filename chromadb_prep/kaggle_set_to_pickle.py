# this script downloads the NY Times comment dataset from Kaggle
# then only takes the data relevant to our project and converts it to a pickle file for faster/easier use
# the dataset contains millions of user comments on NYTimes articles in certain months in 2017 and 2018
# you will need to set up you personal Kaggle API token for this to work

import pandas as pd
import os
import glob
import opendatasets as od

# dataset URL
dataset = 'https://www.kaggle.com/datasets/aashita/nyt-comments/'

# Using opendatasets let's download the data sets (480 MB)
od.download(dataset)

# downloaded folder contains many article csv files - we are not interested in them
# remove article csvs to leave just comments csvs
for f in glob.glob("nyt-comments/Article*"):
    os.remove(f)

# load all 2017 comment csv files into one single dataframe
# Get a list of all CSV files in a directory
csv_files_2017 = glob.glob('nyt-comments/*2017.csv')

# Create an empty dataframe to store the combined data
combined_df_2017 = pd.DataFrame()

# Loop through each CSV file and append its contents to the combined dataframe
for csv_file in csv_files_2017:
    df = pd.read_csv(csv_file)
    combined_df_2017 = pd.concat([combined_df_2017, df])

# add a column with year
combined_df_2017.loc[:, "year"] = 2017

# select only year and comment body
comments_2017 = combined_df_2017[["year", "commentBody"]]

# repeat for 2018 comments
csv_files_2018 = glob.glob('nyt-comments/*2018.csv')
combined_df_2018 = pd.DataFrame()
for csv_file in csv_files_2018:
    df = pd.read_csv(csv_file)
    combined_df_2018 = pd.concat([combined_df_2018, df])
combined_df_2018.loc[:, "year"] = 2018
comments_2018 = combined_df_2018[["year", "commentBody"]]

# combine into single df with year and comment
comments = pd.concat([comments_2017, comments_2018])

# write to pickle
comments.to_pickle("comments.pickle")