import sys
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id', how = 'outer')
    return df


def clean_data(df):
    #split category column into 36 columns split by ';'
    categories = df.categories.str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories[:1]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.str[0:-2]).values[0]
    
    #replace column names with cleaned category labels
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    #some '2's snuck in to the related column - convert those into 1's    
    categories['related'] = (categories.related>=1).astype('int')
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    #drop duplicates
    df.drop_duplicates(subset = ['id', 'message'], inplace = True)
    
    print(df.head())
    return df


def save_data(df, database_filename):
    import sqlalchemy
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messagesCategorizedCommandLine', engine, if_exists='replace', index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()