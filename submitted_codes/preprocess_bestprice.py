"""
This script is used to preprocess the initial 'Bestprice' dataset
"""

# import libraries
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer

# read the initial 'bestprice' dataset
df = pd.read_pickle('data/df_init.pkl')

# extract the id number from each product url
df['product_id'] = df['url'].apply(lambda x: re.search(r"item\/([^']*)\/", x).group(1))

# convert category to lowercase
df['Category_lc'] = df['Category'].apply(lambda text: text.lower())
# convert SubCategory to lowercase
df['SubCategory_lc'] = df['SubCategory'].apply(lambda text: text.lower())
# convert brand_name to lowercase
df['brand_name_lc'] = df['brand_name'].apply(lambda text: text.lower())
# convert title to lowercase
df['Title_lc'] = df['Title'].apply(lambda text: text.lower())

# create a new column for 'Category' and add the string '_cat' after the category
df['Category2'] = df['Category_lc'].apply(lambda x: x+'_cat')
# create a new column for 'SubCategory' and add the string '_subcat' after the SubCategory
df['SubCategory2'] = df['SubCategory_lc'].apply(lambda x: x+'_subcat')
# create a new column for 'brand' and add the string '_brand' after the brand_name
df['brand_name2'] = df['brand_name_lc'].apply(lambda x: x+'_brand')

def replace_brand(row):
	"""
	This function adds (or replaces) the brand_name2 to the product title
    :row: the available information of the product
    :return: the product title with the brand_name2
	"""
    title = row['Title_lc']
    brand = row['brand_name_lc']
    brand2 = row['brand_name2']
    
    if brand in title:
        title2 = re.sub(brand, brand2, title)
    else:
        title2 = title +' '+ brand2
        
    return title2

# add brand in the product title by calling 'replace_brand' function
df['Title_lc2'] = df[['Title_lc', 'brand_name_lc', 'brand_name2']].apply(lambda x: replace_brand(x), axis=1)

# add category in the end of the product title
df['Title_lc2'] = df[['Title_lc2', 'Category2']].apply(lambda x: x['Title_lc2']+' '+x['Category2'], axis=1)

# add sub-category in the end of the product title
df['Title_lc2'] = df[['Title_lc2', 'SubCategory2']].apply(lambda x: x['Title_lc2']+' '+x['SubCategory2'], axis=1)

# keep only useful columns
df = df[['product_id', 'url', 'Title', 'Category2', 'SubCategory2', 'brand_name2', 'Title_lc2']]

# keep each product only once by deleting duplicates
df = df.drop_duplicates(subset='product_id', keep='first')

# save the dataframe as pickle file
# this file will be used in order to create the TF-IDF matrix
df.to_pickle('data/product_details/df_preproc.pkl')