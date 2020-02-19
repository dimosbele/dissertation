"""
This script is used to create the TF-IDF matrix and assign feature weights
"""

# import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# read the dataset with the preprocessed product details of bestprice.gr
df = pd.read_pickle('data/product_details/df_preproc.pkl')

# create a list (corpus) of all preprocessed product titles (product profiles)
corpus = df.Title_lc2.values.tolist()

# convert the above corpus to a matrix of TF-IDF features
tf = TfidfVectorizer() 
tfidf_matrix = tf.fit_transform(corpus)


# get a list of the unique terms in the corpus
feature_names = tf.get_feature_names()

# transform the sparce matrix to a list of dicts
# each dict corresponds to each term of a product title
tfidf_list = []
for doc in range(0,len(corpus)):
    feature_index = tfidf_matrix[doc,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])

    for i, w, s in [(i, feature_names[i], s) for (i, s) in tfidf_scores]:
        doc_dict = {'doc_id':doc, 'term_id':i, 'term':w, 'tfidf':s}
        tfidf_list.append(doc_dict)

# transform the list of dicts to a pandas dataframe
df_tfidf = pd.DataFrame(tfidf_list)

# set the weights for the 3 features (brand, category, subcategory)
# here we increase the significance of only the brand feature 
brand_weight = 1.5
category_weight = 1.0
subcategory_weight = 1.0

# create a dataframe with the terms of each feature (brand, category, subcategory) along with the weights
df_brand_weights = pd.DataFrame({'term':list(df.brand_name2.unique()), 
                                 'weight':brand_weight})

df_category_weights = pd.DataFrame({'term':list(df.Category2.unique()), 
                                 'weight':category_weight})

df_subcategory_weights = pd.DataFrame({'term':list(df.SubCategory2.unique()), 
                                 'weight':subcategory_weight})

# concatenate the above 3 dataframes
df_weights = pd.concat([df_brand_weights, df_category_weights, df_subcategory_weights], axis=0)

# merge the main 'df_tfidf' with the 'df_weights' dataframe
df_tfidf = df_tfidf.merge(df_weights, left_on='term', right_on='term', how='left')
# set the weight of all the other terms to 1
df_tfidf = df_tfidf.fillna(1)

# create a new column with the final weight of each term
df_tfidf['tfidf'] = df_tfidf['tfidf'] * df_tfidf['weight']

# groupby each product(doc_id) to a row and convert the rest of the columns to lists
df_tfidf2 = df_tfidf.groupby(['doc_id'], as_index=False)['term','term_id','tfidf'].agg(lambda x: list(x))

# add a column with the product id
df_tfidf2['product_id'] = df['product_id'].values.tolist()
# add a column with the product brand
df_tfidf2['brand_name2'] = df['brand_name2'].values.tolist()
# add a column with the product category
df_tfidf2['Category2'] = df['Category2'].values.tolist()
# add a column with the product subcategory
df_tfidf2['SubCategory2'] = df['SubCategory2'].values.tolist()

# save the dataframe as pickle file
# this file will be used in order to calculate the Minhash signature of each product
df_tfidf2.to_pickle('data/tfidfs/df_tfidf_brand_1-5.pkl')

