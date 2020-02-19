"""
This script is used to create the weighted Minhash signatures and the recommendations for each product
"""

# import libraries
import pandas as pd
from datasketch import WeightedMinHashGenerator
from datasketch import MinHashLSHForest

# read the pickle file that was created by 'tfidf.py' script
# this file contains the weighted representation of all products in 'Bestprice' dataset
df_tfidf = pd.read_pickle('data/tfidfs/df_tfidf_brand_1-0.pkl')

# create an extra column with the minhash id (m1, m2 etc)
df_tfidf['Minhash_id'] = df_tfidf['doc_id'].apply(lambda x: 'm'+str(x))

# create a WeightedMinHashGenerator object with the appropriate arguments
# dim: dimension - the number of unique terms
# sample_size: number of samples (similar to number of permutation functions in MinHash)
mg = WeightedMinHashGenerator(dim=35405, sample_size=128)

def create_minhash(doc):
	"""
	This function takes the weighted representation of a product and returns its Minhash signature.
    :param doc: The weighted representation of the product
    :return: The Minhash signature of the product as a Minhash object
	"""
    term_ids = doc['term_id']
    tfidfs = doc['tfidf']
    tfidf_list = [0]*35405
    
    i = 0
    for term_id in term_ids:
        tfidf_list[term_id] = tfidfs[i]
        i += 1
		
    m1 = mg.minhash(tfidf_list)
	
    return m1

# create a minhash for each row(product) by calling the 'create_minhash' function
df_tfidf['Minhash'] = df_tfidf[0:].apply(lambda x: create_minhash(x), axis=1)

# create a list with all the Minhash signatures
minhash_list = df_tfidf['Minhash']

# create a MinHashLSHForest object with num_perm parameter equal to sample_size(=128)
# num_perm: the number of permutation functions
forest = MinHashLSHForest(num_perm=128)

# add each Minhash signature into the index
i = 0
for minhash in minhash_list:
    # Add minhash into the index
    forest.add("m"+str(i), minhash)
    i += 1

# call index() in order to make the keys searchable
forest.index()

# create the recommendations by retrieving top 10 keys that have the higest Jaccard for each product

def make_recs(doc_id, n_recs):
    """
    This function takes the id of the target product and returns the top n_recs(=10) keys that have the higest Jaccard
    :param doc_id: the id of the target product
    :param n_recs: the number of similar products to be returned
    :return: top n_recs keys that have the higest Jaccard for each product
    """
    query = minhash_list[doc_id]
    
    # Using m1 as the query, retrieve top 10 keys that have the higest Jaccard
    results = forest.query(query, n_recs)
    
    return results

# for each product find the top 10 most similar products by calling the 'make_recs' function
df_tfidf['recs'] = df_tfidf['doc_id'].apply(lambda x: make_recs(x, 10))

# finalize the dataset

# create a df with only the recs of each product
df_recs = df_tfidf[['product_id', 'recs']]
# expand each row to as many rows as the length of the recs list
df_recs = df_recs.set_index('product_id').recs.apply(pd.Series).stack().reset_index(level=-1, drop=True).astype(str).reset_index()
# rename the columns
df_recs.columns = ['product_id', 'rec_m_id']

# add the brand, category, subcategory of each recommended product
df_recs = df_recs.merge(df_tfidf[['Minhash_id', 'brand_name2', 'Category2', 'SubCategory2']], left_on='rec_m_id', right_on='Minhash_id', how='left')

# groupby each product and convert to lists
df_recs = df_recs.groupby(['product_id'], as_index=False)['brand_name2', 'Category2', 'SubCategory2'].agg(lambda x: list(x))
# rename columns
df_recs.columns = ['product_id', 'Brands', 'Categories', 'Subcategories']

# add the above information to the main dataset
df_recs2 = df_tfidf.merge(df_recs, left_on='product_id', right_on='product_id', how='left')

# create 3 columns with the number of uniique brands, categories, subcategories for the evaluation process
df_recs2['N_Brands'] = df_recs2['Brands'].apply(lambda x: len(set(x)))
df_recs2['N_Categories'] = df_recs2['Categories'].apply(lambda x: len(set(x)))
df_recs2['N_Subcategories'] = df_recs2['Subcategories'].apply(lambda x: len(set(x)))

# save the dataframe as pickle file
df_recs2.to_pickle('data/recommendations/df_recos_brand_1-5.pkl')

