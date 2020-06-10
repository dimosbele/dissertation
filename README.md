# recommendation_system_CB
Promoting Diversity in Content Based Recommendation using Feature Weighting and LSH

This project focuses on an efficient Content-Based (CB) product recommendation methodology that promotes diversity. A heuristic CB approach incorporating feature weighting and Locality-Sensitive Hashing (LSH) is used, along with the TF-IDF method and functionality of tuning the importance of product features to adjust its logic to the needs of various e-commerce sites. The problem of efficiently producing recommendations, without compromising similarity, is addressed by approximating product similarities via the LSH technique. The methodology is evaluated on two sets with real e-commerce data. The evaluation of the proposed methodology shows that the produced recommendations can help customers to continue browsing a site by providing them with the necessary “next step”. Finally, it is demonstrated that the methodology incorporates recommendation diversity which can be adjusted by tuning the appropriate feature weights. The corresponding paper will be presented at the at the 16th International Conference on Artificial Intelligence Applications and Innovations (AIAI 2020). Check the correpsonding paper here https://www.researchgate.net/publication/341728370_Promoting_Diversity_in_Content_Based_Recommendation_Using_Feature_Weighting_and_LSH

## - Folders description

1) bestprice - The folder contains the notebooks and data regarding the 'bestprice' dataset. <br>
1.1. EDA_preprocess.ipynb - Analysing and preprocessing the initial 'bestprice' dataset <br>
1.2. tfidf.ipynb - Converting the dataset to TF-IDF matrix <br>
1.3. minhash_lsh.ipynb - Creating the Minhash signature of each product and producing recommendations based on LSH Forest <br>
1.4. results.ipynb - Evaluating the produced recommendations <br>
1.5. future_work.ipynb - Checking future plans in order to make the RS more personalized <br>
1.6. /data/product_details - The initial 'bestprice' dataset (df_init.pkl) 

1) retailrocket - The folder contains the notebooks and data regarding the 'retailrocket' dataset*. <br>
1.1. EDA_events.ipynb - Analysing and preprocessing the 'events.csv' dataset* <br>
1.2. sessions.ipynb - Analysing and preprocessing the datasets* <br>
1.3. tfidf_retailrocket.ipynb - Converting the dataset to TF-IDF matrix <br>
1.4. minhash_lsh_retailrocket.ipynb - Creating the Minhash signature of each product and producing recommendations based on LSH Forest <br>
1.5. recs_evaluation.ipynb - Evaluating the produced recommendations <br>
1.6. data/final/product_details - The retailrocket products details dataset after preprocessing (df_product_details_retailrocket.pkl)
* The initial datasets are here https://www.kaggle.com/retailrocket/ecommerce-dataset

3) submitted_codes - The final .py files with the codes for the 'bestprice' dataset. <br>
3.1. preprocess_bestprice.py - preprocessing the initial 'bestprice' dataset<br>
3.2. tfidf.py - Converting the dataset to TF-IDF matrix <br>
3.3. minhash_lsh.py - Creating the Minhash signature of each product and producing recommendations based on LSH Forest <br>


Dimosthenis Beleveslis <br>
dimbele4@gmail.com
