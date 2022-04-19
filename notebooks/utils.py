from collections import defaultdict
from lenskit import batch
import pandas as pd
import numpy as np
from tqdm import tqdm
def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
    
    
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def pk(k, rec_df, truth_df):
    '''
    compute precision at k
    '''
    ct_usr = 0
    for usr in tqdm(truth_df.user.values):
        ct_rec = 0
        rec_all = rec_df[rec_df['user'] == usr].item.values[0][:k]
        val_all = truth_df[truth_df['user'] == usr].item.values[0]
        ttl = [rec_item in val_all for rec_item in rec_all]
        ct_rec = sum(ttl)
        ct_usr += ct_rec / k
    return ct_usr / len(truth_df.user.values)



def meanAP(rec_df, truth_df):
    ct_usr = 0
    for usr in tqdm(truth_df.user.values):
        ct_rec = 0
        rec_all = rec_df[rec_df['user'] == usr].item.values[0]
        val_all = truth_df[truth_df['user'] == usr].item.values[0]
        ttl = [rec_item in val_all for rec_item in rec_all]
        ttl = [v/(j+1) for j,v in enumerate(ttl)]
        ct_rec += sum(ttl)
        ct_usr += ct_rec / len(val_all)
        
    return ct_usr / len(truth_df.user.values)



def ndcg(k, rec_df, truth_df):
    ct_usr = 0
    for usr in tqdm(truth_df.user.values):
        rec_all = rec_df[rec_df['user'] == usr].item.values[0]
        val_all = truth_df[truth_df['user'] == usr].item.values[0]
        n = min(max(len(rec_all), len(val_all)), k)
        idcg_n = min(len(val_all), k)
        idcg = sum([1/(np.log(j+2)) for j in range(idcg_n)])
        ttl = [rec_item in val_all for rec_item in rec_all[:n]]
        ttl = sum([v/np.log(j+2) for j, v in enumerate(ttl)])
        ttl *= 1/idcg
        ct_usr += ttl
    return ct_usr / len(truth_df.user.values)

def get_recs(fittable, users, k=10):
    
    recs = batch.recommend(fittable, users, 10, n_jobs = 10)
    recs['user'] = recs['user'].map(int)
    return recs
    

def get_metrics(k, rec_df, truth_df):
    
    rec_df = pd.DataFrame({'item':rec_df.groupby('user').item.apply(list)}).reset_index()
    truth_df = pd.DataFrame({'item':truth_df.groupby('user').item.apply(list)}).reset_index()
    
    return {'pk':pk(k, rec_df, truth_df), 'meanAP': meanAP(rec_df, truth_df), 'NDCG':ndcg(k, rec_df, truth_df)}
   