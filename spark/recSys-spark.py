#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getpass
import sys
import numpy as np
import pyspark.sql.functions as F
from random import sample
from pyspark import SparkConf
from pyspark.sql import SparkSession, functions
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS,ALSModel
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import subprocess

# source shell_setup.sh first
# hfs -put
# spark-submit --driver-memory=6g --executor-memory=20g recSys-spark.py {truth, bert}
# yarn logs -applicationId {app_id} -log_files {stdout, stderr}
# yarn application -kill {app_id}

def main(spark, netID, infile):
    if infile == 'sparse':
        cf = spark.read.parquet(f'/user/{netID}/train_sparse.pq')
    else:
        
        cf = spark.read.parquet(f'/user/{netID}/train_sent.pq')

    cf.createOrReplaceTempView('cf')
    cf_val = spark.read.parquet(f'/user/{netID}/val_data.pq')
    cf_val.createOrReplaceTempView('cf_val') 
    ### Select users
    train_users = set(row['user'] for row in cf.select('user').distinct().collect()) 
    indexer_user = StringIndexer(inputCol="user", outputCol="user_Index")
    indexed_user = indexer_user.setHandleInvalid("skip").fit(cf)

    cf = indexed_user.transform(cf)
    cf_val = indexed_user.transform(cf_val) 
    
    indexer_track = StringIndexer(inputCol="item", outputCol="item_Index")
    indexed_track = indexer_track.setHandleInvalid("skip").fit(cf)
    cf = indexed_track.transform(cf)
    cf_val = indexed_track.transform(cf_val)

    print(f'Number of users in training: {len(train_users)}')


    ### See if there exists a trained model 
    def run_cmd(args_list):
        proc = subprocess.Popen(args_list, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
        proc.communicate()
        return proc.returncode
    

    ###Create model
    ranks = [5]#,10,15,20]
    regParams = [0.01]#[1.#,0.1,0.01]
    for rank in ranks:
        for regParam in regParams:

            cmd = ['hdfs', 'dfs', '-test', '-d', f"model/model_{rank}_{regParam}_{infile}"]
            code = run_cmd(cmd)
            if code == 0:
                print(f"/model/model_{rank}_{regParam}_{infile} exist \n Loading model...")
                print(f'code:{code}') 
                als = ALS.load('model/als_cf_train')
                model = ALSModel.load('model/model_cf_train')
            else:  
                print(f'code:{code}') 
                print('No trained model found, start training...')
                als = ALS(rank=rank, maxIter=5, seed=40, regParam=regParam, userCol="user_Index", itemCol="item_Index", ratingCol="rating", implicitPrefs=False,nonnegative=True,coldStartStrategy="drop")
                model = als.fit(cf)
                ### Save model for testing set
                als.write().overwrite().save(f'model/als_{rank}_{regParam}_{infile}')
                model.write().overwrite().save(f'model/model_{rank}_{regParam}_{infile}')
                print('model saved')
           
            user_ids = cf_val.select(als.getUserCol()).distinct()
            K = 50
            recoms = model.recommendForUserSubset(user_ids, K)

            #recommendForAllItems

            predictions = recoms.select('user_Index','recommendations.item_Index')
            print('predictions done')

            ### Group by index and aggregate
            truth = cf.select('user_Index', 'item_Index').groupBy('user_Index').agg(F.expr('collect_list(item) as truth'))
            print('truth done')


            combined = predictions.join(functions.broadcast(truth), 'user_Index', 'inner').rdd
            combined_mapped = combined.map(lambda x: (x[1], x[2]))
            print('rdd created')
            ### Metrics ref:https://spark.apache.org/docs/2.3.0/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics
            metrics = RankingMetrics(combined_mapped)
            print('metrics done')
            ### Mean Average Precision
            MAP = metrics.meanAveragePrecision
            ### Normalized Discounted Cumulative Gain
            ndcg = metrics.ndcgAt(K)
            ### Precision at k
            pk = metrics.precisionAt(K)
            print(f'Rank:{rank}, regParam: {regParam}, map score:  {MAP}, ndcg score: {ndcg}, pk score: {pk}')

    
if __name__ == "__main__":

    spark = SparkSession.builder.appName('tuning').config('spark.blacklist.enabled', False).getOrCreate()
        
    netID = getpass.getuser()
    
    infile = sys.argv[-1]

    main(spark, netID, infile)