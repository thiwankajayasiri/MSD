# imports
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, QuantileDiscretizer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import json

# Q2 -----------------------------------------------------------------------
# set root directory path
root = 'hdfs:///data/msd/'

# read mismatches
mismatches = (
    spark.read.text(root + 'tasteprofile/mismatches/sid_mismatches.txt')
    .select(
        F.trim(F.col('value').substr(9, 18)).alias('song_id').cast(StringType()),
        F.trim(F.col('value').substr(28, 18)).alias('track_id').cast(StringType())
    )
)

# read triplets
triplets = (
    spark.read.format('com.databricks.spark.csv')
    .option('delimiter', '\t')
    .option('header', 'false')
    .option('inferSchema', 'true')
    .schema(
        StructType([
            StructField('user_id', StringType()),
            StructField('song_id', StringType()),
            StructField('playcount', IntegerType())
        ])
    )
    .load(root + 'tasteprofile/triplets.tsv/*')
)

# remove mismatches song from triplets
triplets = (
    triplets
    .join(
        mismatches
        .select('song_id'),
        on='song_id',
        how='left_anti'
    )
)

# count unique user_id
triplets.select('user_id').dropDuplicates().count()
  # 1019318

# count unique song_id
triplets.select('song_id').dropDuplicates().count()
  # 378309



# most active users
active = (
    triplets
    .groupBy('user_id')
    .agg(
        F.sum('playcount').alias('playcount'),
        F.collect_list('song_id').alias('songs')
    )
    .orderBy('playcount', ascending=False)
    .limit(1)
    .rdd.take(1)[0]
)
active.__getattr__('user_id'), len(active.__getattr__('songs'))    #https://blog.rmotr.com/python-magic-methods-and-getattr-75cf896b3f88
  # ('093cb74eb3c517c5179ae24caf0ebec51b24d2a2', 195)

# song popularity
song_popularity = (
    triplets
    .groupBy('song_id')
    .agg(
        F.sum('playcount').alias('playcount')
    )
)

n = song_popularity.approxQuantile('playcount', [0.0, 0.25, 0.5, 0.75, 1.0], 0.0)
  # [1.0, 8.0, 31.0, 130.0, 726885.0]

# save output as gzip csv
song_popularity.coalesce(1).write.format('com.databricks.spark.csv').option('compression', 'gzip').save('hdfs:///user/gtj13/outputs/msd/song_1',mode = 'overwrite', header = True)

#hdfs dfs -copyToLocal hdfs:///user/gtj13/outputs/msd/song_1/ /users/home/gtj13

  
# user popularity
user_activity = (
    triplets
    .groupBy('user_id')
    .agg(
        F.count('song_id').alias('songcount')
    )
)


# topusers = user_activity.orderBy('songcount',ascending=False)
# topusers.head(n=10)



m = user_activity.approxQuantile('songcount', [0.0, 0.25, 0.5, 0.75, 1.0], 0.0)
  # [3.0, 15.0, 26.0, 53.0, 4316.0]

# save output as parquet file
user_activity.coalesce(1).write.format('com.databricks.spark.csv').option('compression', 'gzip').save('hdfs:///user/gtj13/outputs/msd/user_1',mode = 'overwrite', header = True)

#hdfs dfs -copyToLocal hdfs:///user/gtj13/outputs/msd/user_1/ /users/home/gtj13





# inactive songs
inactive_song = (
    song_popularity
    .filter(song_popularity.playcount < n[1])
    .select('song_id')
)

# inactive users
inactive_user = (
    user_activity
    .filter(user_activity.songcount < m[1])
    .select('user_id')
)

# remove inactives from triplets
triplets = (
    triplets
    .join(inactive_song, on='song_id', how='left_anti')
    .join(inactive_user, on='user_id', how='left_anti')
)




triplets.coalesce(1).write.format('com.databricks.spark.csv').option('compression', 'gzip').save('hdfs:///user/gtj13/outputs/msd/triplet_1',mode = 'overwrite', header = True)

triplets.coalesce(1).write.format('com.databricks.spark.csv').option(delimiter = '\t').save('hdfs:///user/gtj13/outputs/msd/triplet_1',mode = 'overwrite', header = True)


## refer to the plotting file line number 






#hdfs dfs -copyToLocal hdfs:///user/gtj13/outputs/msd/triplet_1/ /users/home/gtj13

# count
triplets.count()
  # 42818160

triplets.show(3,False)
# +----------------------------------------+------------------+---------+
# |user_id                                 |song_id           |playcount|
# +----------------------------------------+------------------+---------+
# |00001cf0dce3fb22b0df0f3a1d9cd21e38385372|SODRFRJ12A8C144167|2        |
# |00001cf0dce3fb22b0df0f3a1d9cd21e38385372|SOUPXLX12A67ADE83A|1        |
# |00001cf0dce3fb22b0df0f3a1d9cd21e38385372|SOLVTNS12AB01809E2|2        |
# +----------------------------------------+------------------+---------+
# only showing top 3 rows


sort_by_playcount = triplets.orderBy('playcount',ascending=False)
sort_by_playcount.head(n=10)

#Top #10 Mostly Played Songs by the User ID

# [Row(user_id='093cb74eb3c517c5179ae24caf0ebec51b24d2a2', song_id='SOAOSDF12A58A779F1', playcount=9667),
#  Row(user_id='c11dea7d1f4d227b98c5f2a79561bf76884fcf10', song_id='SOZTEZR12A8C14204B', playcount=3534),
#  Row(user_id='d8e6fa08d73821f305b9a3af1cf1e0a704473d82', song_id='SOBONKR12A58A7A7E0', playcount=3532),
#  Row(user_id='1854daf178674bbac9a8ed3d481f95b76676b414', song_id='SOVLAWN12A81C234AB', playcount=2948),
#  Row(user_id='69807196f964e5b063af898fd1cb098fab2e6a1f', song_id='SOVQQJO12AB0182328', playcount=2381),
#  Row(user_id='a263000355e6a46de29ec637820771ac7620369f', song_id='SONSTND12AB018516E', playcount=2368),
#  Row(user_id='6b36f65d2eb5579a8b9ed5b4731a7e13b8760722', song_id='SOGDNAW12A6D4F6804', playcount=2165),
#  Row(user_id='0d0f80a34807aab31a3521424d456d30bf2c93d9', song_id='SOFRRFT12A8C140C5C', playcount=1890),
#  Row(user_id='944cdf52364f45b0edd1c972b5a73d3a86b09c6a', song_id='SOKGULH12A6D4F70BB', playcount=1862),
#  Row(user_id='b881152f1394e3c2bbdb9cc853016432ba184c6c', song_id='SOJLWPI12A6D4F93BC', playcount=1739)]


# sort_by_playcount = triplets.orderBy('playcount').argmax()
# sort_by_playcount.head(n=10)


### ALS Recomm



# convert user and song into an integer index
indexer_user = StringIndexer(inputCol='user_id', outputCol='user')
indexer_song = StringIndexer(inputCol='song_id', outputCol='item')

# set up pipeline #https://spark.apache.org/docs/latest/ml-pipeline.html
pipeline = Pipeline(stages=[indexer_user, indexer_song]) 
dataset = pipeline.fit(triplets).transform(triplets)

# convert playcount into double (as rating)
dataset = (
    dataset
    .withColumn('user', F.col('user').cast(IntegerType()))
    .withColumn('item', F.col('item').cast(IntegerType()))
    .withColumn('rating', F.col('playcount').cast(IntegerType()))
    .select(['user', 'item', 'rating'])
)

# print table
dataset.show(3, False)
  # +------+----+------+
  # |user  |item|rating|
  # +------+----+------+
  # |326041|2119|1     |
  # |326041|767 |3     |
  # |326041|1942|1     |
  # +------+----+------+



# +------+------+------+
# |user  |item  |rating|
# +------+------+------+
# |298837|314027|1     |
# |298837|27959 |2     |
# |298837|25826 |4     |
# +------+------+------+
# only showing top 3 rows






# create dictionary of fraction 30% for each user
f = (
    dataset
    .select('user')
    .dropDuplicates()
    .withColumn('temp', F.lit(0))
    .groupBy('temp')
    .agg(F.collect_list('user').alias('list'))
    .select('list')
    .rdd.take(1)[0].__getattr__('list')
)

fractions = dict(
    (user, 0.3) for user in f
)

# sample test set using fractions
test = dataset.sampleBy('user', fractions, seed=1)

# get training set by remove test set from full dataset
training = (
    dataset
    .join(
        test
        .select(['user', 'item']),
        on=['user', 'item'],
        how='left_anti'
    )
)

# count
test.count(), training.count()
  # (12849760, 29968400)

# create als model / Set the seed. 
als = ALS(seed=1)

# fit training set and tranform test set
model = als.fit(training)

# select three users
test.orderBy('rating', ascending=False).show(10, False)
  # +------+-----+------+
  # |user  |item |rating|
  # +------+-----+------+
  # |525941|18502|2948  |
  # |519321|30302|2381  |
  # |208949|48057|2368  |
  # |36740 |2144 |1862  |
  # |189453|67531|1460  |
  # |717383|2092 |1410  |
  # |254758|3855 |1202  |
  # |304135|47969|1174  |
  # |664862|16791|1167  |
  # |271179|152  |1135  |
  # +------+-----+------+

u = [525941, 519321, 208949]
users = sqlContext.createDataFrame(
    [(525941,), (519321,), (208949,)],
    StructType([StructField('user', IntegerType())])
)

# recommend for users
recommends = model.recommendForUserSubset(users, 10)


  # root
  #  |-- user_index: integer (nullable = false)
  #  |-- recommendations: array (nullable = true)
  #  |    |-- element: struct (containsNull = true)
  #  |    |    |-- song_index: integer (nullable = true)
  #  |    |    |-- rating: float (nullable = true)

# get prediction and labels
def recommend(recommendations):
    items = []
    for item, rating in recommendations:
        items.append(item)
    return items

udf_recommend = F.udf(lambda recommendations: recommend(recommendations), ArrayType(IntegerType()))

recommends = (
    recommends
    .withColumn('recommends', udf_recommend(recommends.recommendations))
    .select(
        F.col('user').cast(IntegerType()),
        F.col('recommends')
    )
)

recommends.show(3, False)
  # +------+------------------------------------------------------------------------------+
  # |user  |recommends                                                                    |
  # +------+------------------------------------------------------------------------------+
  # |525941|[241640, 134941, 18502, 103829, 248908, 263679, 155493, 125402, 179474, 85901]|
  # |208949|[241640, 215904, 208705, 251809, 183896, 45731, 277120, 286985, 258726, 60703]|
  # |519321|[209307, 107204, 127445, 80147, 166446, 196345, 69627, 149664, 217693, 163124]|
  # +------+------------------------------------------------------------------------------+

# get labels
ground_truths = (
    test
    .filter(F.col('user').isin(u))
    .orderBy('rating', ascending=False)
    .groupBy('user')
    .agg(F.collect_list('item').alias('ground_truths'))
)

ground_truths.show(3, False)
  # +------+------------------------------------------------------------------------------------------------------------------------------------+
  # |user  |ground_truths                                                                                                                              |
  # +------+------------------------------------------------------------------------------------------------------------------------------------+
  # |525941|[51266, 23962, 27235, 2993]                                                                                                         |
  # |208949|[0, 2642, 4800, 25028, 21565, 29, 120303, 284034, 103178, 5256, 8175, 405, 1559, 1276, 612, 32810, 36614, 21961, 41137, 40332, 2925]|
  # |519321|[3251, 31284, 128667, 85783, 86341, 30302, 2452, 16794, 2742]                                                                       |
  # +------+------------------------------------------------------------------------------------------------------------------------------------+

compare = recommends.join(ground_truths, on='user', how='left')
compare = [(r.__getattr__('recommends'), r.__getattr__('ground_truths')) for r in compare.collect()]
compare = sc.parallelize(compare)

#Alternative method
compare  = sc.parallelize(compare.collect()) # this take longer time than the above '_getattr_()' method. 



# print metrics
metrics = RankingMetrics(compare)
print(metrics.precisionAt(5))
  # 0.06666666666666667
print(metrics.ndcgAt(10))
  # 0.06506334166535027
print(metrics.meanAveragePrecision)
  # 0.027777777777777776

# predict test and rmse
predict = model.transform(test)
predict = predict.filter(F.col('prediction') != float('nan'))
reg_eval = RegressionEvaluator(predictionCol='prediction', labelCol='rating', metricName='rmse')
reg_eval.evaluate(predict)
  # 4.856047802562721

# testing NDCG metric on bad documents
set1 = sc.parallelize([([1, 2, 3], [1, 2])])
set2 = sc.parallelize([([1, 2], [1, 2])])
print('set 1: {}'.format(RankingMetrics(set1).ndcgAt(3)))
print('set 2: {}'.format(RankingMetrics(set2).ndcgAt(3)))
  # set 1: 1.0
  # set 2: 1.0

# testing NDCG metric on missing documents
set1 = sc.parallelize([([1, 2], [1, 2, 3])])
set2 = sc.parallelize([([1, 2, 3], [1, 2, 3])])
print('set 1: {}'.format(RankingMetrics(set1).ndcgAt(2)))
print('set 2: {}'.format(RankingMetrics(set2).ndcgAt(2)))
  # set 1: 1.0
  # set 2: 1.0

# load metadata dataset
metadata = (
    spark.read.format('com.databricks.spark.csv')
    .option('header', 'true')
    .option('inferSchema', 'false')
    .load(root + 'main/summary/metadata.csv.gz')
    .limit(1)
)

# print head
print(json.dumps(metadata.head().asDict(), indent=2))
  # {
  #   "analyzer_version": null,
  #   "artist_7digitalid": "4069",
  #   "artist_familiarity": "0.6498221002008776",
  #   "artist_hotttnesss": "0.3940318927141434",
  #   "artist_id": "ARYZTJS1187B98C555",
  #   "artist_latitude": null,
  #   "artist_location": null,
  #   "artist_longitude": null,
  #   "artist_mbid": "357ff05d-848a-44cf-b608-cb34b5701ae5",
  #   "artist_name": "Faster Pussy cat",
  #   "artist_playmeid": "44895",
  #   "genre": null,
  #   "idx_artist_terms": "0",
  #   "idx_similar_artists": "0",
  #   "release": "Monster Ballads X-Mas",
  #   "release_7digitalid": "633681",
  #   "song_hotttnesss": "0.5428987432910862",
  #   "song_id": "SOQMMHC12AB0180CB8",
  #   "title": "Silent Night",
  #   "track_7digitalid": "7032331"
  # }















## Some random results. 


# In [34]: user_activity.show(10, False)
# +----------------------------------------+---------+
# |user_id                                 |songcount|
# +----------------------------------------+---------+
# |bc7b298b0f49e513b3e704a89e936c6a1983607e|101      |
# |0bbea29eaec570da2426b7c0ce0fbcb7dbf70ba3|67       |
# |4ce9be8957e7d8659c3ff9ea8a736e54b1f1aa83|75       |
# |626480d9dab5934250439738de8168d438965e44|15       |
# |f351a14811ce331c52df762a762d43d92a199420|41       |
# |566f588a7b4d9acdbf1d966f4a2e8b3c311ff1f6|327      |
# |26be505a2073ff7f4b98662d9c3eb7425bf54e45|96       |
# |3263eb883df1fe9b885747d3003ee26dc10be10b|180      |
# |b9ea36b35366eb040b290403cd3ebb7a8a221962|13       |
# |74adee2c2b26f836c8ca536eb61bf8587e5cf092|46       |
# +----------------------------------------+---------+
# only showing top 10 rows


# In [35]: song_popularity.show(10,False)
# +------------------+---------+
# |song_id           |playcount|
# +------------------+---------+
# |SOLVTNS12AB01809E2|2695     |
# |SOWUCFL12AB0188263|2122     |
# |SODRFRJ12A8C144167|13543    |
# |SOBIMTY12A6D4F931F|13821    |
# |SOGZCOB12A8C14280E|13566    |
# |SOIXKRK12A8C140BD1|6184     |
# |SOJGSIO12A8C141DBF|16961    |
# |SOKEYJQ12A6D4F6132|20095    |
# |SOPDRWC12A8C141DDE|15484    |
# |SORRCNC12A8C13FDA9|15963    |
# +------------------+---------+
# only showing top 10 rows


# In [36]: triplets.show(3, False)
# +----------------------------------------+------------------+---------+
# |user_id                                 |song_id           |playcount|
# +----------------------------------------+------------------+---------+
# |00001cf0dce3fb22b0df0f3a1d9cd21e38385372|SOBDRND12A8C13FD08|1        |
# |00001cf0dce3fb22b0df0f3a1d9cd21e38385372|SOJAMXH12A8C138D9B|2        |
# |00001cf0dce3fb22b0df0f3a1d9cd21e38385372|SOLOVPR12AB0182D03|2        |
# +----------------------------------------+------------------+---------+
# only showing top 3 rows
