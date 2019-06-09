# imports
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.mllib.stat import Statistics
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, OneVsRest 
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import json
import pandas as pd

# Q1 -----------------------------------------------------------------------
# set root directory path
root = 'hdfs:///data/msd/'

# list of audio filenames
filenames = [
    'msd-jmir-area-of-moments-all-v1.0',
    'msd-jmir-lpc-all-v1.0',
    'msd-jmir-methods-of-moments-all-v1.0',
    'msd-jmir-mfcc-all-v1.0',
    'msd-jmir-spectral-all-all-v1.0',
    'msd-jmir-spectral-derivatives-all-all-v1.0',
    'msd-marsyas-timbral-v1.0',
    'msd-mvd-v1.0',
    'msd-rh-v1.0',
    'msd-rp-v1.0',
    'msd-ssd-v1.0',
    'msd-trh-v1.0',
    'msd-tssd-v1.0'
]

# function to read features into dataframe
def read_features(filename):
    # create path
    path_attr = root + 'audio/attributes/' + filename + '.attributes.csv'
    path_feat = root + 'audio/features/' + filename + '.csv/*'

    # create schema from attributes
    dtypes = {
        'real': DoubleType(),
        'numeric': DoubleType(),
        'string': StringType()
    }
    schema = StructType()
    for row in spark.read.csv(path_attr).collect():
        colname = row[0].lower()
        if colname in ['msd_trackid', 'track_id', 'instancename']:
            colname = 'track_id'
        dtype = row[1].lower()
        schema.add(StructField(colname, dtypes[dtype], True))

    # read features
    df = (
        spark.read.format('com.databricks.spark.csv')
        .option('header', 'false')
        .option('inferSchema', 'true')
        .schema(schema)
        .load(path_feat)
        .withColumn('track_id', F.regexp_replace('track_id', "'", ''))
    )
    return df

# Loading small feature data set. 

features = read_features('msd-jmir-methods-of-moments-all-v1.0')   #Ref : https://docs.python.org/2/library/json.html
# Print the feature heads
print(json.dumps(features.head().asDict(), indent=2))

# {
#   "method_of_moments_overall_standard_deviation_1": 0.1545,
#   "method_of_moments_overall_standard_deviation_2": 13.11,
#   "method_of_moments_overall_standard_deviation_3": 840.0,
#   "method_of_moments_overall_standard_deviation_4": 41080.0,
#   "method_of_moments_overall_standard_deviation_5": 7108000.0,
#   "method_of_moments_overall_average_1": 0.319,
#   "method_of_moments_overall_average_2": 33.41,
#   "method_of_moments_overall_average_3": 1371.0,
#   "method_of_moments_overall_average_4": 64240.0,
#   "method_of_moments_overall_average_5": 8398000.0,
#   "track_id": "TRHFHQZ12903C9E2D5"
# }




# count
features.count()
  # 994623

# get statistics
print(features.drop('track_id').describe().toPandas().transpose())
 
#                                                      0                    1                    2          3         4
# summary                                          count                 mean               stddev        min       max
# method_of_moments_overall_standard_deviation_1  994623   0.1549817600174655  0.06646213086143041        0.0     0.959
# method_of_moments_overall_standard_deviation_2  994623   10.384550576951835   3.8680013938747018        0.0     55.42
# method_of_moments_overall_standard_deviation_3  994623    526.8139724398112    180.4377549977511        0.0    2919.0
# method_of_moments_overall_standard_deviation_4  994623    35071.97543290272   12806.816272955532        0.0  407100.0
# method_of_moments_overall_standard_deviation_5  994623    5297870.369577217   2089356.4364557962        0.0   4.657E7
# method_of_moments_overall_average_1             994623   0.3508444432531261   0.1855795683438387        0.0     2.647
# method_of_moments_overall_average_2             994623    27.46386798784021    8.352648595163698        0.0     117.0
# method_of_moments_overall_average_3             994623   1495.8091812075486   505.89376391902437        0.0    5834.0
# method_of_moments_overall_average_4             994623   143165.46163257837   50494.276171032136  -146300.0  452500.0
# method_of_moments_overall_average_5             994623  2.396783048473542E7    9307340.299219608        0.0   9.477E7








# get correlation matrix
rdd = features.drop('track_id').rdd.map(lambda row: row[0:])
corr_mat = Statistics.corr(rdd, method='pearson')
corr_df = pd.DataFrame(corr_mat).round(2)

print(corr_df.to_string())
  
#       0     1     2     3     4     5     6     7     8     9
# 0  1.00  0.43  0.30  0.06 -0.06  0.75  0.50  0.45  0.17  0.10
# 1  0.43  1.00  0.86  0.61  0.43  0.03  0.41  0.40  0.02 -0.04
# 2  0.30  0.86  1.00  0.80  0.68 -0.08  0.13  0.18 -0.09 -0.14
# 3  0.06  0.61  0.80  1.00  0.94 -0.33 -0.22 -0.16 -0.25 -0.22
# 4 -0.06  0.43  0.68  0.94  1.00 -0.39 -0.36 -0.29 -0.26 -0.21
# 5  0.75  0.03 -0.08 -0.33 -0.39  1.00  0.55  0.52  0.35  0.28
# 6  0.50  0.41  0.13 -0.22 -0.36  0.55  1.00  0.90  0.52  0.42
# 7  0.45  0.40  0.18 -0.16 -0.29  0.52  0.90  1.00  0.77  0.69
# 8  0.17  0.02 -0.09 -0.25 -0.26  0.35  0.52  0.77  1.00  0.98
# 9  0.10 -0.04 -0.14 -0.22 -0.21  0.28  0.42  0.69  0.98  1.00








# read genre
Genre1 = (
    spark.read.format('com.databricks.spark.csv')
    .option('delimiter', '\t')
    .option('header', 'false')
    .option('inferSchema', 'true')
    .schema(
        StructType([
            StructField('track_id', StringType()),
            StructField('genre', StringType())
        ])
    )
    .load(root + 'genre/msd-MAGD-genreAssignment.tsv')
)
  
# print table
Genre1.show(3, False)
  # +------------------+--------+
  # |track_id          |genre   |
  # +------------------+--------+
  # |TRAAAAK128F9318786|Pop_Rock|
  # |TRAAAAV128F421A322|Pop_Rock|
  # |TRAAAAW128F429D538|Rap     |
  # +------------------+--------+

# count
Genre1.count()
  # 422714

# genre distribution
genre_dist = (
    Genre1
    .groupBy('genre')
    .count()
    .orderBy('count', ascending=False)
)

# print table
genre_dist.show(3, False)
#   +----------+------+
#   |genre     |count |
#   +----------+------+
#   |Pop_Rock  |238786|
#   |Electronic|41075 |
#   |Rap       |20939 |
#   +----------+------+

# save output as parquet file
genre_dist.write.parquet('hdfs:///user/gtj13/outputs/msd/genre_dist1')

genre_dist.coalesce(1).write.format('com.databricks.spark.csv').options(delimiter = '\t').save('hdfs:///user/gtj13/outputs/msd/genre_dist1', mode = 'overwrite', header = True)


#Take the fileout and plot.

# hdfs dfs -copyToLocal hdfs:///user/gtj13/outputs/msd/genre_dist1/ /users/home/gtj13




# join features with genre, remove missing genre, check is_electronic

features = (
    features
    .join(
     Genre1,
        on='track_id',
        how='left'
    )
    .filter(F.col('genre').isNotNull())
)

#df.na.drop(subset=["dt_mvmt"])
#features1.na.drop(subset=["genre"])




# print table
features.select(['track_id', 'genre']).show(3, False)
  # +------------------+--------+
  # |track_id          |genre   |
  # +------------------+--------+
  # |TRAAAAK128F9318786|Pop_Rock|
  # |TRAAAAV128F421A322|Pop_Rock|
  # |TRAAAAW128F429D538|Rap     |
  # +------------------+--------+

# count after removed missing genre
features1.count()
  # 420620

# Q2 -----------------------------------------------------------------------
# convert features into vector
assembler = (
    VectorAssembler()
    .setInputCols([x for x in features.columns if x.startswith('method')])
    .setOutputCol('features')
)




# normalize vector
scaler = StandardScaler(inputCol='features', outputCol='scfeatures', withStd=True, withMean=False)

# set up pipeline
pipeline = Pipeline(stages=[assembler, scaler])
dataset = pipeline.fit(features).transform(features)

#model = Pipeline(stages=[assembler,scaler]).fit(features).transform(features)


# print table
dataset.select(['track_id', 'genre', 'features', 'scfeatures']).show(3, 30)

# +------------------+--------------+------------------------------+------------------------------+
# |          track_id|         genre|                      features|                    scfeatures|
# +------------------+--------------+------------------------------+------------------------------+
# |TRAAABD128F429CF47|      Pop_Rock|[0.1308,9.587,459.9,27280.0...|[2.022118802771498,2.624321...|
# |TRAAADT12903CCC339|Easy_Listening|[0.08392,7.541,423.7,36060....|[1.2973716355396339,2.06425...|
# |TRAAAEF128F4273421|      Pop_Rock|[0.1199,9.381,474.5,26990.0...|[1.8536089025405398,2.56793...|
# +------------------+--------------+------------------------------+------------------------------+
# only showing top 3 rows

# define algorithms

lr = LogisticRegression(featuresCol='scfeatures')
rf = RandomForestClassifier(featuresCol='scfeatures')
dt = DecisionTreeClassifier(featuresCol='scfeatures')


# parameter grid and cross-validator

lr_cv = CrossValidator(
    estimator=lr,
    estimatorParamMaps=(
        ParamGridBuilder()
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
        .addGrid(lr.regParam, [0.0, 0.25, 0.5])
        .build()
    ),
    evaluator=MulticlassClassificationEvaluator(metricName='f1'),
    numFolds=5
)


rf_cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=(
        ParamGridBuilder()
        .addGrid(rf.featureSubsetStrategy, ['auto','all', 'onethird', 'sqrt'])
        .addGrid(dt.maxBins, [16, 32, 64])
        .addGrid(dt.maxDepth, [1, 5, 10])
        .addGrid(rf.impurity, ['gini', 'entropy'])
        .build()
    ),
    evaluator=MulticlassClassificationEvaluator(metricName='f1'),
    numFolds=5
)


dt_cv = CrossValidator(
    estimator=dt,
    estimatorParamMaps=(
        ParamGridBuilder()
        .addGrid(dt.maxBins, [16, 32, 64])
        .addGrid(dt.maxDepth, [1, 5, 10])
        .addGrid(dt.impurity, ['gini', 'entropy'])
        .build()
    ),
    evaluator=MulticlassClassificationEvaluator(metricName='f1'),
    numFolds=5
)


#Ref : https://stackoverflow.com/questions/52820144/in-pyspark-how-to-group-after-a-partitionby-and-orderby / https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/

def binary(algorithms, dataset):
    # create label
    df = dataset.withColumn('label', F.when(F.col('genre') == 'Electronic', 1.0).otherwise(0.0))

    # split training / test
    training, test = df.randomSplit([0.7, 0.3])

    # get class balance
    class_0 = training.filter(F.col('label') == 0.0).count()
    class_1 = training.filter(F.col('label') == 1.0).count()

    # create dictionary of fraction of each class
    fractions = dict()
    fractions[0.0] = min(class_0, class_1) / class_0
    fractions[1.0] = min(class_0, class_1) / class_1

    # down-sampling training using fractions
    training_bal = training.sampleBy('label', fractions, seed=1)   #https://stackoverflow.com/questions/39994587/spark-train-test-split

    # print head
    str = '{:>7}|{:>7}|{:>7}|{:>7}|{:>10}|{:>10}|{:>10}|{:>10}|{}'
    print(str.format('tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'f1', 'auc', 'algorithm'))


    # iterate algorithms , Ref  : https://stackabuse.com/cross-validation-and-grid-search-for-model-selection-in-python/
    for name, normal, cv in algorithms:
        # train
        model_nor = normal.fit(training)
        model_bal = normal.fit(training_bal)

        # predict
        predict_nor = model_nor.transform(test)
        predict_bal = model_bal.transform(test)

        metrics = [('normal', predict_nor), ('balance', predict_bal)]

        # cv
        if cv != None:
            model_cv = cv.fit(training_bal)
            predict_cv = model_cv.transform(test)
            metrics.append(('cv', predict_cv))

        # metrics  https://en.wikipedia.org/wiki/Sensitivity_and_specificity / https://en.wikipedia.org/wiki/Sensitivity_and_specificity /https://en.wikipedia.org/wiki/False_positive_rate
        for mtype, d in metrics:
            tp = d.filter((F.col('label') == 1.0) & (F.col('prediction') == 1.0)).count()
            fp = d.filter((F.col('label') == 0.0) & (F.col('prediction') == 1.0)).count()
            tn = d.filter((F.col('label') == 0.0) & (F.col('prediction') == 0.0)).count()
            fn = d.filter((F.col('label') == 1.0) & (F.col('prediction') == 0.0)).count()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            eval_binary = BinaryClassificationEvaluator()
            auc = eval_binary.evaluate(d, {eval_binary.metricName: 'areaUnderROC'})
            str = '{:>7}|{:>7}|{:>7}|{:>7}|{:>10.3f}|{:>10.3f}|{:>10.3f}|{:>10.3f}|{}'
            print(str.format(tp, fp, tn, fn, precision, recall, f1, auc, name + ' (' + mtype +')'))

binary(
    [
        ('logistic regression', lr, lr_cv),
        ('random forest', rf, rf_cv),
        ('decision tree', dt, dt_cv)
    ],
    dataset
)
    

   #   tp|     fp|     tn|     fn| precision|    recall|        f1|       auc|algorithm
   #  349|    379| 113471|  11837|     0.479|     0.029|     0.054|     0.763|logistic regression (normal)
   # 8769|  35608|  78242|   3417|     0.198|     0.720|     0.310|     0.769|logistic regression (balance)
   # 8769|  35608|  78242|   3417|     0.198|     0.720|     0.310|     0.769|logistic regression (cv)
   #    0|      0| 113850|  12186|     0.000|     0.000|     0.000|     0.754|random forest (normal)
   # 8476|  31917|  81933|   3710|     0.210|     0.696|     0.322|     0.781|random forest (balance)
   # 8476|  31917|  81933|   3710|     0.210|     0.696|     0.322|     0.781|random forest (cv)
   #  279|    195| 113655|  11907|     0.589|     0.023|     0.044|     0.372|decision tree (normal)
   # 8206|  31478|  82372|   3980|     0.207|     0.673|     0.316|     0.674|decision tree (balance)
   # 8366|  27506|  86344|   3820|     0.233|     0.687|     0.348|     0.586|decision tree (cv)




# Q3 -----------------------------------------------------------------------

def multiclass(algorithms, dataset):
    # create label
    indexer = StringIndexer(inputCol='genre', outputCol='label') #https://spark.apache.org/docs/2.1.0/ml-features.html|https://stackoverflow.com/questions/36942233/apply-stringindexer-to-several-columns-in-a-pyspark-dataframe

    # set up pipeline
    pipeline = Pipeline(stages=[indexer])
    df = pipeline.fit(dataset).transform(dataset)

    # split training / test
    training, test = df.randomSplit([0.7, 0.3])

    # get class distribution
    class_dist = (
        df
        .groupBy(['genre', 'label'])
        .count()
        .withColumn('fraction', F.when(F.col('count') < 5000, 1).otherwise(5000 / F.col('count')))
        .orderBy('label')
    )

# create dictionary of fraction of each class
    
    fractions = dict()
    for row in class_dist.collect():
        fractions[row.label] = row.fraction

    # down-sampling audio features using fractions
    training_bal = training.sampleBy('label', fractions, seed=1)

    # print head
    str = '{:>10}|{:>10}|{:>10}|{:>10}|{}'
    print(str.format('accuracy', 'precision', 'recall', 'f1', 'algorithm'))

    # iterate algorithms
    for name, normal, cv in algorithms:
        # train
        model_nor = normal.fit(training)
        model_bal = normal.fit(training_bal)
        model_ovr = OneVsRest(classifier=normal).fit(training)

        # predict
        predict_nor = model_nor.transform(test)
        predict_bal = model_bal.transform(test)
        predict_ovr = model_ovr.transform(test)

        metrics = [('normal', predict_nor), ('balance', predict_bal), ('one-vs-rest', predict_ovr)]

        # cv
        if cv != None:
            model_cv = cv.fit(training)
            predict_cv = model_cv.transform(test)
            metrics.append(('cv', predict_cv))

        # metrics
        for mtype, d in metrics:
            eval_multi = MulticlassClassificationEvaluator()
            accuracy = eval_multi.evaluate(d, {eval_multi.metricName: 'accuracy'})
            precision = eval_multi.evaluate(d, {eval_multi.metricName: 'weightedPrecision'})
            recall = eval_multi.evaluate(d, {eval_multi.metricName: 'weightedRecall'})
            f1 = eval_multi.evaluate(d, {eval_multi.metricName: 'f1'})
            str = '{:>10.3f}|{:>10.3f}|{:>10.3f}|{:>10.3f}|{}'
            print(str.format(accuracy, precision, recall, f1, name + ' (' + mtype +')'))

multiclass(
    [
        ('random forest', rf, rf_cv)
    ],
    dataset
)
    # accuracy| precision|    recall|        f1|algorithm
    #    0.571|     0.394|     0.571|     0.430|random forest (normal)
    #    0.269|     0.566|     0.269|     0.329|random forest (balance)
    #    0.578|     0.401|     0.578|     0.448|random forest (one-vs-rest)
    #    0.575|     0.397|     0.575|     0.449|random forest (cv)

#ref https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html

 # accuracy| precision|    recall|        f1|algorithm
 #     0.569|     0.371|     0.569|     0.423|random forest (normal)
 #     0.271|     0.567|     0.271|     0.330|random forest (balance)
 #     0.565|     0.388|     0.565|     0.411|random forest (one-vs-rest)
 #     0.573|     0.396|     0.573|     0.448|random forest (cv)



#https://stackoverflow.com/questions/52847408/pyspark-extract-roc-curve

output_path = os.path.expanduser("~/plots")  # M:/plots on windows
if not os.path.exists(output_path):
  os.makedirs(output_path)


def plot_roc_curve(fpr, tpr):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


f.savefig(os.path.join(output_path, f"tags_combined.png"), bbox_inches="tight")  # save as png and view in windows
plt.close(f)



