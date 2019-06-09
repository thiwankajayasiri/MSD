# imports
from pyspark.sql.types import *
from pyspark.sql import functions as F
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

# print table
mismatches.show(3, False)
# +------------------+------------------+
# |song_id           |track_id          |
# +------------------+------------------+
# |SOUMNSI12AB0182807|TRMMGKQ128F9325E10|
# |SOCMRBE12AB018C546|TRMMREB12903CEB1B1|
# |SOLPHZY12AC468ABA8|TRMMBOC12903CEB46E|
# +------------------+------------------+

# count
mismatches.count()
  # 19094

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

# print table
triplets.show(3, False)
# +----------------------------------------+------------------+---------+
# |user_id                                 |song_id           |playcount|
# +----------------------------------------+------------------+---------+
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQEFDN12AB017C52B|1        |
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQOIUJ12A6701DAA7|2        |
# |f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQOKKD12A6701F92E|4        |
# +----------------------------------------+------------------+---------+

# count
triplets.count()
  # 48373586

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

# count after removed mismatches
triplets.count()
  # 45795100
