# DATA PROCESSING ============================================================


#awk

hadoop fs -ls -R /data/msd | awk '{print $8}' | \
sed -e 's/[^-][^\/]*\//--/g' -e 's/^/ /' -e 's/-/|/'



# read data structure --------------------------------------------------------
# get directory tree
hdfs dfs -ls -R hdfs:///data/msd > msd_filename.txt

# get directory sizes for top directories
hdfs dfs -du -h -v hdfs:///data/msd
  # SIZE     DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
  # 12.3 G   98.1 G                                 hdfs:///data/msd/audio
  # 30.1 M   241.0 M                                hdfs:///data/msd/genre
  # 174.4 M  1.4 G                                  hdfs:///data/msd/main
  # 490.4 M  3.8 G                                  hdfs:///data/msd/tasteprofile

# get directory sizes for audio directories
hdfs dfs -du -h -v hdfs:///data/msd/audio
  # SIZE     DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
  # 103.0 K  824.3 K                                hdfs:///data/msd/audio/attributes
  # 12.2 G   97.8 G                                 hdfs:///data/msd/audio/features
  # 40.3 M   322.1 M                                hdfs:///data/msd/audio/statistics

# get line counts for audio attributes
for i in $(hdfs dfs -find hdfs:///data/msd/audio/attributes -name '*.*')
do
  (hdfs dfs -cat $i | wc -l)
done
  # 21
  # 21
  # 11
  # 27
  # 17
  # 17
  # 125
  # 421
  # 61
  # 1441
  # 169
  # 421
  # 1177

# get line counts for audio features
for i in $(hdfs dfs -find hdfs:///data/msd/audio/features -name 'msd-*')
do
  (hdfs dfs -cat $i/* | gunzip | wc -l)
done
  # 994623
  # 994623
  # 994623
  # 994623
  # 994623
  # 994623
  # 995001
  # 994188
  # 994188
  # 994188
  # 994188
  # 994188
  # 994188

# get line counts for audio statistics
hdfs dfs -cat hdfs:///data/msd/audio/statistics/* | gunzip | wc -l
 # 992866

# get line counts for genre
for i in $(hdfs dfs -find hdfs:///data/msd/genre -name '*.*')
do
  (hdfs dfs -cat $i | wc -l)
done
  # 422714
  # 273936
  # 406427

# get line counts for main summary
for i in $(hdfs dfs -find hdfs:///data/msd/main/summary -name '*.*')
do
  (hdfs dfs -cat $i | gunzip | wc -l)
done
  # 1000001
  # 1000001

# get line counts for tasteprofile mismatches
for i in $(hdfs dfs -find hdfs:///data/msd/tasteprofile/mismatches -name '*.*')
do
  (hdfs dfs -cat $i | wc -l)
done
  # 938
  # 19094

# get line counts for tasteprofile triplets
hdfs dfs -cat hdfs:///data/msd/tasteprofile/triplets.tsv/* | gunzip | wc -l
  # 48373586
