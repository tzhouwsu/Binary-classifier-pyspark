from pyspark.sql import SparkSession
from pyspark.sql import types as TP
from pyspark.sql import functions as F
from os import system

name = "step5"
logfile = open("log.step5","w")

spark = SparkSession.builder.appName(name).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# first load the data into a DataFrame
myschema = TP.StructType([
  TP.StructField("userid", TP.IntegerType()),
  TP.StructField("dog_prediction", TP.FloatType()),
  TP.StructField("cat_prediction", TP.FloatType())
])

df1 = spark.read.csv('step3/my_pred_step3.csv', header=True, schema=myschema).na.drop()
#df1.show()

def myremovespace(istring) :  # remove the heading space in "userid"
  if istring :
    out = istring.replace(' ','')
  else :
    out = "0"
  return out

myrmsp = F.udf(myremovespace, TP.StringType())

df2 = spark.read.csv('animals_comments_1.csv', header=True).na.drop()
df2 = (
  df2
  .withColumn("userid2", myrmsp(df2.userid).cast(TP.IntegerType()) ) # make it to integer to join
  .select("creator_name","userid2")
  .dropDuplicates(["creator_name", "userid2"])  # remove duplicates, a user make many comments on one creator_name will only count once
)


def calfrac(x, y) :
  if y <= 0 :
    out = 0.0
  else :
    out = (x + 0.0) / (y + 0.0)
  return out

mycalfrac = F.udf(calfrac, TP.FloatType())  # this function is to calculate fraction

dfcombine = (  
  df2.join(df1, df1.userid == df2.userid2, how='left_outer')
  .select(df1.userid, df2.creator_name, df1.dog_prediction, df1.cat_prediction)
  .groupBy("creator_name")
  .agg(F.sum("dog_prediction").alias("N_dog_owner"), F.sum("cat_prediction").alias("N_cat_owner"), F.count("userid").alias("N_audience"))
  .withColumn("percent_dog_owner", F.lit( mycalfrac(F.col("N_dog_owner"), F.col("N_audience")) ) )
  .withColumn("percent_cat_owner", F.lit( mycalfrac(F.col("N_cat_owner"), F.col("N_audience")) ) )
)
#dfcombine.show(100)


# order the dataframe, by N_dog_owner, i.e. find creators with most dog owners
logfile.write("\n\n find creators with the most dog owners\n")
logfile.flush()

dfdog1 = (
  dfcombine
  .orderBy(dfcombine.N_dog_owner.desc())
)
namedog1 = name + "_Ndog"
dfdog1.repartition(1).write.csv(namedog1, header=True)

system('mv %s/part* %s/output_%s.csv' % (namedog1, namedog1, namedog1) )


# find the creators with highest percentage of dog owners
logfile.write("\n\n find creators with the highest percentage of dog owners\n")
logfile.flush()

dfdog2 = (
  dfcombine
  .orderBy(dfcombine.percent_dog_owner.desc())
)
namedog2 = name + "_fdog"
dfdog2.repartition(1).write.csv(namedog2, header=True)

system('mv %s/part* %s/output_%s.csv' % (namedog2, namedog2, namedog2) )


# below are for cat
logfile.write("\n\n find creators with the most cat owners\n")
logfile.flush()

dfcat1 = (
  dfcombine
  .orderBy(dfcombine.N_cat_owner.desc())
)
namecat1 = name + "_Ncat"
dfcat1.repartition(1).write.csv(namecat1, header=True)

system('mv %s/part* %s/output_%s.csv' % (namecat1, namecat1, namecat1) )


logfile.write("\n\n find creators with the highest percentage of cat owners\n")
logfile.flush()

dfcat2 = (
  dfcombine
  .orderBy(dfcombine.percent_cat_owner.desc())
)
namecat2 = name + "_fcat"
dfcat2.repartition(1).write.csv(namecat2, header=True)

system('mv %s/part* %s/output_%s.csv' % (namecat2, namecat2, namecat2) )



logfile.close()


