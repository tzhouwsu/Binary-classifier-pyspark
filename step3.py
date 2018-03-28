from pyspark.sql import SparkSession
from pyspark.sql import types as TP
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml import classification as CL
from os import system

name = "step3"
logfile = open("log.step3","w")

spark = SparkSession.builder.appName(name).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# first load the data into a DataFrame
myschema1 = TP.StructType([
  TP.StructField("userid", TP.IntegerType()),
  TP.StructField("0", TP.IntegerType()),
  TP.StructField("1", TP.IntegerType()),
  TP.StructField("2", TP.IntegerType()),
  TP.StructField("3", TP.IntegerType()),
  TP.StructField("dog_owner", TP.IntegerType()),
  TP.StructField("cat_owner", TP.IntegerType())
])

df1 = spark.read.csv('step1/output_step1_1.csv', header=True, schema=myschema1).na.drop()
df1sort = df1.orderBy(F.asc("userid"))
#df1sort.show()

def myremovespace(istring) :
  if istring :
    out = istring.replace(' ','')
  else :
    out = "0"
  return out

myrmsp = F.udf(myremovespace, TP.StringType())

df2 = spark.read.csv('animals_comments_1.csv', header=True, encoding = "utf-8").na.drop()
df2sort = (
  df2
  .select("creator_name","userid","comment")
  .withColumn("userid2", myrmsp(df2.userid).cast(TP.IntegerType()) )
  .groupBy("userid2")
  .agg(F.concat_ws("  ", F.collect_list("comment") ).alias("allcomments"))
  .orderBy(F.asc("userid2"))
)
#df2sort.show()

dfcombine = (  
  df2sort.join(df1sort, df1sort.userid == df2sort.userid2, how='left_outer')
  .select(df1sort.userid, df1sort.dog_owner, df1sort.cat_owner, df2sort.allcomments)
)
#dfcombine.show()


# read the pipelineModel fitted from step2, for dog classifier
path = "step2/dog_pipeline"
mypipelineModel = PipelineModel.load(path)
dataset_dog = mypipelineModel.transform(dfcombine)

mydatacols = ["userid","features","label"]  # I'm add "userid" column for joining the prediction, feature may be duplicated
dataset1 = dataset_dog.select(mydatacols)
#dataset1.show()

# read the previously saved logistic regression model
path = "step2/dog_BinaryClassifier"
lr_dogModel = CL.LogisticRegressionModel.load(path)

predictions = lr_dogModel.transform(dataset1)

# get the fraction of users who are cat/dog owner
total = dataset_dog.count()
myP_dog = predictions.filter(predictions.label == 1).count()  # this is ~ fraction from label
mypred_P_dog = predictions.filter(predictions.prediction == 1).count()  # this is ~ fraction from model prediction

logfile.write("\n\n total {} myP_dog {} my_pred_P_dog {} \n\n".format(total, myP_dog, mypred_P_dog))
logfile.flush()

dfout1 = (
  dataset_dog.join(predictions, dataset_dog.userid == predictions.userid, how='left_outer') # join by "userid"
  .select(dataset_dog.userid, predictions.prediction.alias("dog_prediction"))
)


## below is for cat pipelines and cat classifier
path = "step2/cat_pipeline"
mypipelineModel = PipelineModel.load(path)
dataset_cat = mypipelineModel.transform(dfcombine)

mydatacols = ["userid","features","label"]
dataset2 = dataset_cat.select(mydatacols)
#dataset2.show()

# read the previously saved cat logistic regression model
path = "step2/cat_BinaryClassifier"
lr_catModel = CL.LogisticRegressionModel.load(path)

predictions = lr_catModel.transform(dataset2)

# get the fraction of users who are cat/dog owner
total = dataset_cat.count()
myP_cat = predictions.filter(predictions.label == 1).count()  # this is ~ fraction from label
mypred_P_cat = predictions.filter(predictions.prediction == 1).count()  # this is ~ fraction from model prediction

logfile.write("\n\n total {} myP_cat {} my_pred_P_cat {} \n\n".format(total, myP_cat, mypred_P_cat))
logfile.flush()

dfout2 = (
  dataset_cat.join(predictions, dataset_cat.userid == predictions.userid, how='left_outer')
  .select(dataset_cat.userid.alias("userid2"), predictions.prediction.alias("cat_prediction"))
)
#dfout2.show()

dftotal = (
  dfout1.join(dfout2, dfout1.userid == dfout2.userid2, how='left_outer')
  .select(dfout1.userid, dfout1.dog_prediction, dfout2.cat_prediction)
)
#dftotal.show()

# save the predictions to file
dftotal.repartition(1).write.csv(name, header=True)

system('mv %s/part* %s/my_pred_%s.csv' % (name, name, name) )



logfile.close()




