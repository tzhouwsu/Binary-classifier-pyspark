from pyspark.sql import SparkSession
from pyspark.sql import types as TP
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml import feature as FT
from pyspark.ml import classification as CL
from pyspark.ml import evaluation as EV
from pyspark.ml import tuning as TU

name = "step2"
logfile = open("log.step2","w")

spark = SparkSession.builder.appName(name).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# first load the data
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

def myremovespace(istring) :  # this is to remove the space in "userid" in original csv
  if istring :
    out = istring.replace(' ','')
  else :
    out = "0"
  return out

myrmsp = F.udf(myremovespace, TP.StringType())

df2 = spark.read.csv('animals_comments_1.csv', header=True).na.drop()
df2sort = (
  df2
  .select("creator_name","userid","comment")
  .withColumn("userid2", myrmsp(df2.userid).cast(TP.IntegerType()) )
  .groupBy("userid2")
  .agg(F.concat_ws("  ", F.collect_list("comment") ).alias("allcomments"))  # combine all coments of a user
  .orderBy("userid2")
)
#df2sort.show()

dfcombine = (
  df2sort.join(df1sort, df1sort.userid == df2sort.userid2, how='left_outer')
  .select(df1sort.userid, df1sort.dog_owner, df1sort.cat_owner, df2sort.allcomments)
)
#dfcombine.show()

logfile.write("\n\n data-size for build model: {} \n\n".format(dfcombine.count()))
logfile.flush()


# second process data, I'm using allcomments as features, and treat output as categorical (though it is already numerical)
mystages = []
mylabelindexer = FT.StringIndexer(inputCol="dog_owner", outputCol="label")  # it is treated as catogorical
mystages += [mylabelindexer]

# take "allcomments" as input, need to token it into words, and vectorize it
myregexTokenizer = FT.RegexTokenizer(inputCol="allcomments", outputCol="words", toLowercase=True, pattern="\\W") 
mycountVectors = FT.CountVectorizer(inputCol="words", outputCol="features")
mystages += [myregexTokenizer, mycountVectors]


mypipeline = Pipeline(stages=mystages) # create a pipeline for data transform
mypipelineModel = mypipeline.fit(dfcombine)
dataset = mypipelineModel.transform(dfcombine)


# below is to fit & test dog_owner classifier
mydatacols = ["features","label"]
dataset = dataset.select(mydatacols)
#dataset.show()

# split the dataset into train and test
(traindata, testdata) = dataset.randomSplit([0.7,0.3], seed = 12345)
numtrain = traindata.count()
numtest = testdata.count()
logfile.write("\n\n dog_classifier :\n traindata# {}\n testdata# {}".format(numtrain,numtest))
logfile.flush()

# use logistic regression for classification
lr_dog = CL.LogisticRegression(featuresCol="features", labelCol="label")
lr_dogModel = lr_dog.fit(traindata)
predictions = lr_dogModel.transform(testdata)

# below is to evaluate the classifier
evaluator = EV.BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
performance = evaluator.evaluate(predictions)

myregparam = lr_dogModel._java_obj.getRegParam()
myelasticnetparam = lr_dogModel._java_obj.getElasticNetParam()

logfile.write("\n default parameter-set {}, {} :  evaluate {}\n".format(myregparam, myelasticnetparam, performance))
logfile.flush()

# then use several ParamGrid for cross validation, to find best hyperparameters, I comment it for same-time
paramGrid = (
  TU.ParamGridBuilder()
  .addGrid(lr_dog.regParam, [0.0, 0.5, 1.0]) # weight of the penalty, how much reguliation to use
  .addGrid(lr_dog.elasticNetParam, [0.0, 0.5, 1.0]) # penalty of L2, L1 regularization
  .build()
)

cv_dog = TU.CrossValidator(estimator=lr_dog, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
cv_dogModel = cv_dog.fit(traindata)
best_dogModel = cv_dogModel.bestModel

predictions = best_dogModel.transform(testdata)
performance = evaluator.evaluate(predictions)

myregparam = best_dogModel._java_obj.getRegParam()
myelasticnetparam = best_dogModel._java_obj.getElasticNetParam()

logfile.write("\n best-model parameter-set {}, {} :  evaluate {}\n".format(myregparam, myelasticnetparam, performance))
logfile.flush()

# below are simple measure of the performance, i.e. to get true-positive-rate, and true-negative-rate
myP = predictions.filter(predictions.label == 1).count()
myN = predictions.filter(predictions.label == 0).count()
myTP = predictions.filter(predictions.label == 1).filter(predictions.prediction == 1).count()
myTN = predictions.filter(predictions.label == 0).filter(predictions.prediction == 0).count()

logfile.write("\n performance evaluate: {}\n P {} TP {} N {} TN {}\n\n".format(performance, myP, myTP, myN, myTN))
logfile.flush()

# below is to same the model & pipeline
savedfname = '%s/dog_BinaryClassifier' % name
best_dogModel.save(savedfname)  # this is the best-model in cross validation
savedfname = "%s/dog_pipeline" % name
mypipelineModel.save(savedfname)




# below is for cat classifier, the same as dog, but use "cat_owner" as label
mystages = []
mylabelindexer = FT.StringIndexer(inputCol="cat_owner", outputCol="label")
mystages += [mylabelindexer]

# take "allcomments" as input 
myregexTokenizer = FT.RegexTokenizer(inputCol="allcomments", outputCol="words", toLowercase=True, pattern="\\W")
mycountVectors = FT.CountVectorizer(inputCol="words", outputCol="features")
mystages += [myregexTokenizer, mycountVectors]

mypipeline = Pipeline(stages=mystages) # create a pipeline
mypipelineModel = mypipeline.fit(dfcombine)
dataset = mypipelineModel.transform(dfcombine)

# below is to fit & test cat_owner classifier
mydatacols = ["features","label"]
dataset = dataset.select(mydatacols)
#dataset.show()

# split the dataset into train and test
(traindata, testdata) = dataset.randomSplit([0.7,0.3], seed = 12345)
numtrain = traindata.count()
numtest = testdata.count()
logfile.write("\n\n cat_classifier :\n traindata# {}\n testdata# {}".format(numtrain,numtest))
logfile.flush()

# use logistic regression for classification
lr_cat = CL.LogisticRegression(featuresCol="features", labelCol="label")
lr_catModel = lr_cat.fit(traindata)
predictions = lr_catModel.transform(testdata)

# below is to evaluate the classifier
evaluator = EV.BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
performance = evaluator.evaluate(predictions)

myregparam = lr_catModel._java_obj.getRegParam()
myelasticnetparam = lr_catModel._java_obj.getElasticNetParam()

logfile.write("\n default parameter-set {}, {} :  evaluate {}\n".format(myregparam, myelasticnetparam, performance))
logfile.flush()

# then use several ParamGrid for cross validation, to find best hyperparameters, I comment it for same-time
paramGrid = (
  TU.ParamGridBuilder()
  .addGrid(lr_dog.regParam, [0.0, 0.5, 1.0]) # weight of the penalty, how much reguliation to use
  .addGrid(lr_dog.elasticNetParam, [0.0, 0.5, 1.0]) # penalty of L2, L1 regularization
  .build()
)

cv_cat = TU.CrossValidator(estimator=lr_cat, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
cv_catModel = cv_cat.fit(traindata)
best_catModel = cv_catModel.bestModel

predictions = best_catModel.transform(testdata)
performance = evaluator.evaluate(predictions)

myregparam = best_catModel._java_obj.getRegParam()
myelasticnetparam = best_catModel._java_obj.getElasticNetParam()

logfile.write("\n default parameter-set {}, {} :  evaluate {}\n".format(myregparam, myelasticnetparam, performance))
logfile.flush()

# below are simple measure of the performance
myP = predictions.filter(predictions.label == 1).count()
myN = predictions.filter(predictions.label == 0).count()
myTP = predictions.filter(predictions.label == 1).filter(predictions.prediction == 1).count()
myTN = predictions.filter(predictions.label == 0).filter(predictions.prediction == 0).count()

logfile.write("\n performance evaluate: {}\n P {} TP {} N {} TN {}\n\n".format(performance, myP, myTP, myN, myTN))
logfile.flush()

# below is to same the model
savedfname = '%s/cat_BinaryClassifier' % name
best_catModel.save(savedfname)
savedfname = "%s/cat_pipeline" % name
mypipelineModel.save(savedfname)


logfile.close()




