from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml import classification as CL

name = "step4"

logfile = open("log.step4","w")

spark = SparkSession.builder.appName(name).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# load the pipelinemodel, to know the vocabulary
fpath = "step2/dog_pipeline"
dogpipelineModel = PipelineModel.load(fpath)

# load the LogisticRegression model
fpath = "step2/dog_BinaryClassifier"
lr_dogModel = CL.LogisticRegressionModel.load(fpath)

mycoeff = lr_dogModel.coefficients
myintercept = lr_dogModel.intercept
myvocab = dogpipelineModel.stages[2].vocabulary

size_vocab = len(myvocab)
size_coeff = len(mycoeff)

logfile.write("\n\n dog_classifier\n")
logfile.write("\n size_of_vocabulary {}  size_of_coefficient {} \n\n".format(size_vocab, size_coeff))
logfile.flush

size = size_coeff  # it should equal to size_vocab

largestcoeff = 0.0
dogid = 0
pupid = 0
for i in range(0, size, 1) :
  if mycoeff[i] > largestcoeff :
    largestcoeff = mycoeff[i]

for i in range(0, size, 1) :
  if mycoeff[i] == largestcoeff :
    logfile.write("{} {} {}\n".format(i, mycoeff[i], myvocab[i].encode('utf-8')))
  if "dog" == myvocab[i] :
    dogid = i
  if "pup" == myvocab[i] :
    pupid = i

logfile.write("\n\n to compare:\n")
logfile.write("{} {} {}\n".format(dogid, mycoeff[dogid], myvocab[dogid]))
logfile.write("{} {} {}\n\n".format(pupid, mycoeff[pupid], myvocab[pupid]))


# here to print a csv file for the coefficient and vocabulary
outfilename = "output_step4_dog.csv"
fout = open(outfilename,"w")
fout.write("index,coefficient,word\n")
for i in range(0, size, 1) :
  fout.write("{},{},{}\n".format(i, mycoeff[i], myvocab[i].encode('utf-8')))  
  fout.flush()

fout.close()




# load the pipelinemodel, to know the vocabulary, for cat classifier
fpath = "step2/cat_pipeline"
catpipelineModel = PipelineModel.load(fpath)

# load the LogisticRegression model
fpath = "step2/cat_BinaryClassifier"
lr_catModel = CL.LogisticRegressionModel.load(fpath)

mycoeff = lr_catModel.coefficients
myintercept = lr_catModel.intercept
myvocab = catpipelineModel.stages[2].vocabulary

size_vocab = len(myvocab)
size_coeff = len(mycoeff)

logfile.write("\n\n cat_classifier\n")
logfile.write("\n size_of_vocabulary {}  size_of_coefficient {} \n\n".format(size_vocab, size_coeff))
logfile.flush

size = size_coeff

largestcoeff = 0.0
catid = 0
kittenid = 0
for i in range(0, size, 1) :
  if mycoeff[i] > largestcoeff :
    largestcoeff = mycoeff[i]

for i in range(0, size, 1) :
  if mycoeff[i] == largestcoeff :
    logfile.write("{} {} {}\n".format(i, mycoeff[i], myvocab[i].encode('utf-8')))
  if "cat" == myvocab[i] :
    catid = i
  if "kitten" == myvocab[i] :
    kittenid = i

logfile.write("\n\n to compare:\n")
logfile.write("{} {} {}\n".format(catid, mycoeff[catid], myvocab[catid]))
logfile.write("{} {} {}\n\n".format(kittenid, mycoeff[kittenid], myvocab[kittenid]))



# here to print a csv file for the coefficient and vocabulary
outfilename = "output_step4_cat.csv"
fout1 = open(outfilename,"w")
fout1.write("index,coefficient,word\n")
for i in range(0, size, 1) :
  fout1.write("{},{},{}\n".format(i, mycoeff[i], myvocab[i].encode('utf-8')))  
  fout1.flush()

fout1.close()



logfile.close()







