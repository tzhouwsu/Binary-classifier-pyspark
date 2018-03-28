from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as TP
from os import system

name = "step1"
logfile = open("log.step1","w")

spark = SparkSession.builder.appName(name).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# this function is to determine whether this comment indicats the user is dog/cat owner
def mycheckpet(icomment) :
  dognamelist = ["dog", "pup"]  # this is the keyword list for identifying dog owner
  catnamelist = ["cat", "kitten"]
  if icomment == None :   
    out = 0
  else :
    num_dog = 0  # how many times "dog" is mentioned in icomment
    for key in dognamelist :
      if icomment.lower().count(key) >= 1 :
        num_dog = 1
        break
    num_cat = 0
    for key in catnamelist :
      if icomment.lower().count(key) >= 1 :
        num_cat = 1
        break

    if num_dog == 1 :
      if num_cat == 1 :  
        out = 1   # both dog and cat
      else :
        out = 2   # dog owner only
    else :
      if num_cat == 1 :
        out = 3   # cat owner only
      else :
        out = 0   # neither dog nor cat
  return out


# this is to determine if this user is a dog/cat owner, based on how many comments that support the determination
def my_identify_owner (num1, num2) :
   result = 0
   if( num1 == 0 and num2 ==0 ) :
     result = 0
   else :
     result = 1
   return result

# define some functions
checkpet = F.udf(mycheckpet, TP.IntegerType())  # whether a comment indicates the user is dog/cat owner
identify_owner = F.udf(my_identify_owner, TP.IntegerType())

df = spark.read.csv('animals_comments_1.csv', header=True, encoding='utf-8').na.drop()

dfnew = (
  df
  .withColumn("checkpet", F.lit( checkpet(F.col("comment")) ) )  # search keyword per comment
  .groupBy("userid").pivot("checkpet").agg(F.count("checkpet").alias("count")).na.fill(0) # this is to find how many comments that support this user is a dog/cat owner
  .withColumn("dog_owner", F.lit( identify_owner(F.col("1"), F.col("2")) ) )
  .withColumn("cat_owner", F.lit( identify_owner(F.col("1"), F.col("3")) ) )
#  .show(100)
)


dfnew.repartition(1).write.csv(name, header=True)

system('mv %s/part* %s/output_%s_1.csv' % (name, name, name) )

total = dfnew.count()
dogowner = dfnew.filter(F.col("dog_owner") == 1).count()
catowner = dfnew.filter(F.col("cat_owner") == 1).count()

logfile.write("\n total user#: {}\n dog_owner#: {}\n cat_owner#: {}\n\n".format(total,dogowner,catowner))

logfile.close()



