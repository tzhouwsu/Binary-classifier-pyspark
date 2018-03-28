
# This is an example for using logistic regression for text-classification

the perpose is to create a classifier that can select the dog/cat owners based on their comments made on some Youtube channel creator

## dataset description

the raw dataset (animals_comments_1): creator_name, userid, comment

creator_name: name of the Youtube channel creator

userid:  user id for posting comments

comment:  the text of the comment


## step1: identify cat and dog owners

Use simple string search of "dog", "pup", "cat", "kitten", in the comment to identify cat/dog owner.

## step2: build and evaluate classifiers

Use logistic classification (pyspark.ml), with the combined-comments of a user (all his comments) as input-feature, and the owner_tag from step1 as output-label.

Use RegexTokenizer and CountVectorizer to transform the combined-comments to a feature vector

## step3: classify all the users

Load the classifier that is saved in step2, and apply it on all the data-set-users, to identify cat and/or dog owners.

## step4: find most important keywords associated with cat/dog owners

Find the largest weight-coefficient in the classifier, and the corresponding keywords. Since I use CountVectorizer, the keywords with the largest coefficient are the most important

## step5: identify creators with cat/dog owners in the audience

Try to group and count the number_of_users (audience) for each creator, the number_of_dog_owner_users (and fraction), the number_of_cat_owner_users (and fraction).





