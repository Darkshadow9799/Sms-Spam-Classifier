# SMS-Spam-Classifier
> By Aayush Jain (Darkshadow)

* The SMS Spam Dataset from Kaggle is used to in the project to classify whether a SMS received is a spam or not(here HAM).
* The Multinomial Classifier is used in this.

## Steps:
* Import the libraries.
* Select the dataset.
* Remove the regular expression and Lemmatize it.
* Create Word2Vec.
* Split the data.
* Train and Test the data.
* After that to get answer for the new user inputs we can do model.predict(cv.tranform(user_input).toarray())

## Accuracy:
* The accuracy of the model is around 98%.

## Libraries used:
* nltk
* re
* pandas
* sklearn
