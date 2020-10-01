# SMS-Spam-Classifier
> By Aayush Jain (Darkshadow)

* Created a machine learning model which will classify whether the SMS received is SPAM or HAM(not a spam message).
* The dataset used here is from Kaggle which has 5500 records of messages being SPAM or HAM.
* We will use Scikit Learn's Multinomial classifier to get the results.

---

## Code and Resources Used:
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, nltk, re  
**Dataset:** [Kaggle Link](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

---

## Steps:
1. Import the libraries.
2. Select the dataset.
3. Remove the regular expression and Lemmatize it.
4. Create Tfidf vector.
5. Split the data.
6. Train and Test the data.
7. After that to get answer for the new user inputs we can perform ```model.predict(cv.tranform(user_input).toarray())```

---

## Model Accuracy:
* The Multinomial Naive Bayes helped to achieve 98% of accuracy in classifying SPAM or HAM
