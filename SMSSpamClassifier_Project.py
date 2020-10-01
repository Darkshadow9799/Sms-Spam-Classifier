## Importing libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score

## Importing dataset
messages=pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])

## Data Preprocessing
#ps=PorterStemmer()
lm=WordNetLemmatizer()
corpus=[]
for i in range(len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

#cv=CountVectorizer(max_features=5000)
cv=TfidfVectorizer(max_features=5000)
x=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

## Train-Test Split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

## Model fitting 
spam_detect_model=MultinomialNB().fit(X_train,y_train)

y_pred=spam_detect_model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
