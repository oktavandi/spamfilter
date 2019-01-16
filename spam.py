from collections import Counter
import csv

#ambil dataset
f = open('data.csv')
reader = csv.reader(f)

#misah dataset
data = []
label = []
for row in reader:
    data.append(row[0])
    label.append(row[1])


print("jumlah data:{}".format(len(data)))
 

#random urutan dan split ke data training dan test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( data, label, test_size=0.1)

print("Data training:")
 
## X_train = data
## y_train = label
## X_test = data baru yg akan ditesting
## y_test = label data yg akan ditesting

print("Data training:")
print(len(X_train))
print(Counter(y_train))
 
print("Data testing:")
print(len(X_test))
print(Counter(y_test))

#transform ke tfidf dan train dengan naive bayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', MultinomialNB())])
text_clf.fit(X_train, y_train)

# coba prediksi data baru
sms_baru = ['Mas Agus']
print(sms_baru[0])
pred = text_clf.predict(sms_baru)
print("Hasil prediksi {}".format(pred))

#hitung akurasi data test
import numpy as np
pred = text_clf.predict(X_test)

akurasi = np.mean(pred==y_test)
print("Akurasi: {}".format(akurasi))
