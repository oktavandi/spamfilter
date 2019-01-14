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

#train test split ??
 
## X_train = data
## y_train = label
## X_test = data baru yg akan ditesting
## y_test = label data yg akan ditesting
## predict ????
print("Data training:")
#train test split ??
 
## X_train = data
## y_train = label
## X_test = data baru yg akan ditesting
## y_test = label data yg akan ditesting
## predict ????
print("Data training:")
print(len(X_train))
print(Counter(y_train))
 
print("Data testing:")
print(len(X_test))
print(Counter(y_test))
