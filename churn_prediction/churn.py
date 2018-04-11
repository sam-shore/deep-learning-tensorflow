#import packages
import pandas as pd
import numpy as np
import sklearn.preprocessing
import tensorflow as tf
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#read in data
df = pd.read_csv('Churn_Modelling.csv')
df_test = pd.read_csv('/Users/SamShore/Desktop/Churn-Modelling-Artificial-Neural-Network-master/Churn_Modelling_Test.csv')

#View Data Labels
print ("Left")
print (df.RowNumber[df.Exited == 1].describe())
print ()
print ("Stayed")
print (df.RowNumber[df.Exited == 0].describe())

#Data preprocessing
numpy_df = df.values
numpy_df_test = df_test.values

labels = df['Exited']
test_labels = df_test['Exited']

data = labels.values
test_data = test_labels.values

values = array(data)

test_values = array(test_data)

features = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
test_features = df_test.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)

features = features.values
test_features = test_features.values

#one-hot encoding the labels
label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values)
integer_encoded_test = label_encoder.fit_transform(test_values)

onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)

labels = onehot_encoder.fit_transform(integer_encoded)
test_labels = onehot_encoder.fit_transform(integer_encoded_test)

#turning country and gender into values
features[:, 1] = label_encoder.fit_transform(features[:, 1])
features[:, 2] = label_encoder.fit_transform(features[:, 2])

test_features[:, 1] = label_encoder.fit_transform(test_features[:, 1])
test_features[:, 2] = label_encoder.fit_transform(test_features[:, 2])

#shuffle data
shuffle = np.random.choice(len(features), len(features), replace=False)
X_values = features[shuffle]
y_values = labels[shuffle]

shuffle_test = np.random.choice(len(test_features), len(test_features), replace=False)
test_values = test_features[shuffle_test]
test_labels = test_labels[shuffle_test]

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size = 0.2, random_state = 0)

#transform data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
test_values = sc.fit_transform(test_values)

length_X_test = len(X_test)
length_X_test
#neural net model
X = tf.placeholder(tf.float32, shape=[None, 10])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

w1 = tf.Variable(tf.random_normal(shape=[10, 6]))
b1 = tf.Variable(tf.random_normal(shape=[6]))

w2 = tf.Variable(tf.random_normal(shape=[6, 2]))
b2 = tf.Variable(tf.random_normal(shape=[2]))

interval = 50
epoch = 4000

Y1 = tf.nn.relu(tf.add(tf.matmul(X, w1), b1))
Y2 = tf.nn.softmax(tf.add(tf.matmul(Y1, w2), b2))

with tf.name_scope("loss"):
    loss = -tf.reduce_sum(y_ * tf.log(Y2))
    optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)    # Add scalar summary for cost tensor
    tf.summary.scalar("loss", loss)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(Y2, 1)) # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
    # Add scalar summary for accuracy tensor
    tf.summary.scalar("accuracy", acc_op)

"""
with tf.name_scope("per_class_accuracy"):
    y_test_class = tf.convert_to_tensor(test_labels)
    X_test_class = tf.convert_to_tensor(test_values)
    X_test_class = np.swapaxes(test_values, 0,1)
    X_test_class.shape
    acc_class = tf.metrics.accuracy(y_test_class, X_test_class)
    tf.summary.scalar("per_class_accuracy", acc_class)
"""

with tf.Session() as sess:
    #writer = tf.summary.FileWriter("/Users/SamShore/Documents/ML/logs/nn_logs", sess.graph)
    fw = tf.summary.FileWriter("../logs/churn_log-max_data")
    summaries_op = tf.summary.merge_all()

    #initialize all variables
    tf.initialize_all_variables().run()

# Training
    print('Training the model...')
    for i in range(1, (epoch + 1)):
        result = sess.run([summaries_op, acc_op], feed_dict={X: X_train, y_: y_train})
        summary_str = result[0]
        acc = result[1]
        fw.add_summary(summary_str, i)
        sess.run(optimizer, feed_dict={X: X_train, y_: y_train})
        if i % interval == 0:
            print('Epoch', i, '|', 'Loss:', sess.run(loss, feed_dict={X: X_train, y_: y_train}))

    # Prediction
    count = 0
    for i in range(len(X_test)):
        feed_dic = sess.run(Y2, feed_dict={X: [X_test[i]]})
        #print('Actual:', y_test[i], 'Predicted:', np.around(feed_dic, 2))
        actual  = y_test[i][1]
        predicted = np.rint(feed_dic[0][1])
        predicted_unround = feed_dic[0][1]
        print ('actual', ' ; ', 'churn_prob', ' ; ','predicted')
        print (actual, ' ; ', predicted_unround, ' ; ',predicted)
        if predicted == actual:
            count +=1
    exited_acc = count/length_X_test
    print ('Accuracy of excited: ', exited_acc)
