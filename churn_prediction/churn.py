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
df = pd.read_csv('churn_prediction/churn_data.csv')

#View Data Labels
print ("Exited")
print (df.RowNumber[df.Exited == 1].describe())
print ()
print ("Stayed")
print (df.RowNumber[df.Exited == 0].describe())

#Data preprocessing
numpy_df = df.values
labels = df['Exited']
data = labels.values
values = array(data)

features = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
features = features.values

#one-hot encoding the labels
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
labels = onehot_encoder.fit_transform(integer_encoded)

#turning country and gender into values
features[:, 1] = label_encoder.fit_transform(features[:, 1])
features[:, 2] = label_encoder.fit_transform(features[:, 2])

#shuffle data
shuffle = np.random.choice(len(features), len(features), replace=False)
X_values = features[shuffle]
y_values = labels[shuffle]

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size = 0.2, random_state = 0)

#transform data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#find length of traing data, used to predict class accuracy later
length_X_test = len(X_test)

#neural net model
X = tf.placeholder(tf.float32, shape=[None, 10])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

w1 = tf.Variable(tf.random_normal(shape=[10, 6]))
b1 = tf.Variable(tf.random_normal(shape=[6]))

w2 = tf.Variable(tf.random_normal(shape=[6, 2]))
b2 = tf.Variable(tf.random_normal(shape=[2]))

interval = 50
epoch = 4000
LR = .005

Y1 = tf.nn.relu(tf.add(tf.matmul(X, w1), b1))
Y2 = tf.nn.softmax(tf.add(tf.matmul(Y1, w2), b2))

with tf.name_scope("loss"):
    loss = -tf.reduce_sum(y_ * tf.log(Y2))
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss)    # Add scalar summary for cost tensor
    tf.summary.scalar("loss", loss)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(Y2, 1)) # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
    # Add scalar summary for accuracy tensor
    tf.summary.scalar("accuracy", acc_op)

with tf.Session() as sess:
    #tensorboard config ... use: tensorboard --logdir=/logs/churn/
    fw = tf.summary.FileWriter("churn_prediction/logs/churn", sess.graph)
    summaries_op = tf.summary.merge_all()

    #initialize all variables
    tf.initialize_all_variables().run()

    #Training
    print('Training the model...')
    for i in range(1, (epoch + 1)):
        result = sess.run([summaries_op, acc_op], feed_dict={X: X_train, y_: y_train})
        summary_str = result[0]
        acc = result[1]
        fw.add_summary(summary_str, i)
        sess.run(optimizer, feed_dict={X: X_train, y_: y_train})
        if i % interval == 0:
            print('Epoch', i, '|', 'Loss:', sess.run(loss, feed_dict={X: X_train, y_: y_train}))

    #Prediction/Accuracy
    count = 0
    print ('Actual', ' ; ', 'Churn Probability', ' ; ','Churn?')
    for i in range(len(X_test)):
        feed_dic = sess.run(Y2, feed_dict={X: [X_test[i]]})
        #other prediction output option
        #print('Actual:', y_test[i], 'Predicted:', np.around(feed_dic, 2))
        actual  = y_test[i][1]
        predicted = np.rint(feed_dic[0][1])
        probability_churn = np.around(feed_dic[0][1], 2)
        print (actual, ' ; ', probability_churn, ' ; ',predicted)
        if predicted == actual:
            count +=1
    exited_acc = count/length_X_test
    print ('Churn Accuracy: ', exited_acc)
