import os

import pandas as pd
import requests

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

# This code is inspired from Google's notebook in the link below:
# https://github.com/googledatalab/notebooks/blob/master/samples/TensorFlow/Machine%20Learning%20with%20Financial%20Data.ipynb

API_KEY = '91x_HX5rSkHJT7xNMgUb'

codes = ["WSE/OPONEO_PL", "WSE/VINDEXUS", "WSE/WAWEL", "WSE/WIELTON", "WIKI/SNPS"]
start_date = '2010-01-01'
end_date = '2015-01-01'
order = 'asc'
column_index = 4

train_test_split = 0.8

data_specs = 'start_date={}&end_date={}&order={}&column_index={}&api_key={}'.format(start_date, end_date, order,
                                                                                    column_index, API_KEY)

base_url = "https://www.quandl.com/api/v3/datasets/{}/{" \
           "}/data.json?" + data_specs

output_path = os.path.realpath('../../datasets/TimeSeries')

if not os.path.isdir(os.path.realpath('../../datasets/TimeSeries')):
    os.makedirs(output_path)


def show_plot(key="", show=True):
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(15)

    for code in codes:
        index = code.split("/")[1]
        if key and len(key) > 0:
            label = "{}_{}".format(index, key)
        else:
            label = index
        _ = plt.plot(closings[label], label=label)

    _ = plt.legend(loc='upper right')

    if show:
        plt.show()


closings = pd.DataFrame()

for code in codes:
    code_splits = code.split("/")
    stock_exchange = code_splits[0]
    index = code_splits[1]

    stock_data = requests.get(base_url.format(stock_exchange, index)).json()
    dataset_data = stock_data['dataset_data']
    data = np.array(dataset_data['data'])
    closings[index] = pd.Series(data[:, 1].astype(float))
    closings[index + "_scaled"] = closings[index] / max(closings[index])
    closings[index + "_log_return"] = np.log(closings[index] / closings[index].shift())

closings = closings.fillna(method='ffill')  # Fill the gaps in data

show = False

show_plot("", show=show)
show_plot("scaled", show=show)
show_plot("log_return", show=show)

# Features and labels selection
feature_columns = ['SNPS_log_return_positive', 'SNPS_log_return_negative']
for i in range(len(codes)):
    index = codes[i].split("/")[1]
    feature_columns.extend([
        '{}_log_return_1'.format(index),
        '{}_log_return_2'.format(index),
        '{}_log_return_3'.format(index)
    ])

features_and_labels = pd.DataFrame(
    columns=feature_columns)

closings['SNPS_log_return_positive'] = 0
closings.ix[closings['SNPS_log_return'] >= 0, 'SNPS_log_return_positive'] = 1
closings['SNPS_log_return_negative'] = 0
closings.ix[closings['SNPS_log_return'] < 0, 'SNPS_log_return_negative'] = 1

for i in range(7, len(closings)):
    feed_dict = {
        'SNPS_log_return_positive': closings['SNPS_log_return_positive'].ix[i],
        'SNPS_log_return_negative': closings['SNPS_log_return_negative'].ix[i]
    }

    for j in range(len(codes)):
        index = codes[j].split("/")[1]
        k = 1 if j == len(codes) - 1 else 0
        feed_dict.update(
            {
                '{}_log_return_1'.format(index): closings['{}_log_return'.format(index)].ix[i - k],
                '{}_log_return_2'.format(index): closings['{}_log_return'.format(index)].ix[i - 1 - k],
                '{}_log_return_3'.format(index): closings['{}_log_return'.format(index)].ix[i - 2 - k]
            })

    features_and_labels = features_and_labels.append(feed_dict, ignore_index=True)

features = features_and_labels[features_and_labels.columns[2:]]
labels = features_and_labels[features_and_labels.columns[:2]]

train_size = int(len(features_and_labels) * train_test_split)
test_size = len(features_and_labels) - train_size

train_features = features[:train_size]
train_labels = labels[:train_size]
test_features = features[train_size:]
test_labels = labels[train_size:]


def tf_confusion_metrics(model, actual_classes, session, feed_dict):
    predictions = tf.argmax(model, 1)
    actuals = tf.argmax(actual_classes, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    tp, tn, fp, fn = \
        session.run(
            [tp_op, tn_op, fp_op, fn_op],
            feed_dict
        )

    tpfn = float(tp) + float(fn)
    tpr = 0 if tpfn == 0 else float(tp) / tpfn
    # fpr = 0 if tpfn == 0 else float(fp) / tpfn

    total = float(tp) + float(fp) + float(fn) + float(tn)
    accuracy = 0 if total == 0 else (float(tp) + float(tn)) / total

    recall = tpr
    tpfp = float(tp) + float(fp)
    precision = 0 if tpfp == 0 else float(tp) / tpfp

    f1_score = 0 if recall == 0 else (2 * (precision * recall)) / (precision + recall)

    print('Precision = ', precision)
    print('Recall = ', recall)
    print('F1 Score = ', f1_score)
    print('Accuracy = ', accuracy)


sess = tf.Session()

num_predictors = len(train_features.columns)
num_classes = len(train_labels.columns)

feature_data = tf.placeholder("float", [None, num_predictors])
actual_classes = tf.placeholder("float", [None, 2])

weights1 = tf.Variable(tf.truncated_normal([len(codes) * 3, 50], stddev=0.0001))
biases1 = tf.Variable(tf.ones([50]))

weights2 = tf.Variable(tf.truncated_normal([50, 25], stddev=0.0001))
biases2 = tf.Variable(tf.ones([25]))

weights3 = tf.Variable(tf.truncated_normal([25, 2], stddev=0.0001))
biases3 = tf.Variable(tf.ones([2]))

hidden_layer_1 = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1)
hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)

cost = -tf.reduce_sum(actual_classes * tf.log(model))

train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

init = tf.initialize_all_variables()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

for i in range(1, 30001):
    sess.run(
        train_op1,
        feed_dict={
            feature_data: train_features.values,
            actual_classes: train_labels.values.reshape(len(train_labels.values), 2)
        }
    )
    if i % 5000 == 0:
        print(i, sess.run(
            accuracy,
            feed_dict={
                feature_data: train_features.values,
                actual_classes: train_labels.values.reshape(len(train_labels.values), 2)
            }
        ))

feed_dict = {
    feature_data: test_features.values,
    actual_classes: test_labels.values.reshape(len(test_labels.values), 2)
}

tf_confusion_metrics(model, actual_classes, sess, feed_dict)
