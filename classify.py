import numpy as np
import glob
import scipy as scp
import scipy.misc
import tensorflow as tf
import random


def prepare_data(class_number):
    source_img_ruptured_list = []
    source_img_unruptured_list = []
    source_img_normal_list = []

    for i in glob.glob('C:\\Sources\\Data\\Temp\\Dataset\\Ruptured\\*.jpg'):
        img = scp.misc.imread(i)
        #img = scp.misc.imresize(img, (128, 128))
        source_img_ruptured_list.append(img)
    for i in glob.glob('C:\\Sources\\Data\\Temp\\Dataset\\Unruptured\\*.jpg'):
        img = scp.misc.imread(i)
        #img = scp.misc.imresize(img, (128, 128))
        source_img_unruptured_list.append(img)
    for i in glob.glob('C:\\Sources\\Data\\Temp\\Dataset\\Normal\\*.jpg'):
        img = scp.misc.imread(i)
        #img = scp.misc.imresize(img, (128, 128))
        source_img_normal_list.append(img)

    source_img_ruptured_list = np.array(source_img_ruptured_list)
    source_img_unruptured_list = np.array(source_img_unruptured_list)
    source_img_normal_list = np.array(source_img_normal_list)

    source_img_ruptured_list = source_img_ruptured_list.reshape(-1, 128, 128, 1).astype(np.float32)
    #source_img_ruptured_list = source_img_ruptured_list.transpose(0, 2, 3, 1)
    source_img_unruptured_list = source_img_unruptured_list.reshape(-1, 128, 128, 1).astype(np.float32)
    #source_img_unruptured_list = source_img_unruptured_list.transpose(0, 2, 3, 1)
    source_img_normal_list = source_img_normal_list.reshape(-1, 128, 128, 1).astype(np.float32)
    #source_img_normal_list = source_img_normal_list.transpose(0, 2, 3, 1)

    label_ruptured = np.zeros((len(source_img_ruptured_list), class_number))
    label_unruptured = np.zeros((len(source_img_unruptured_list), class_number))
    label_normal = np.zeros((len(source_img_normal_list), class_number))
    label_ruptured[:, 0] = 1
    label_unruptured[:, 1] = 1
    label_normal[:, 2] = 1

    source_img = np.concatenate((source_img_ruptured_list, source_img_unruptured_list, source_img_normal_list), 0)
    label = np.concatenate((label_ruptured, label_unruptured, label_normal), 0)

    data = list(zip(source_img, label))

    print("Image load complete!")

    return data


def get_batch_data(batch_size, count, data):
    total_length = len(data)
    try:
        repeat = total_length / batch_size
        remain = total_length % batch_size
    except ZeroDivisionError:
        print("ZeroDivision")
    batch_start = batch_size * count

    result_source_img = []
    result_label = []

    if batch_size == total_length:
        for i in range(total_length):
            temp = data[i]
            result_source_img.append(temp[0])
            result_label.append(temp[1])

        result_source_img = np.array(result_source_img)
        result_label = np.array(result_label)

        return result_source_img, result_label

    if (batch_start + remain) == total_length:
        for i in range(batch_start, total_length):
            temp = data[i]
            result_source_img.append(temp[0])
            result_label.append(temp[1])

        result_source_img = np.array(result_source_img)
        result_label = np.array(result_label)

        return result_source_img, result_label

    else:
        for i in range(batch_start, batch_start + batch_size):
            temp = data[i]
            result_source_img.append(temp[0])
            result_label.append(temp[1])

        result_source_img = np.array(result_source_img)
        result_label = np.array(result_label)

        return result_source_img, result_label



class_num = 3
learning_rate = 0.001
training_epochs = 30
batch_size = 10
width = 128
height = 128
data = prepare_data(class_num)

keep_prob = tf.placeholder(tf.float32)
input_image = tf.placeholder(tf.float32, [None, width, height, 1])
label = tf.placeholder(tf.float32, [None, class_num])

filters = {
    'cf1': tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.01)),
    'cf2': tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01)),
    'cf3': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01)),
    'cf4': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
}

cl1 = tf.nn.conv2d(input_image, filters['cf1'], strides=[1, 1, 1, 1], padding='SAME')
cl1 = tf.nn.max_pool(cl1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#cl1 : [-1, width / 2, height / 2, 32]

cl2 = tf.nn.conv2d(cl1, filters['cf2'], strides=[1, 1, 1, 1], padding='SAME')
cl2 = tf.nn.relu(cl2)
cl2 = tf.nn.max_pool(cl2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
cl2 = tf.nn.dropout(cl2, keep_prob=keep_prob)
#cl2 : [-1, width / 4, height / 4, 64]

cl3 = tf.nn.conv2d(cl2, filters['cf3'], strides=[1, 1, 1, 1],  padding='SAME')
cl3 = tf.nn.max_pool(cl3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#cl3 : [-1, width / 8, height / 8, 64]

cl4 = tf.nn.conv2d(cl3, filters['cf4'], strides=[1, 1, 1, 1],  padding='SAME')
cl4 = tf.nn.relu(cl4)
cl4 = tf.nn.max_pool(cl4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
cl4 = tf.nn.dropout(cl4, keep_prob=keep_prob)
#cl4 : [-1, width / 16, height / 16, 128]

flat_cl4 = tf.reshape(cl4, [-1, int((width / 16) * (height / 16) * 128)])

fc1_w = tf.get_variable("fc1_w", shape=[(width / 16) * (height / 16) * 128, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
fc1 = tf.nn.relu(tf.matmul(flat_cl4, fc1_w) + b1)
fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

fc2_w = tf.get_variable("fc2_w", shape=[256, 3], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([3]))
classify_layer = tf.matmul(fc1, fc2_w) + b2

classify_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=classify_layer, labels=label), name='Classification_Cost')
classify_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(classify_cost)
classify_prediction = tf.equal(tf.argmax(classify_layer, 1), tf.argmax(label, 1))
classify_accuracy = tf.reduce_mean(tf.cast(classify_prediction, tf.float32), name='Classification_Accuracy')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    random.shuffle(data)
    training_data = data[:int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9) + 1:]
    test_source_img, test_label = get_batch_data(int(len(test_data)), 0, test_data)

    for i in range(training_epochs):
        print("-------Epoch", i + 1, "Training Start!!-------")
        random.shuffle(training_data)
        eval_data = training_data[int(len(training_data)*0.9)+1:]
        eval_source_img, eval_label = get_batch_data(int(len(eval_data)), 0, eval_data)

        training_steps = int(len(training_data) / batch_size)

        for j in range(training_steps):
            batch_s_img, batch_l = get_batch_data(batch_size, j, training_data)
            class_cost, _, = sess.run([classify_cost, classify_optimizer], feed_dict={input_image: batch_s_img, label: batch_l, keep_prob: 0.8})
            if (j % (batch_size / 2)) == 0:
                eval_c_cost, eval_c_acc = sess.run([classify_cost, classify_accuracy], feed_dict={input_image: eval_source_img, label: eval_label, keep_prob: 1.0})
                print("Evaluation Classification Cost : ", '{:5f}'.format(eval_c_cost), "\nEvaluation Classification Accuracy : ", '{:5f}'.format(eval_c_acc))

    print("-------------Training Finished-------------")
    test_c_cost, test_c_acc = sess.run([classify_cost, classify_accuracy],
                                       feed_dict={input_image: test_source_img, label: test_label, keep_prob: 1.0})
    print("-------------Test results-------------")
    print("Classification Cost : ", '{:5f}'.format(test_c_cost), "\nClassification Accuracy : ",
          '{:5f}'.format(test_c_acc))