import numpy as np
import glob
import scipy as scp
import scipy.misc
import tensorflow as tf
import random


def training_data(class_number):
    source_img_ruptured_list = []
    source_img_unruptured_list = []
    source_img_normal_list = []

    for i in glob.glob('C:\\train\\Ruptured\\*.jpg'):
        img = scp.misc.imread(i)
        #img = scp.misc.imresize(img, (128, 128))
        source_img_ruptured_list.append(img)
    for i in glob.glob('C:\\train\\Unruptured\\*.jpg'):
        img = scp.misc.imread(i)
        #img = scp.misc.imresize(img, (128, 128))
        source_img_unruptured_list.append(img)
    for i in glob.glob('C:\\train\\Normal\\*.jpg'):
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


def test_data(class_number):
    source_img_ruptured_list = []
    source_img_unruptured_list = []
    source_img_normal_list = []

    for i in glob.glob('C:\\test\\Ruptured\\*.jpg'):
        img = scp.misc.imread(i)
        #img = scp.misc.imresize(img, (128, 128))
        source_img_ruptured_list.append(img)
    for i in glob.glob('C:\\test\\Unruptured\\*.jpg'):
        img = scp.misc.imread(i)
        #img = scp.misc.imresize(img, (128, 128))
        source_img_unruptured_list.append(img)
    for i in glob.glob('C:\\test\\Normal\\*.jpg'):
        img = scp.misc.imread(i)
        img = scp.misc.imresize(img, (128, 128))
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


def class2_test_data(class_number):
    source_img_ruptured_list = []
    source_img_normal_list = []

    for i in glob.glob('C:\\test\\Ruptured\\*.jpg'):
        img = scp.misc.imread(i)
        #img = scp.misc.imresize(img, (128, 128))
        source_img_ruptured_list.append(img)
    for i in glob.glob('C:\\test\\Normal\\*.jpg'):
        img = scp.misc.imread(i)
        #img = scp.misc.imresize(img, (128, 128))
        source_img_normal_list.append(img)

    source_img_ruptured_list = np.array(source_img_ruptured_list)
    source_img_normal_list = np.array(source_img_normal_list)

    source_img_ruptured_list = source_img_ruptured_list.reshape(-1, 128, 128, 1).astype(np.float32)
    #source_img_ruptured_list = source_img_ruptured_list.transpose(0, 2, 3, 1)
    source_img_normal_list = source_img_normal_list.reshape(-1, 128, 128, 1).astype(np.float32)
    #source_img_normal_list = source_img_normal_list.transpose(0, 2, 3, 1)

    label_ruptured = np.zeros((len(source_img_ruptured_list), class_number))
    label_normal = np.zeros((len(source_img_normal_list), class_number))
    label_ruptured[:, 0] = 1
    label_normal[:, 2] = 1

    source_img = np.concatenate((source_img_ruptured_list, source_img_normal_list), 0)
    label = np.concatenate((label_ruptured, label_normal), 0)

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

    if batch_size >= total_length:
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
learning_rate = 0.0005
training_epochs = 30
batch_size = 10
width = 128
height = 128
train_data = training_data(class_num)
test_data = test_data(class_num)
rup_nor_test_data = class2_test_data(3)

keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
input_image = tf.placeholder(tf.float32, [None, width, height, 1])
label = tf.placeholder(tf.float32, [None, class_num])

filters = {
    'cf1': tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)),
    'cf2': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)),
    'cf3': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)),
    'cf4': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
}

cl1_bn = tf.layers.batch_normalization(inputs=input_image, axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, training=is_training)
cl1 = tf.nn.conv2d(cl1_bn, filters['cf1'], strides=[1, 1, 1, 1], padding='SAME')
cl1 = tf.nn.max_pool(cl1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#cl1 : [-1, width / 2, height / 2, 32]

cl2_bn = tf.layers.batch_normalization(inputs=cl1, axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, training=is_training)
cl2 = tf.nn.conv2d(cl2_bn, filters['cf2'], strides=[1, 1, 1, 1], padding='SAME')
cl2 = tf.nn.relu(cl2)
cl2 = tf.nn.max_pool(cl2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
cl2 = tf.nn.dropout(cl2, keep_prob=keep_prob)
#cl2 : [-1, width / 4, height / 4, 64]

cl3_bn = tf.layers.batch_normalization(inputs=cl2, axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, training=is_training)
cl3 = tf.nn.conv2d(cl3_bn, filters['cf3'], strides=[1, 1, 1, 1],  padding='SAME')
cl3 = tf.nn.max_pool(cl3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#cl3 : [-1, width / 8, height / 8, 128]

cl4_bn = tf.layers.batch_normalization(inputs=cl3, axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, training=is_training)
cl4 = tf.nn.conv2d(cl4_bn, filters['cf4'], strides=[1, 1, 1, 1],  padding='SAME')
cl4 = tf.nn.relu(cl4)
cl4 = tf.nn.max_pool(cl4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
cl4 = tf.nn.dropout(cl4, keep_prob=keep_prob)
#cl4 : [-1, width / 16, height / 16, 256]

flat_cl4 = tf.reshape(cl4, [-1, int((width / 16) * (height / 16) * 256)])

fc1_w = tf.get_variable("fc1_w", shape=[(width / 16) * (height / 16) * 256, 256], initializer=tf.contrib.layers.xavier_initializer())
fc1_w = tf.layers.batch_normalization(inputs=fc1_w, axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, training=is_training)
b1 = tf.Variable(tf.random_normal([256]))
fc1 = tf.nn.relu(tf.matmul(flat_cl4, fc1_w) + b1)
fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

fc2_w = tf.get_variable("fc2_w", shape=[256, 3], initializer=tf.contrib.layers.xavier_initializer())
fc2_w = tf.layers.batch_normalization(inputs=fc2_w, axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, training=is_training)
b2 = tf.Variable(tf.random_normal([3]))
classify_layer = tf.matmul(fc1, fc2_w) + b2

classify_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=classify_layer, labels=label), name='Classification_Cost')
classify_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(classify_cost)
classify_prediction = tf.equal(tf.argmax(classify_layer, 1), tf.argmax(label, 1))
classify_accuracy = tf.reduce_mean(tf.cast(classify_prediction, tf.float32), name='Classification_Accuracy')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test_source_img, test_label = get_batch_data(int(len(test_data)), 0, test_data)
    test2_img, test2_label = get_batch_data(int(len(rup_nor_test_data)), 0, rup_nor_test_data)

    for i in range(training_epochs):
        print("-------Epoch", i + 1, "Training Start!!-------")
        random.shuffle(train_data)
        eval_data = train_data[int(len(train_data)*0.9):]
        eval_source_img, eval_label = get_batch_data(int(len(eval_data)), 0, eval_data)

        training_steps = int(len(train_data) / batch_size)

        for j in range(training_steps+1):
            batch_s_img, batch_l = get_batch_data(batch_size, j, train_data)
            class_cost, _ = sess.run([classify_cost, classify_optimizer], feed_dict={input_image: batch_s_img, label: batch_l,is_training: True, keep_prob: 0.8})

        eval_c_cost, eval_c_acc = sess.run([classify_cost, classify_accuracy], feed_dict={input_image: eval_source_img, label: eval_label, is_training: False, keep_prob: 1.0})
        print("Evaluation Classification Cost : ", '{:5f}'.format(eval_c_cost), "\nEvaluation Classification Accuracy : ", '{:5f}'.format(eval_c_acc))

    print("-------------Training Finished-------------")
    test_c_cost, test_c_acc = sess.run([classify_cost, classify_accuracy],
                                       feed_dict={input_image: test_source_img, label: test_label, is_training: False, keep_prob: 1.0})
    print("-------------3 class Test results-------------")
    print("Classification Cost : ", '{:5f}'.format(test_c_cost), "\nClassification Accuracy : ",
          '{:5f}'.format(test_c_acc))

    test_2_cost, test_2_acc = sess.run([classify_cost, classify_accuracy],
                                       feed_dict={input_image: test2_img, label: test2_label, is_training: False, keep_prob: 1.0})
    print("-------------2 class Test results-------------")
    print("Classification Cost : ", '{:5f}'.format(test_2_cost), "\nClassification Accuracy : ",
          '{:5f}'.format(test_2_acc))