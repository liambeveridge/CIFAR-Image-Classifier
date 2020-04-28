import numpy as np
import tensorflow as tf
import itertools

def unpickle(file):
    """Adapted from the CIFAR page: http://www.cs.utoronto.ca/~kriz/cifar.html"""
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

# Gather data
data_dir = '/home/users/drake/data/cifar-10-batches-py/' # BLT
train = [unpickle(data_dir + 'data_batch_{}'.format(i)) for i in [1, 2, 3, 4]]
y_train = np.array(list(itertools.chain(*[t[b'labels'] for t in train])))
X_train = np.concatenate([t[b'data'] for t in train], axis=0)
valid = unpickle(data_dir + 'data_batch_5')
X_valid = valid[b'data']
y_valid = np.array(valid[b'labels'])
test = unpickle(data_dir + 'test_batch')
X_test = test[b'data']
y_test = np.array(test[b'labels'])


# Build network
n_inputs = 32 * 32 * 3
n_outputs = 10
learning_rate = 0.001

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    shaped = tf.transpose(tf.reshape(X, [-1, 3, 32, 32]), (0, 2, 3, 1))
    n_filters1 = 32
    n_filters2 = 64
    n_filters3 = 128
    conv1 = tf.layers.conv2d(shaped, n_filters1, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='valid')
    conv2 = tf.layers.conv2d(pool1, n_filters2, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='valid')
    conv3 = tf.layers.conv2d(pool2, n_filters3, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2, padding='valid')
    flat = tf.reshape(pool3, [-1, 4 * 4 * n_filters3])
    n_hidden1 = 1024
    hidden1 = tf.layers.dense(flat, n_hidden1, name="hidden1", activation=tf.nn.elu)
    hidden2 = tf.layers.dense(hidden1, n_hidden1, name="hidden2", activation=tf.nn.elu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 400
batch_size = 40000

#Run Network printing training and validation errors each epoch, then print testing accuracy.
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(1):
            X_batch = X_train
            y_batch = y_train
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Train accuracy:", acc_train, "Validation accuracy:", acc_valid)
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("\nTesting accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")
