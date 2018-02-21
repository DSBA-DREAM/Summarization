import numpy as np
import tensorflow as tf

data = np.reshape(np.load("C:/Users/Mo/Desktop/cal_data.npy"),(32,21*119,100))
label = np.load("C:/Users/Mo/Desktop/label_data.npy")

batch_size = 2
hidden_size = 200
learning_rate = 0.001
word_count = data.shape[1]
sen_length = label.shape[1]
word_dimension = data.shape[2]

x = tf.placeholder(dtype=tf.float32, shape=[None, word_count, word_dimension], name="input_x")
y = tf.placeholder(dtype=tf.float32, shape=[None, sen_length, 2], name="label")

word_level_forward_gru = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                                                activation=tf.tanh,
                                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                bias_initializer=tf.contrib.layers.variance_scaling_initializer())
word_level_forward_hidden_state = tf.zeros([batch_size, hidden_size])
word_level_backward_gru = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                                                 activation=tf.tanh,
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                 bias_initializer=tf.contrib.layers.variance_scaling_initializer())
word_level_backward_hidden_state = tf.zeros([batch_size, hidden_size])
output, hidden_output = tf.nn.bidirectional_dynamic_rnn(cell_fw=word_level_forward_gru,
                                                        cell_bw=word_level_backward_gru,
                                                        inputs=x,
                                                        initial_state_fw=word_level_forward_hidden_state,
                                                        initial_state_bw=word_level_backward_hidden_state,
                                                        dtype=tf.float32,
                                                        scope="word_bidirectional_rnn")

rescale_output = tf.tanh(tf.add(tf.matmul(tf.concat(output,2),
                                          tf.truncated_normal([batch_size, 400, hidden_size], mean=0, stddev=1)),
                                tf.Variable(tf.constant(0.1, shape=[batch_size,2499,hidden_size]))))
sen_input = tf.reduce_mean(tf.reshape(rescale_output,[batch_size,119,21,200]),axis=1)

sentence_level_forward_gru = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                                                    activation=tf.tanh,
                                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                    bias_initializer=tf.contrib.layers.variance_scaling_initializer())
sentence_level_forward_hidden_state = tf.zeros([batch_size, hidden_size])
sentence_level_backward_gru = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                                                     activation=tf.tanh,
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                     bias_initializer=tf.contrib.layers.variance_scaling_initializer())
sentence_level_backward_hidden_state = tf.zeros([batch_size, hidden_size])
sen_output, sen_hidden_output = tf.nn.bidirectional_dynamic_rnn(cell_fw=sentence_level_forward_gru,
                                                        cell_bw=sentence_level_backward_gru,
                                                        inputs=sen_input,
                                                        initial_state_fw=sentence_level_forward_hidden_state,
                                                        initial_state_bw=sentence_level_backward_hidden_state,
                                                        dtype=tf.float32,
                                                        scope="sen_bidirectional_rnn")

pred = tf.tanh(tf.add(tf.matmul(tf.concat(sen_output,2),
                                          tf.truncated_normal([batch_size, 400, 2], mean=0, stddev=1)),
                                tf.Variable(tf.constant(0.1, shape=[batch_size,21,2]))))

loss = tf.losses.softmax_cross_entropy(y,pred)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def feed_dict(bol):
    if bol == True:
        batch = np.random.choice(data.shape[0], batch_size, False)
        xs, ys = data[batch], label[batch]
        return {x: xs, y: ys}

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(50):
    _, los = sess.run([train_step,loss], feed_dict(True))
    print(los)