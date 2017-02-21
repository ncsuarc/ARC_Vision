import tensorflow as tf
import os
import traceback

class Model:
    def __init__(self,
                sess,
                n_features1 = 32,
                n_features2 = 64,
                n_neurons = 2048,
                learning_rate = 0.001,
                dropout = .75,
                batch_size = 100,
                display_step = 5,
                img_height = 60,
                img_width = 60,
                color_channels = 3,
                n_classes = 13,
                load=True):
        self.n_features1 = n_features1
        self.n_features2 = n_features2
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.display_step = display_step
        self.img_height = img_height
        self.img_width = img_width
        self.color_channels = color_channels

        self.n_input = self.img_height * self.img_width * self.color_channels
        self.n_classes = n_classes
        
        if load:
            try:
                path = os.path.dirname(os.path.realpath(__file__)) + '/training'
                ckpt = tf.train.get_checkpoint_state(path)
                print("Reading saved model parameters from %s" % ckpt.model_checkpoint_path)
                self.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            except Exception as e:
                print(traceback.format_exc())
                self.create_network(sess)
            self.x = tf.get_collection("x")[0]
            self.y = tf.get_collection("y")[0]
            self.keep_prob = tf.get_collection("kp")[0]
            self.predictor = tf.get_collection("predictor")[0]
            self.softmax = tf.get_collection("softmax")[0]
            self.cost = tf.get_collection("cost")[0]
            self.global_step = tf.get_collection("step")[0]
            self.optimizer = tf.get_collection("optimizer")[0]
            self.accuracy = tf.get_collection("accuracy")[0]
        else:
            self.create_network(sess)

    def create_network(self, sess):
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        tf.add_to_collection("x", self.x)

        self.y = tf.placeholder(tf.uint8, [None])
        tf.add_to_collection("y", self.y)

        self.y_one_hot = tf.one_hot(self.y, self.n_classes)
        
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        tf.add_to_collection("kp", self.keep_prob)

        self.predictor = self.conv_net()
        tf.add_to_collection("predictor", self.predictor)

        self.softmax = tf.nn.softmax(self.predictor)
        tf.add_to_collection("softmax", self.softmax)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predictor, labels=self.y_one_hot))
        tf.add_to_collection("cost", self.cost)
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False) 
        tf.add_to_collection("step", self.global_step)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)
        tf.add_to_collection("optimizer", self.optimizer)

        self.correct_pred = tf.equal(tf.argmax(self.predictor, 1), tf.argmax(self.y_one_hot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        tf.add_to_collection("accuracy", self.accuracy)
            
        self.saver = tf.train.Saver()
        
        print("Creating a new model.")
        sess.run(tf.global_variables_initializer())
    
    def conv_net(self):
        #Reshape input image
        x_image = tf.reshape(self.x, shape=[-1, self.img_height, self.img_width, self.color_channels])

        #2 convolutional layers, 1 pooling layer
        h_conv1 = conv_layer(x_image, self.color_channels, self.n_features1)
        h_conv2 = conv_layer(h_conv1, self.n_features1, self.n_features2)

        h_pool1 = max_pool2d(h_conv2)

        #2 convolutional layers, 1 pooling layer
        h_conv3 = conv_layer(h_pool1, self.n_features2, self.n_features2)
        h_conv4 = conv_layer(h_conv3, self.n_features2, self.n_features2)

        h_pool2 = max_pool2d(h_conv4)

        # Fully connected layer       (height/4) * (width/4) * n_features2
        fc1_weight = weight_variable([int(self.img_height* self.img_width *self.n_features2/16), self.n_neurons])
        fc1_bias   = weight_variable([self.n_neurons])

        # Reshape pooling output to fit fully connected layer input
        h_pool3_flat = tf.reshape(h_pool2, [-1, fc1_weight.get_shape().as_list()[0]])
        h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, fc1_weight), fc1_bias))
        # Apply Dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout)

        fc2_weight = weight_variable([self.n_neurons, self.n_classes])
        fc2_bias   = weight_variable([self.n_classes])

        # Output, class prediction
        out = tf.add(tf.matmul(h_fc1_drop, fc2_weight), fc2_bias)
        return out

    def train(self, sess, images, labels):
        step = 0
        while step * self.batch_size < len(images):
            batch_x = images[step*self.batch_size:(step+1)*self.batch_size]
            batch_y = labels[step*self.batch_size:(step+1)*self.batch_size]
            # Run optimization
            sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.dropout})
            if step % self.display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc= sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.})

                    # Save the variables to disk.
                    save_path = self.saver.save(sess, "training/model", global_step=self.global_step)

                    print("Checkpoint saved in file: %s" % save_path)
                    print("Iter " + str(step*self.batch_size) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
            step += 1
        save_path = self.saver.save(sess, "training/model", global_step=self.global_step)
        print("Final checkpoint saved in file: %s" % save_path)

    def test(self, sess, images):
        return sess.run(self.predictor, feed_dict={self.x: images, self.keep_prob: 1.})

#Wrappers
def conv2d(x, W, b):
    #Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def max_pool2d(x):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_layer(input, input_size, output_size):
    conv_weight = weight_variable([5, 5, input_size, output_size])
    conv_bias = bias_variable([output_size])

    h_conv = conv2d(input, conv_weight, conv_bias)
    return h_conv
