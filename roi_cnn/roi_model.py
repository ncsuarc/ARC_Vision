import tensorflow as tf

class Model:
    def __init__(self, sess, load=True):
        self.learning_rate = 0.001
        self.batch_size = 100 
        self.display_step = 5 

        self.n_input = 60*60*3
        self.n_classes = 2
        self.dropout = .75
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.uint8, [None])
        self.y_one_hot = tf.one_hot(self.y, self.n_classes)
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        
        self.predictor = self.conv_net()
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predictor, self.y_one_hot))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
   
        self.correct_pred = tf.equal(tf.argmax(self.predictor, 1), tf.argmax(self.y_one_hot, 1)) 
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.saver = tf.train.Saver(tf.trainable_variables())
        if load:
            try:
                ckpt = tf.train.get_checkpoint_state('./training')
                print("Reading saved model parameters from %s" % ckpt.model_checkpoint_path)
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            except Exception as e:
                print("Creating a new model.")
                sess.run(tf.global_variables_initializer())

    def conv_net(self):
        #Reshape input image
        x_image = tf.reshape(self.x, shape=[-1, 60, 60, 3])
        
        #1 convolutional layer, 1 pooling layer
        h_conv1 = conv_layer(x_image, 3, 32)
        h_pool1 = max_pool2d(h_conv1)
        #2 convolutional layers, 1 pooling layer
        h_conv2 = conv_layer(h_pool1, 32, 64)
        h_pool2 = max_pool2d(h_conv2)
        
        # Fully connected layer
        fc1_weight = weight_variable([15*15*64, 2048])
        fc1_bias   = weight_variable([2048])
        
        # Reshape pooling output to fit fully connected layer input
        h_pool3_flat = tf.reshape(h_pool2, [-1, fc1_weight.get_shape().as_list()[0]])
        h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, fc1_weight), fc1_bias))
        # Apply Dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout)
        
        fc2_weight = weight_variable([2048, self.n_classes])
        fc2_bias   = weight_variable([self.n_classes])
        
        # Output, class prediction
        out = tf.add(tf.matmul(h_fc1_drop, fc2_weight), fc2_bias)
        return out

    def train(self, sess, images, labels):
        step = 0
        while step * self.batch_size < len(images):
            batch_x = images[step*self.batch_size:(step+1)*self.batch_size] #Take one image from every character
            batch_y = labels[step*self.batch_size:(step+1)*self.batch_size]
            # Run optimization op (backprop)
            sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.dropout})
            if step % self.display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc= sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.})
                    
                    # Save the variables to disk.
                    save_path = self.saver.save(sess, "training/model", global_step=step)
            
                    print("Checkpoint saved in file: %s" % save_path)
                    print("Iter " + str(step*self.batch_size) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
            step += 1
        save_path = self.saver.save(sess, "training/model", global_step=step)
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
    
