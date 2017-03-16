import tensorflow as tf

def create_network(save_location, n_classes,
        feature_layers = [32, 64, -1, 128, 256, -1],
        n_neurons = 2048,
        img_height = 60,
        img_width = 60,
        color_channels = 3,
        learning_rate = 0.001,
        dropout = .75):

    sess = tf.Session()

    x = tf.placeholder(tf.float32, [None, img_height * img_width * color_channels])
    tf.add_to_collection("x", x)

    y = tf.placeholder(tf.uint8, [None])
    tf.add_to_collection("y", y)

    y_one_hot = tf.one_hot(y, n_classes)
    
    predictor = conv_net(x, img_height, img_width, color_channels, n_neurons, dropout, n_classes, feature_layers)
    tf.add_to_collection("predictor", predictor)

    softmax = tf.nn.softmax(predictor)
    tf.add_to_collection("softmax", softmax)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictor, labels=y_one_hot))
    tf.add_to_collection("cost", cost)
    
    global_step = tf.Variable(0, name='global_step', trainable=False) 
    tf.add_to_collection("step", global_step)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
    tf.add_to_collection("optimizer", optimizer)

    correct_pred = tf.equal(tf.argmax(predictor, 1), tf.argmax(y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.add_to_collection("accuracy", accuracy)
        
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    save_path = saver.save(sess, save_location, global_step=global_step)
    print("Saved initial model in file: %s" % save_path)

def conv_net(x, img_height, img_width, color_channels, n_neurons, dropout, n_classes, feature_layers):
    #Reshape input image
    x_image = tf.reshape(x, shape=[-1, img_height, img_width, color_channels])

    prev_n = color_channels
    layer = x_image
    for n in feature_layers:
        if n < 0:
            layer = max_pool2d(layer)
            print('Pooling from %d x %d to %d x %d' % (img_height, img_width, img_height / 2, img_width / 2))
            img_height = img_height / 2
            img_width = img_width / 2
        else:
            layer = conv_layer(layer, prev_n, n)
            print('Convolving %d features to %d features.' % (prev_n, n))
            prev_n = n

    # Fully connected layer       height * width * features of last layer
    fc1_weight = weight_variable([int(img_height* img_width * prev_n), n_neurons])
    fc1_bias   = weight_variable([n_neurons])

    # Reshape pooling output to fit fully connected layer input
    out_layer_flat = tf.reshape(layer, [-1, fc1_weight.get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.add(tf.matmul(out_layer_flat, fc1_weight), fc1_bias))
    # Apply Dropout
    keep_prob = tf.placeholder(tf.float32) 
    tf.add_to_collection("kp", keep_prob)

    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    fc2_weight = weight_variable([n_neurons, n_classes])
    fc2_bias   = weight_variable([n_classes])

    # Output, class prediction
    out = tf.add(tf.matmul(fc1_drop, fc2_weight), fc2_bias)
    return out

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

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create a model')
    parser.add_argument("-s", "--save-location", required=True, help="Directory to save training in.")
    parser.add_argument("-f", "--features", nargs='+', type=int, required=True, help="List of feature layer sizes. A layer size < 0 indicated a pooling layer. \n\tFor Example: -f 32 64 -1 128 256 -1")
    parser.add_argument("-c", "--classes", required=True, type=int, help="The number of classes.")
    args = parser.parse_args()
    print(vars(args))
    create_network(args.save_location + "/model", args.classes, feature_layers = args.features)
