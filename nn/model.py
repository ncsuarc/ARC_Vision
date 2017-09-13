import tensorflow as tf
import os
import traceback

from .tfsession import TFSession

class Model:
    """
    A wrapper for loading and training Tensorflow CNN models.

    This class loads data from a directory (populated by the create_model.py script).
    See train.py for an example of training the model.
    See test.py for an example of using the trained model to classify images.

    Arguments:
        model_path: The path to load the model from.
            The directory should have the following files:
                checkpoint
                classes.txt
                model-*.data-*
                model-*.index
                model-*.meta
        batch_size: Optional argument specifying how many images to process
            in each training/testing batch.
    """

    model_n = 0

    def __init__(self, model_path, batch_size = 100):
        self.model_path = model_path
        self.batch_size = batch_size
        self.sess = TFSession().sess

        self.n = Model.model_n
        Model.model_n += 1

        with open(os.path.join(self.model_path, 'classes.txt'), 'r') as class_file:
            self.classes = class_file.read().split('\n')

        try:
            path = self.model_path
            ckpt = tf.train.get_checkpoint_state(path)
            print("Reading saved model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        except Exception as e:
            raise ValueError("Error loading model: %s" % traceback.format_exc())

        self.input = tf.get_collection("image")[self.n]
        self.keep_prob = tf.get_collection("kp")[self.n]
        self.predictor = tf.get_collection("predictor")[2 * self.n]
        self.softmax = tf.get_collection("predictor")[2 * self.n + 1]
        self.global_step = tf.get_collection("step")[self.n]

    def train(self, images, labels, learning_rate = 0.001, dropout = 0.75, display_step = 5):
        """
        Trains the model on the given set of images.

        Arguments:
            images: The list of flattened images.
            labels: The list of labels
            learning_rate: Optional parameter for the learning rate for the AdamOptimizer (This only works if this is the first time training the model)
            dropout: Optional parameter for the dropout rate to use while training
            display_step: Optional parameter for the number of batches to process between printing accuracy
        """
        training_ops = tf.get_collection("training_ops")
        if len(training_ops) == 0:
            #If no training operations have been previously defined,
            #create new training ops.
            labels_placeholder = tf.placeholder(tf.uint8, [None])
            labels_one_hot = tf.one_hot(labels_placeholder, self.predictor.get_shape()[0])
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predictor, labels=labels_one_hot))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=self.global_step)
            correct_pred = tf.equal(tf.argmax(self.predictor, 1), tf.argmax(labels_one_hot, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            tf.add_to_collection("training_ops", labels_placeholder)
            tf.add_to_collection("training_ops", labels_one_hot)
            tf.add_to_collection("training_ops", cost)
            tf.add_to_collection("training_ops", optimizer)
            tf.add_to_collection("training_ops", correct_pred)
            tf.add_to_collection("training_ops", accuracy)
        else:
            print('Loading previous training operations.')
            labels_placeholder  = training_ops[6 * self.n]
            labels_one_hot      = training_ops[6 * self.n + 1]
            cost                = training_ops[6 * self.n + 2]
            optimizer           = training_ops[6 * self.n + 3]
            correct_pred        = training_ops[6 * self.n + 4]
            accuracy            = training_ops[6 * self.n + 5]

        #Initialize only training variables, leaving model variables alone
        self.sess.run(tf.variables_initializer([v for v in tf.global_variables() if 'beta' in v.name or 'Adam' in v.name]))

        step = 1
        while step * self.batch_size < len(images):
            batch_x = images[(step-1)*self.batch_size:step*self.batch_size]
            batch_y = labels[(step-1)*self.batch_size:step*self.batch_size]
            # Run optimization
            self.sess.run(optimizer, feed_dict={self.input: batch_x, labels_placeholder: batch_y, self.keep_prob: dropout})
            if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc= self.sess.run([cost, accuracy], feed_dict={self.input: batch_x, labels_placeholder: batch_y, self.keep_prob: 1.})

                    # Save the variables to disk.
                    save_path = self.saver.save(self.sess, self.model_path + '/model', global_step=self.global_step)

                    print("Checkpoint saved in file: %s" % save_path)
                    print("Iter " + str(step*self.batch_size) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
            step += 1
        save_path = self.saver.save(self.sess, self.model_path + '/model', global_step=self.global_step)
        print("Final checkpoint saved in file: %s" % save_path)

    def test(self, images):
        """
        Runs the CCN predictor on a list of images.

        Arguments:
            images: The list of flattened images.

        Returns:
            The list of labels predicted by the CNN.
        """
        labels = []
        step = 0
        while step * self.batch_size < len(images):
            batch_x = images[step*self.batch_size:(step+1)*self.batch_size]
            labels.extend(self.sess.run(self.softmax, feed_dict={self.input: batch_x, self.keep_prob: 1.}))
            step += 1
        return labels

    def classify(self, images):
        """
        Runs the CNN predictor on a list of images.

        Arguments:
            images: The list of flattened images.

        Returns:
            A list of tuples of the form (class_name, confidence) where
                class_name is the string representing the name of the class
                confidence is the output of the softmax for the given class
            The list is sorted in order of confidence, so the predicted class is at index 0
        """
        labels = self.test(images)
        named_labels = []
        for label in labels:
            named_label = [(value, self.classes[i]) for value, i in zip(label, range(len(label)))]
            sorted_label = sorted(named_label, key=lambda x: x[0], reverse=True)
            named_labels.append(sorted_label)

        return named_labels
