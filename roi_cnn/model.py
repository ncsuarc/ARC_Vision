import tensorflow as tf

import sys
import os
import traceback

sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) )

from roi_cnn.tfsession import TFSession

class Model:
    def __init__(self, model_path, batch_size = 100, display_step = 5):
        self.model_path = model_path
        self.batch_size = batch_size
        self.display_step = display_step
        self.sess = TFSession()

        try:
            path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.model_path
            ckpt = tf.train.get_checkpoint_state(path)
            print("Reading saved model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        except Exception as e:
            raise ValueError("Error loading model: %s" % traceback.format_exc())

        self.x = tf.get_collection("x")[0]
        self.y = tf.get_collection("y")[0]
        self.keep_prob = tf.get_collection("kp")[0]
        self.predictor = tf.get_collection("predictor")[0]
        self.softmax = tf.get_collection("softmax")[0]
        self.cost = tf.get_collection("cost")[0]
        self.global_step = tf.get_collection("step")[0]
        self.optimizer = tf.get_collection("optimizer")[0]
        self.accuracy = tf.get_collection("accuracy")[0]
    
    def train(self, images, labels, dropout = 0.75):
        step = 0
        while step * self.batch_size < len(images):
            batch_x = images[step*self.batch_size:(step+1)*self.batch_size]
            batch_y = labels[step*self.batch_size:(step+1)*self.batch_size]
            # Run optimization
            self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: dropout})
            if step % self.display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc= self.sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.})

                    # Save the variables to disk.
                    save_path = self.saver.save(self.sess.sess, self.model_path + '/model', global_step=self.global_step)

                    print("Checkpoint saved in file: %s" % save_path)
                    print("Iter " + str(step*self.batch_size) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
            step += 1
        save_path = self.saver.save(self.sess.sess, self.model_path + '/model', global_step=self.global_step)
        print("Final checkpoint saved in file: %s" % save_path)

    def test(self, images):
        labels = []
        step = 0
        while step * self.batch_size < len(images):
            batch_x = images[step*self.batch_size:(step+1)*self.batch_size]
            batch_y = labels[step*self.batch_size:(step+1)*self.batch_size]
            labels.extend(self.sess.run(self.predictor, feed_dict={self.x: images, self.keep_prob: 1.}))
            step += 1
        return labels 

