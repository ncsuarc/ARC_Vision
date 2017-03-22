import tensorflow as tf

import atexit

class TFSession:
    class __TFSession:
        def __init__(self):
            self.sess = tf.Session()
            atexit.register(self.close)

        def __getattr__(self, name):
            return getattr(self.sess, name)

        def close(self):
            print('Closing Tensorflow Session...')
            self.sess.close()

    instance = None

    def __init__(self):
        if not TFSession.instance:
            print('Creating Tensorflow Session...')
            TFSession.instance = TFSession.__TFSession()

    def __getattr__(self, name):
        return getattr(self.instance, name)

