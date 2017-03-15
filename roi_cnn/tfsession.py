import tensorflow as tf

import atexit

class TFSession:
    class __TFSession:
        def __init__(self):
            self.sess = tf.Session()

        def __getattr__(self, name):
            return getattr(self.sess, name)

    instance = None

    def __init__(self):
        if not TFSession.instance:
            print('Creating Tensorflow Session...')
            TFSession.instance = TFSession.__TFSession()

    def __getattr__(self, name):
        return getattr(self.instance, name)

@atexit.register
def __close_session():
    print('Closing Tensorflow Session...')
    TFSession().close()
