'''
Created on Nov 17, 2017

@author: en
'''

import tensorflow as tf
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
import os
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework.errors_impl import OutOfRangeError
import operator
from sklearn.externals import joblib

left_mean, left_std, right_mean, right_std = 0, 0, 0, 0

def _parse_function(example_proto):
    global left_mean, left_std, right_mean, right_std

    features = {"left": tf.FixedLenFeature((), tf.string, default_value=''),
                "right": tf.FixedLenFeature((), tf.string, default_value=''),
                "label": tf.FixedLenFeature((), tf.string, default_value=''),
                "index": tf.FixedLenFeature((), tf.int64, default_value=0),
                "left_name": tf.FixedLenFeature((), tf.string, default_value=''),
                "right_name": tf.FixedLenFeature((), tf.string, default_value=''),
                }
    parsed_features = tf.parse_single_example(example_proto, features)

    left = tf.decode_raw(parsed_features["left"], tf.uint8)
    right = tf.decode_raw(parsed_features["right"],  tf.uint8)
    label = tf.decode_raw(parsed_features["label"],  tf.int32)
    index = tf.cast(parsed_features['index'], tf.int64)
    
    left, right, label = tf.reshape(left, (1, 64, 64)), tf.reshape(right, (1, 64, 64)), tf.reshape(label, (2,))

    
    left, right, label = tf.cast(left, tf.float32), tf.cast(right, tf.float32), tf.cast(label, tf.float32)
    
    return (left - left_mean)/left_std, (right - right_mean)/right_std, label, index, parsed_features["left_name"], parsed_features["right_name"]

def _errorRateAt95(params):
    labels, scores, recall_point = params
    
    # Sort label-score tuples by the score in descending order.
    sorted_scores = zip(labels, scores)
    sorted_scores = sorted(sorted_scores, key=operator.itemgetter(1), reverse=True)

    # Compute error rate
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_point * n_match
    tp = 0
    count = 0
    for label, _ in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
        if tp >= n_thresh:
            break

    return float(count - tp) / count


def ErrorRateAt95Recall(labels, scores, recall_point):
    
    if len(recall_point) == 1:
        return _errorRateAt95([labels, scores, recall_point[0]])    
    
    results = [_errorRateAt95((labels, scores, e)) for e in recall_point]
    return results

class AbstractNetwork(object):
    '''
    The Abstract Network class to define all the common definitions among different architecture Matchnet, MatchNetLecun and TS-Net
    
    '''

    def __init__(self, flags):

        self.optimization = flags.optimization
        self.lr = flags.lr
        self.lr_decay_rate = flags.lr_decay_rate
        self.lr_decay_step = flags.lr_decay_step
        
        if not hasattr(self, 'nb_epochs'):
            self.nb_epochs = flags.nb_epochs
            self.batch_size = flags.batch_size
        self.train_test_phase= flags.train_test_phase
        self.name = flags.name
        self.resume_cpt = flags.resume_cpt
        
        self.loss_function= flags.loss_function

        self.dropout = flags.dropout
        self.use_batchnom = flags.batchnorm
        self.wd = flags.wd
        
        self.dataset = flags.dataset
        self.margin = flags.margin
        
        self.Bsize = flags.Bsize
        self._load_mean_std()
        self.stddev = 5e-2

        self.use_normal = flags.use_normal
        self.nb_run = flags.nb_run
        
        self.valid_result_fn = 'result_valid'
        self.test_result_fn = 'result_test'
    
    def smooth(self, x, window_len=3):
        new_x = np.zeros((len(x)), 'f4')
        new_x[:] = x[:]
        half_window = window_len//2
        for i in range(window_len//2, len(x) - window_len//2):
            new_x[i] = np.mean(new_x[i-half_window:i+half_window])
        return new_x
    
    # All the dataset are stored as uint8 for storage purposes. The normalization is done during run time.
    # This function is used to load the mean and the std of each dataset and each modality
    def _load_mean_std(self):

        if self.dataset == 'vedai':
            self.mean_1, self.std_1, self.mean_2, self.std_2 = np.load('../data/vedai/tfrecord/Vehicules512/std_mean.npy')
            self.nb_iterations = 446464 // self.batch_size
        elif self.dataset == 'cuhk':
            self.mean_1, self.std_1, self.mean_2, self.std_2 = np.load('../data/cuhk/tfrecord/std_mean.npy')
            self.nb_iterations =  113184 // self.batch_size
        elif self.dataset == 'nirscene':
            self.mean_1, self.std_1, self.mean_2, self.std_2 = np.load('../data/nirscene/tfrecord/std_mean.npy')
            self.nb_iterations =  427392 // self.batch_size
        else:
            raise NotImplementedError
        
        global left_mean, left_std, right_mean, right_std
        left_mean, left_std, right_mean, right_std = self.mean_1, self.std_1, self.mean_2, self.std_2

  

    # Attention: we prepare our data using NCHW format to speedup the CNN layer
    def _feature_tower(self, x):

        output_dim = self.Bsize
        size = self.network_size
    
        input_format = 'channels_first'
        pooling_stride = [1, 2, 2, 1] if input_format == 'channels_last' else [1, 1, 2, 2]
        pooling_ksize = [1, 3, 3, 1] if input_format == 'channels_last' else [1, 1, 3, 3]
        pooling_format = 'NHWC' if input_format == 'channels_last' else 'NCHW'
        # conv0
        
        cnn_initializer = tf.contrib.layers.xavier_initializer_conv2d()  # @UndefinedVariable
        
        if self.use_normal:
            cnn_initializer = tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32)

        conv0 = tf.layers.conv2d(x, filters=int(size*24), kernel_size=[7, 7], strides=1, padding='SAME',  # @UndefinedVariable
                                          data_format=input_format, activation=tf.nn.relu, use_bias=True, 
                                          kernel_initializer=cnn_initializer,
                                          bias_initializer=tf.constant_initializer(0.1), 
                                          kernel_regularizer=l2_regularizer(self.wd),
                                          bias_regularizer=l2_regularizer(self.wd),
                                          name='conv0')

        if self.use_batchnom:
            assert input_format == 'channels_first'
            conv0 = tf.layers.batch_normalization(conv0, axis=1, name = 'conv0_batcnorm')  # @UndefinedVariable
    
        # pool0
        pool0 = tf.nn.max_pool(conv0, ksize=pooling_ksize, strides=pooling_stride, padding='SAME', name='pool0', data_format=pooling_format)
    

        conv1 = tf.layers.conv2d(pool0, filters=int(size*64), kernel_size=[5, 5], strides=1, padding='SAME',  # @UndefinedVariable
                                          data_format=input_format, activation=tf.nn.relu, use_bias=True, 
                                          kernel_initializer=cnn_initializer,
                                          bias_initializer=tf.constant_initializer(0.1), 
                                          kernel_regularizer=l2_regularizer(self.wd),
                                          bias_regularizer=l2_regularizer(self.wd),
                                          name='conv1')
        if self.use_batchnom:
            assert input_format == 'channels_first'
            conv1 = tf.layers.batch_normalization(conv1, axis=1, name = 'conv1_batcnorm')  # @UndefinedVariable

    
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=pooling_ksize, strides=pooling_stride, padding='SAME', name='pool1', data_format=pooling_format)
        
        
        # conv2
        conv2 = tf.layers.conv2d(pool1, filters=int(size*96), kernel_size=[3, 3], strides=1, padding='SAME',  # @UndefinedVariable
                                          data_format=input_format, activation=tf.nn.relu, use_bias=True, 
                                          kernel_initializer=cnn_initializer,
                                          bias_initializer=tf.constant_initializer(0.1), 
                                          kernel_regularizer=l2_regularizer(self.wd),
                                          bias_regularizer=l2_regularizer(self.wd),
                                          name='conv2')

        if self.use_batchnom:
            assert input_format == 'channels_first'
            conv2 = tf.layers.batch_normalization(conv2, axis=1, name = 'conv2_batcnorm')      # @UndefinedVariable
    
        # conv3

        conv3 = tf.layers.conv2d(conv2, filters=int(size*96), kernel_size=[3, 3], strides=1, padding='SAME',  # @UndefinedVariable
                                          data_format=input_format, activation=tf.nn.relu, use_bias=True, 
                                          kernel_initializer=cnn_initializer,
                                          bias_initializer=tf.constant_initializer(0.1), 
                                          kernel_regularizer=l2_regularizer(self.wd),
                                          bias_regularizer=l2_regularizer(self.wd),
                                          name='conv3')
        if self.use_batchnom:
            assert input_format == 'channels_first'
            conv3 = tf.layers.batch_normalization(conv3, axis=1, name = 'conv3_batcnorm')  # @UndefinedVariable

             
        # conv4
        conv4 = tf.layers.conv2d(conv3, filters=int(size*64), kernel_size=[3, 3], strides=1, padding='SAME',  # @UndefinedVariable
                                          data_format=input_format, activation=tf.nn.relu, use_bias=True, 
                                          kernel_initializer=cnn_initializer,
                                          bias_initializer=tf.constant_initializer(0.1), 
                                          kernel_regularizer=l2_regularizer(self.wd),
                                          bias_regularizer=l2_regularizer(self.wd),
                                          name='conv4')    
        if self.use_batchnom:
            assert input_format == 'channels_first'
            conv4 = tf.layers.batch_normalization(conv4, axis=1, name = 'conv4_batcnorm')  # @UndefinedVariable
        
        # pool4
        pool4 = tf.nn.max_pool(conv4, ksize=pooling_ksize, strides=pooling_stride, padding='SAME', name='pool4', data_format='NCHW')
    
        
        flat = tf.layers.flatten(pool4)  # @UndefinedVariable
        
        B = tf.layers.dense(flat, output_dim, activation=tf.nn.relu, use_bias=True,  # @UndefinedVariable
                            kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                            bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
                            kernel_regularizer=l2_regularizer(self.wd),
                            bias_regularizer=l2_regularizer(self.wd), 
                            name='bottleneck')    

        return B
    
    
    def save_images(self, left, right, label, name):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        for i in range(min(40, left.shape[0])):
            plt.subplot(9, 9,  2*i + 1)
            plt.imshow(left[i, 0], cmap='gray')
            plt.title('%d_lb:%d'%(i, label[i, 0]))
            plt.subplot(9, 9, 2*i + 2)
            plt.imshow(right[i, 0], cmap='gray')
            plt.title('%d_lb:%d'%(i, label[i, 0]))
        plt.savefig('%s/%s'%(self.train_dir, name))
    
    # define the metric network architecture
    def _metric_nework(self, x, output_dim):
        

        fc1 = tf.layers.dense(x, 512, activation=tf.nn.relu, use_bias=True,  # @UndefinedVariable
                            kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                            bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
                            kernel_regularizer=l2_regularizer(self.wd),
                            bias_regularizer=l2_regularizer(self.wd), 
                            name='fc1')
        

        if self.dropout > 0.:
            fc1 = tf.layers.dropout(fc1, rate=self.dropout, training=self.train_test_phase == 'train')  # @UndefinedVariable
        
        fc2 = tf.layers.dense(fc1, 512, activation=tf.nn.relu, use_bias=True,  # @UndefinedVariable
                            kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                            bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
                            kernel_regularizer=l2_regularizer(self.wd),
                            bias_regularizer=l2_regularizer(self.wd), 
                            name='fc2')

        if self.dropout > 0.:
            fc2 = tf.layers.dropout(fc2, rate=self.dropout, training=self.train_test_phase == 'train')  # @UndefinedVariable

        fc3 = tf.layers.dense(fc2, output_dim, activation=tf.nn.relu if output_dim== 1 else None, use_bias=True,  # @UndefinedVariable
                            kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                            bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
                            kernel_regularizer=l2_regularizer(self.wd),
                            bias_regularizer=l2_regularizer(self.wd), 
                            name='fc3')
        

        return fc3
    
    
    def _cross_entropy_loss(self, logits, labels, factor = 1, name = 'Entropy_loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='%s_per_example'%name)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('Train/%s'%name, cross_entropy_mean)  # @UndefinedVariable
        tf.add_to_collection('losses', factor * cross_entropy_mean)
        self.entropy_loss = cross_entropy_mean
        return self.entropy_loss

    # For more detail on this loss, please refer to 
    # Sumit Chopra,  Raia Hadsell, Yann LeCun, "Learning a Similarity Metric Discriminatively, with Application to Face Verification"
    def _lecun_loss(self, logits, name = 'contrastive_loss'):
        # y == 0 then x1 and x2 is a positive pair
        # margin = 1
        labels = self.label[:, 1]
    
        mismatch_loss = tf.multiply(labels *2 *self.margin , tf.exp(-2.77 *logits/self.margin)) # @UndefinedVariable
        match_loss = tf.multiply(2 * (1-labels), tf.square(logits)/ self.margin)  # @UndefinedVariable
    
        # if label is 1, only match_loss will count, otherwise mismatch_loss
        loss = tf.add(mismatch_loss, match_loss)
        loss_mean = tf.reduce_mean(loss, name = name)
 
        if not hasattr(self, 'balance_var'):
            self.balance_var = tf.minimum(self.balance + self.balance_step * tf.cast(self.global_step, tf.float32), self.balance_max)
            tf.summary.scalar('balances_between_two_losses', self.balance_var)  # @UndefinedVariable
        
        tf.summary.scalar('Train/%s'%name, loss_mean)  # @UndefinedVariable
        tf.add_to_collection('losses', loss_mean * self.balance_var)
        self.lecun_loss = loss_mean


    def _lecun_loss_1(self, logits):
        # y == 0 then x1 and x2 is a positive pair
        # margin = 1
        labels = self.label[:, 1]
    
        mismatch_loss = tf.multiply(labels *2 *self.margin , tf.exp(-2.77 *logits/self.margin), name= 'miss_match_term_1') # @UndefinedVariable
        match_loss = tf.multiply(2 * (1-labels), tf.square(logits)/ self.margin, name ='match_term_1')  # @UndefinedVariable
    
        # if label is 1, only match_loss will count, otherwise mismatch_loss
        loss = tf.add(mismatch_loss, match_loss, name = 'loss_add_1')
        loss_mean = tf.reduce_mean(loss, name = 'contrastive_loss_1')
        
        self.balance_var = tf.minimum(self.balance + self.balance_step * tf.cast(self.global_step, tf.float32), self.balance_max)
        tf.summary.scalar('balances_between_two_losses_1', self.balance_var)  # @UndefinedVariable

        
        tf.summary.scalar('Train/Lecun_loss_1', loss_mean)  # @UndefinedVariable
        tf.add_to_collection('losses', loss_mean * self.balance_var)
        self.lecun_loss = loss_mean

        
    def _try_create_directory(self, path):
        try: os.mkdir(path)
        except: pass

    # This function is used to create log directory. All the important hyper-parameters are used to create a unique name        
    def _train_dir_path(self, suffix = ''):

        path = os.path.dirname(os.path.realpath(__file__))

        path += '/%s/'%self.dataset
        self._try_create_directory(path)

        path += '%s_%s/'%(self.name, self.loss_function)
        self._try_create_directory(path)

        path += '%s_%d_%s_%0.4f_%0.2f_%d_%d_%0.2f_%d_%0.4f_%d_%d%s'%(self.optimization, self.batch_size, self.loss_function, self.lr,
                                                                self.lr_decay_rate, self.lr_decay_step, self.nb_epochs, self.dropout,
                                                                self.use_batchnom, self.wd, self.margin, self.use_normal, suffix)

        self._try_create_directory(path)

        for i in range(self.nb_run):
            tmp_path = path + '/%d'%i
            if self.train_test_phase == 'train':
                if os.path.exists(tmp_path):
                    continue
                else:
                    self._try_create_directory(tmp_path)
                    return tmp_path
            else :
                # after each training, the program will generate "done.txt" to indicate this configuration is trained
                if os.path.exists(tmp_path + '/done.txt') and self.checkpoint_exists(tmp_path):
                    
                    if self.train_test_phase == 'test':
                        return tmp_path
                    
                    tmp_fn = self.valid_result_fn
                    if os.path.exists(tmp_path + '/%s.pkl'%tmp_fn):
                        print('--- \t result file exists')
                        continue
                    else:
                        # temporary create a filename, so that the next process knows this configuration is being evaluated
                        # useful only when running many processes concurrently
                        joblib.dump('1', tmp_path + '/%s.pkl'%tmp_fn, compress=3)
                        return tmp_path
        return ''

    def checkpoint_exists(self, folder):
        for e in os.listdir(folder):
            if 'model.ckpt' in e:
                return True
        return False

    # raise Error so that all the subclass has to define its definition
    def prepare_inference(self):
        raise NotImplementedError
    
    # raise Error so that all the subclass has to define its definition
    def prepare_loss(self):
        raise NotImplementedError 

    
    def get_train_op(self):
        
        lr = tf.train.exponential_decay(self.lr, self.global_step, self.lr_decay_step * self.nb_iterations, self.lr_decay_rate, staircase=True)
        tf.summary.scalar('learning_rate', lr)  # @UndefinedVariable
        
        self.total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar('Train/total_loss', self.total_loss)  # @UndefinedVariable
        
        if self.optimization == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(lr)
        elif self.optimization == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate= lr)
        elif self.optimization == 'adam':
            self.optimizer = tf.train.AdamOptimizer()
        elif self.optimization == 'sgdm':
            self.optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov = True)
        else:
            raise NotImplementedError()
        
        grads = self.optimizer.compute_gradients(self.total_loss)
        
        # Apply gradients.
        self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)

    

    def _get_train_test_set(self):
        path = os.path.dirname(os.path.realpath(__file__))

        if self.dataset == 'vedai':
            tmp_path = '%s/../data/vedai/tfrecord/Vehicules512'%(path)
            self.train_filenames = ['%s/%s'%(tmp_path, e) for e in os.listdir(tmp_path) if 'train' in e and e.endswith('tfrecord')]
            self.test_filenames = ['%s/%s'%(tmp_path, e) for e in os.listdir(tmp_path) if 'test' in e and e.endswith('tfrecord')]
            self.valid_filenames = ['%s/%s'%(tmp_path, e) for e in os.listdir(tmp_path) if 'validation' in e and e.endswith('tfrecord')]
        
        elif self.dataset == 'cuhk':
            tmp_path = '%s/../data/cuhk/tfrecord/cuhk'%(path)
            self.train_filenames = ['%s/%s'%(tmp_path, e) for e in os.listdir(tmp_path) if 'train' in e and e.endswith('tfrecord')]
            self.test_filenames = ['%s/%s'%(tmp_path, e) for e in os.listdir(tmp_path) if 'test' in e and e.endswith('tfrecord')]
            self.valid_filenames = ['%s/%s'%(tmp_path, e) for e in os.listdir(tmp_path) if 'validation' in e and e.endswith('tfrecord')]

        elif self.dataset == 'nirscene':
            tmp_path = '%s/../data/nirscene/tfrecord/'%(path)

            self.train_filenames, self.test_filenames, self.valid_filenames = [], [], []
            
            for each in os.listdir(tmp_path):
                if not os.path.isdir('%s/%s'%(tmp_path, each)): continue
                
                self.train_filenames.extend(['%s/%s/%s'%(tmp_path, each, e) for e in os.listdir('%s/%s'%(tmp_path, each)) if 'train' in e and e.endswith('tfrecord')])
                self.test_filenames.extend(['%s/%s/%s'%(tmp_path, each, e) for e in os.listdir('%s/%s'%(tmp_path, each)) if  'test' in e and e.endswith('tfrecord')])
                self.valid_filenames.extend(['%s/%s/%s'%(tmp_path, each, e) for e in os.listdir('%s/%s'%(tmp_path, each)) if  'validation' in e and e.endswith('tfrecord')])

        else:
            raise NotImplementedError

#         print(self.train_filenames, self.test_filenames, self.valid_filenames)

    def _prepare_dataset(self, fns):

        dataset = tf.data.TFRecordDataset(fns)  # @UndefinedVariable
        dataset = dataset.prefetch(100000)
        
        dataset = dataset.map(_parse_function, num_parallel_calls = 5)

        if self.train_test_phase =='train': 
            dataset = dataset.shuffle(buffer_size=20000) 
            dataset = dataset.repeat(self.nb_epochs + 6) # add some extra epochs because the nb of iternation is approximated only
        dataset = dataset.batch(self.batch_size  if self.train_test_phase == 'train' else 512)
        iterator = dataset.make_initializable_iterator()
        return iterator
                    
    def prepare_data(self, category = []):
        
        if self.train_test_phase == 'train':
            self.iterator = self._prepare_dataset(self.train_filenames)
        elif self.train_test_phase == 'test':
            self.iterator = self._prepare_dataset(self.test_filenames)
        elif self.train_test_phase == 'validation':
            self.iterator = self._prepare_dataset(self.valid_filenames)
        
        # the inputs to the network
        self.left, self.right, self.label, self.index, self.nameleft, self.nameright = self.iterator.get_next()
        self.global_step = tf.Variable(0, dtype='int32', trainable = False, name ='global_step')


    def should_stop_training(self, sess, epoch):
        
        if os.path.exists('%s/model.ckpt-%d.index'%(self.train_dir, sess.run(self.global_step))):
            self.done_training('other process is running here')
            return True
        
        _, loss_value  = sess.run((self.train_op, self.entropy_loss))
        if np.isnan(loss_value):
            self.done_training('nan in loss')
            return True
        
        if epoch >= 10 and loss_value >= 0.68:
            self.done_training('nan in weights, loss is always around 0.69')
            return True 
        
        return False
    
    
    # The train function
    # At every epoch, a checkpoint file is saved
    # we then evaluate the network using each checkpoint on the validation
    # the checkpoint that produces the best performance on validation set is retained for evaluation
    def train(self):
        
        summary_op = tf.summary.merge_all()  # @UndefinedVariable
        
        init = tf.global_variables_initializer()  # @UndefinedVariable
    
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, 
                                                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95)))
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 1000)  # @UndefinedVariable
        
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)    # @UndefinedVariable
        
        sess.run(self.iterator.initializer)

        if self.resume_cpt != '':
            assert tf.train.latest_checkpoint(self.train_dir) is None
            self.load_from_cpt(sess)

        start_epoch = 0
        if tf.train.latest_checkpoint(self.train_dir) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(self.train_dir))
            start_epoch = int(tf.train.latest_checkpoint(self.train_dir).split('/')[-1].replace('model.ckpt-', ''))
            print ('resuming training from ', start_epoch)
            sess.run(self.global_step.assign(start_epoch))
            start_epoch = int(start_epoch) // self.nb_iterations + 1            



        for _ in range(2):
            left_np, right_np, label_np = sess.run([self.left, self.right, self.label])
            self.save_images(left_np, right_np, label_np, 'train_%d.png'%sess.run(self.global_step))
        
        for epoch in range(start_epoch, self.nb_epochs):

            # log at every first iteration of a new epoch
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, sess.run(self.global_step))

            # train for half epoch
            for _ in range(self.nb_iterations//2):
                sess.run(self.train_op)

            # log again
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, sess.run(self.global_step))
                
            for _ in range(self.nb_iterations//2 -1 ):
                sess.run(self.train_op)

            # finish an epoch, start saving checkpoint
            if self.should_stop_training(sess, epoch):
                # another process is running here
                sess.close()
                ops.reset_default_graph()
                return
            saver.save(sess, '%s/model.ckpt'%self.train_dir, global_step=self.global_step)

        self.done_training()
        sess.close()
        ops.reset_default_graph()
    
    def get_all_cpt_in_order(self):
        s = [e[:-6] for e in os.listdir(self.train_dir) if 'model.ckpt-' in e and '.index' in e]
        step = [int(e.replace('model.ckpt-', '')) for e in s]
        indx = np.argsort(step)
        return [self.train_dir + '/' + s[i] for i in indx]
    
    def get_best_checkpoint(self, train_dir = '', use_smooth = True, use_last = False, recall_point=0.95):
        
        if train_dir == '':
            train_dir = self.train_dir
            
        # load default checkpoint 
        fns = [e for e in os.listdir(train_dir) if 'model.ckpt-' in e and e.endswith('.index')]
        if len(fns) == 1: 
            print('=========\n')
            print('Is going to load the default checkpoint ', fns[0][:-6])
            return self.train_dir + '/' + fns[0][:-6]
            
        results = joblib.load('%s/result_valid.pkl'%train_dir)
        
        for k, _ in results.items():
            f_path = k.split('model.ckpt-')[0] + 'model.ckpt-'
        
        results = {k.split('model.ckpt-')[1]:v[recall_point] for k, v in results.items()}    
        results = np.array([[k,v] for k,v in results.items()]).astype('f4')
        
        if use_last:
            return '%s%d'%(f_path, int(results[-1, 0]))
            
        assert len(results.shape) == 2
        indx = np.argsort(results[:, 0])
        results = results[indx, :]
        
        if use_smooth:
            index = np.argmin(self.smooth(results[:, 1]))
            return '%s%d'%(f_path, int(results[index, 0]))
        
        return  '%s%d'%(f_path, int(results[np.argmin(results[:, 1]), 0]))
            
    def evaluate_test(self, different_recall = True):
        
        init = tf.global_variables_initializer()  # @UndefinedVariable
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, 
                                                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95)))
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)  # @UndefinedVariable
        
        recall_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99]

        sess.run(self.iterator.initializer)
        saver.restore(sess, self.get_best_checkpoint())

        results_by_cpt = []    
        while True:
            try: results_by_cpt.append(sess.run([self.logits, self.label]))
            except OutOfRangeError: break
        
        scores, lbs = zip(*results_by_cpt)

        scores = np.vstack(scores)      
        
        assert scores.shape[1] == 2

        test_lb = np.vstack(lbs)
        if self.loss_function == 'entropy':
            error_logs = ErrorRateAt95Recall(test_lb[:, 0], scores[:, 0], recall_point=recall_points)

        error_logs = {recall_points[i]: error_logs[i] for i in range(len(recall_points))}

        print('\n')
        print('\n===================\n')
        print('Error rate @95 (in percentage): ', error_logs[0.95]*100)
        print()
        joblib.dump(error_logs, '%s/%s_%s.pkl'%(self.train_dir, self.test_result_fn, self.get_best_checkpoint().split('model.ckpt-')[1]), compress=3)
        sess.close()
        ops.reset_default_graph()


    def evaluate_validate(self):
        
        init = tf.global_variables_initializer()  # @UndefinedVariable
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95)))
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)  # @UndefinedVariable
        
        recall_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99]
        results = {}
        loss_value = []
        for each_cpt in self.get_all_cpt_in_order():
            sess.run(self.iterator.initializer)
            saver.restore(sess, each_cpt)

            results_by_cpt = []    
            while True:
                try: results_by_cpt.append(sess.run([self.entropy_loss, self.logits, self.label]))
                except OutOfRangeError: break
            
            losses, scores, lbs = zip(*results_by_cpt)
    
            loss_value.append(np.mean(losses))
    
            try: scores = np.vstack(scores)
            except: scores = np.hstack(scores)
            
            test_lb = np.vstack(lbs)
            if self.loss_function == 'entropy':
                error_logs = ErrorRateAt95Recall(test_lb[:, 0], scores[:, 0], recall_point=recall_points)
    
            error_logs = {recall_points[i]: error_logs[i] for i in range(len(recall_points))}
            results[each_cpt] = error_logs
        
        joblib.dump(results, '%s/%s.pkl'%(self.train_dir, self.valid_result_fn), compress=3)
        joblib.dump(loss_value, '%s/losses.pkl'%(self.train_dir), compress=3)
        sess.close()
        ops.reset_default_graph()

       
    def done_training(self, message = 'finished'):
        with open('%s/done.txt'%self.train_dir, 'w') as f:
            f.write('done\n')
            f.write(message)
            
    def process(self):
        
        if self.train_dir == '':
            return
        print('\n')
        print('train_dir: %s'%self.train_dir)
        
        if self.train_test_phase == 'train':
            self.train()

        elif self.train_test_phase == 'test':
            self.evaluate_test()

        elif self.train_test_phase == 'validation':
            self.evaluate_validate()

        ops.reset_default_graph()