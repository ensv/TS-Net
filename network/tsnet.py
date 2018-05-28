'''
Created on Jan 20, 2018

@author: en
'''


import tensorflow as tf
import argparse, sys, os
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
print (os.path.dirname(os.path.realpath(__file__)) + '/../')
from network.abstract import AbstractNetwork

class TSNet(AbstractNetwork):
    '''
    classdocs
    '''
    def __init__(self, flags):
        '''
        Constructor
        '''

        flags.name = 'TSNet'
        if flags.balance_max == 0: 
            flags.name = 'TSNetPlain'

        if flags.network_size > 1:
            flags.name += '%f'%flags.network_size
        
        self.experiment = flags.experiment
        self.balance = flags.balance
        self.balance_step = flags.balance_step
        self.balance_max = flags.balance_max
        self.network_size =flags.network_size

        self.flags = flags
        AbstractNetwork.__init__(self, flags)
        self.train_dir = self._train_dir_path(suffix='_%0.4f_%0.4f'%(self.balance_step * 100, self.balance_max))
        if self.train_dir == '': return
        self._get_train_test_set()
        self.prepare_inference()
        self.prepare_loss()
        self.get_train_op()
        
    def _lecun_loss(self, logits, name = 'contrastive_loss'):
        # y == 0 then x1 and x2 is a positive pair
        # margin = 1
        labels = self.label[:, 1]
     
        mismatch_loss = tf.multiply(labels *2 *self.margin , tf.exp(-2.77 *logits/self.margin)) # @UndefinedVariable
        match_loss = tf.multiply(2 * (1-labels), tf.square(logits)/ self.margin)  # @UndefinedVariable
     
        # if label is 1, only match_loss will count, otherwise mismatch_loss
        loss = tf.add(mismatch_loss, match_loss)
        loss_mean = tf.reduce_mean(loss, name = name)
  
        current_subnetwork_name = name.split('_')[0]
  
        balance_var = tf.minimum(self.balance + self.balance_step * tf.cast(self.global_step, tf.float32), self.balance_max)

        # for pseudo network, the best balance_max is 1e-4, not 1e-2 
        if current_subnetwork_name == 'pseudo' and self.dataset == 'vedai':
            balance_var = tf.minimum(self.balance + self.balance_step * tf.cast(self.global_step, tf.float32), self.balance_max/100)

        tf.summary.scalar('balances_between_two_losses_%s'%current_subnetwork_name, balance_var)  # @UndefinedVariable
         
        tf.summary.scalar('Train/%s'%name, loss_mean)  # @UndefinedVariable
        tf.add_to_collection('losses', loss_mean * balance_var)
        self.lecun_loss = loss_mean
        

    def make_subnetwork(self, is_siamese):

        if is_siamese:
            with tf.variable_scope("siamese") as scope:
                f1 = self._feature_tower(self.left)
                scope.reuse_variables()
                f2 = self._feature_tower(self.right)
        else:
            with tf.variable_scope("pseudo") as scope:  # @UnusedVariable
                with tf.variable_scope("siamese_left") as scope:  # @UnusedVariable
                    f1 = self._feature_tower(self.left)
    
                with tf.variable_scope("siamese_right") as scope:  # @UnusedVariable
                    f2= self._feature_tower(self.right)

        siamese_ft = tf.subtract(f1, f2)  # @UnusedVariable @UndefinedVariable

        if self.balance_max > 0 :
            distance = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(siamese_ft), axis=1)))  # @UndefinedVariable
            self._lecun_loss(distance, 'siamese_con'if is_siamese else 'pseudo_con')

        if is_siamese:
            with tf.variable_scope("siamese_metric") as scope:
                logits = self._metric_nework(siamese_ft, output_dim=2)
            self._cross_entropy_loss(logits, self.label, name = 'siamese_entropy')

        else:
            with tf.variable_scope("pseudo_metric") as scope:
                logits = self._metric_nework(siamese_ft, output_dim=2)
            self._cross_entropy_loss(logits, self.label, name = 'pseudo_entropy')

        return logits
        
    def prepare_inference(self):
        
        self.prepare_data()

        siamese = self.make_subnetwork(is_siamese = True)
        pseudo = self.make_subnetwork(is_siamese = False)
            
        fused_fts = tf.concat((siamese, pseudo), axis=1, name='fused_features')
        
        with tf.variable_scope("fuse_layer") as scope:  # @UnusedVariable
            self.logits = tf.layers.dense(fused_fts, 2, None, use_bias=True,  # @UndefinedVariable
                            kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                            bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
                            kernel_regularizer=l2_regularizer(self.wd),
                            bias_regularizer=l2_regularizer(self.wd), 
                            name='fused_layer')

      
    def prepare_loss(self):
        self._cross_entropy_loss(self.logits, self.label)
        accuracy = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        tf.summary.scalar('Train/accuracy', self.accuracy)  # @UndefinedVariable

        

def train_multimodal():
    FLAGS.nb_run = 1
    for dataset in['vedai', 'cuhk', 'nirscene']:
        FLAGS.dataset = dataset
        FLAGS.train_test_phase = 'test' if FLAGS.train_test_phase == 'test' else 'train'
        net = TSNet(FLAGS)
        net.process()
        if FLAGS.train_test_phase == 'train':
            FLAGS.train_test_phase= 'validation'
            net = TSNet(FLAGS)
            net.process()     

def train_multimodal_additional_loss():
    FLAGS.nb_run = 1
    for dataset in['vedai', 'cuhk', 'nirscene']:
        FLAGS.dataset = dataset
        # define the best parameters for the additional loss (contrastive loss)
        FLAGS.balance = 1e-6
        FLAGS.balance_step = 1e-6
        FLAGS.balance_max = 1e-2
        # end
            
        FLAGS.train_test_phase = 'test' if FLAGS.train_test_phase == 'test' else 'train'
        net = TSNet(FLAGS)
        net.process()
        if FLAGS.train_test_phase == 'train':
            FLAGS.train_test_phase= 'validation'
            net = TSNet(FLAGS)
            net.process()     

                                               
def main(_):
    
    working_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
    if FLAGS.experiment == 'multimodal':
        train_multimodal()
    elif FLAGS.experiment == 'multimodalAdditionalLoss':
        train_multimodal_additional_loss()
    elif FLAGS.experiment == '':
        net = TSNet(FLAGS)
        net.process()
        FLAGS.train_test_phase= 'validation'
        net = TSNet(FLAGS)
        net.process()
    else:
        print ('unknown experimentation')
        raise NotImplementedError
    os.chdir(working_dir)


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    
    parser.register("type", "bool", lambda v: v.lower() == "true")
    
    # common arguments for abstractNetwork
    parser.add_argument("--optimization", default= 'sgdm', type=str, help= "optimization method to use: adadelta, sgd, sgdm")
    parser.add_argument("--lr", type = float, default = 0.001, help = "Initial learning rate")
    parser.add_argument("--lr_decay_rate", type = float, default = 0.95, help = 'decay rate for learning rate')
    parser.add_argument("--lr_decay_step", type = int, default = 1000, help = 'step to decay learning rate, use big number to avoid decaying the lr')
    parser.add_argument("--nb_epochs", type = int, default = 151, help = "number of epochs to train the model")
    parser.add_argument("--batch_size", type = int, default = 32, help="number of example per batch")

    parser.add_argument("--train_test_phase", default = 'train', help = 'train, validation or test')
    parser.add_argument("--loss_function", type = str, default='entropy', help = 'choose loss function to optimize: entropy = binary_cross_entropy')

    parser.add_argument("--dropout", default = 0., type = float, help = "drop probability")
    parser.add_argument("--batchnorm", default = False, help = "use batchnorm or not (default)", action = 'store_true')
    parser.add_argument("--wd", type = float, default = 1e-3, help="weight decay applied to all the learn-able parameters")

    parser.add_argument("--dataset", default = 'vedai', type=str, help ='dataset name to train the network: vedai, cuhk, nirscene')
    parser.add_argument("--Bsize", default = 128, type = int, help = 'the output dimension of the bottleneck layer, 128 for all the experiments')

    parser.add_argument("--use_normal", default = False, help ='use xavier initializor (default) in cnn layer or truncated normal distribution', action = 'store_true')
    parser.add_argument("--nb_run", default = 2, help ='number of training per configuration')
    parser.add_argument("--resume_cpt", default = '', type=str, help ='checkpoint to load, loading is based on name of variable')
    parser.add_argument("--concate", default= False, help='indicates whether the concatenation should be used to fuse the information from the two modalities, default = False = subtraction', action = 'store_true')
    parser.add_argument("--network_size", default= 1, type=int, help='a factor used to increase/decrease the feature extraction network')

    # specific to this architecture
    parser.add_argument("--experiment", default= '', help='multimodal')
    
    # specific argument for training matchnet with contrastive loss
    parser.add_argument("--balance", type = float, default = 1e-6, help="balance between two weights, 0 mean do not use contrastive loss")
    parser.add_argument("--balance_step", type = float, default = 1e-6, help="balance between two weights")
    parser.add_argument("--balance_max", type = float, default = 0., help = 'maximum allowed balance between two weights')
    parser.add_argument("--margin", type= float, default = 50, help = 'the upper bound to train contrastive loss, the lower bound is 0')

    FLAGS, unparsed = parser.parse_known_args()

    
    tf.app.run(main= main, argv=[sys.argv[0]] + unparsed)
    