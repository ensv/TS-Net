'''
Created on Dec 7, 2017

@author: en
'''

import tensorflow as tf
import argparse, sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
print (os.path.dirname(os.path.realpath(__file__)) + '/../')
from network.abstract import AbstractNetwork

class MatchNet(AbstractNetwork):
    '''
    A modified version of MatchNet network: Unifying Feature and Metric Learning for Patch-Based Matching
    '''
    
    
    def __init__(self, flags):
        '''
        Constructor
        '''
        if flags.pseudo:
            flags.name = 'PseudoMatchNet'
        else:
            flags.name = 'MatchNet'
        if flags.network_size > 1:
            flags.name += '%f'%flags.network_size
        
        if flags.balance_max == 0: 
            flags.name = '%sPlain'%flags.name
        
        self.network_size = flags.network_size
        self.pseudo = flags.pseudo
        self.experiment = flags.experiment
        self.concate = flags.concate
        
        # define attribute for training matchnet with contrastive loss
        self.balance = flags.balance
        self.balance_step = flags.balance_step
        self.balance_max = flags.balance_max # this corresponds to lambda and beta in the paper
        self.margin = flags.margin
        # end
        
        AbstractNetwork.__init__(self, flags)
        self.train_dir = self._train_dir_path(suffix='_%0.4f_%0.4f_%d'%(self.balance_step * 100, self.balance_max, self.concate))
        if self.train_dir == '': return
        
        # get train, test and validation file paths to create Dataset object
        self._get_train_test_set()
        # preparing the inference
        self.prepare_inference()
        # preparing loss function
        self.prepare_loss()
        # define the training operation
        self.get_train_op()
        
    def prepare_inference(self):
        # create input pipeline: dataset object to read tfrecord files
        self.prepare_data()

        if self.pseudo:
            with tf.variable_scope("siamese_left") as scope:  # @UnusedVariable
                tower_left = self._feature_tower(self.left)
    
            with tf.variable_scope("siamese_right") as scope:  # @UnusedVariable
                tower_right = self._feature_tower(self.right)
        else:
            with tf.variable_scope("siamese") as scope:
                tower_left = self._feature_tower(self.left)
                scope.reuse_variables()
                tower_right = self._feature_tower(self.right)

        if self.concate:
            self.fts = tf.concat((tower_left, tower_right), axis= 1)
        else:
            self.fts = tf.subtract(tower_left, tower_right)  # @UndefinedVariable
            
        with tf.variable_scope("metric") as scope:
            self.logits = self._metric_nework(self.fts, output_dim=2)

        # if the \lambda == 0, train only with entropy loss == matchnet        
        if self.balance_max > 0:  
            distance = tf.sqrt(tf.nn.relu(tf.reduce_sum(tf.square(tf.subtract(tower_left, tower_right)), axis=1)))  # @UndefinedVariable
            self._lecun_loss(distance)

    
    def prepare_loss(self):
        self._cross_entropy_loss(self.logits, self.label)
        accuracy = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        tf.summary.scalar('Train/accuracy', self.accuracy)  # @UndefinedVariable
              
                
def train_multimodal():
    FLAGS.nb_run = 2
    for dataset in['vedai', 'cuhk', 'nirscene']:
        FLAGS.dataset = dataset
        FLAGS.train_test_phase = 'test' if FLAGS.train_test_phase == 'test' else 'train'
        matchnet = MatchNet(FLAGS)
        matchnet.process()
        if FLAGS.train_test_phase == 'train':
            FLAGS.train_test_phase= 'validation'
            matchnet = MatchNet(FLAGS)
            matchnet.process()

def train_multimodal_additional_loss():
    FLAGS.nb_run = 2
    for dataset in['vedai', 'cuhk', 'nirscene']:
        FLAGS.dataset = dataset
        # define the best parameters for the additional loss (contrastive loss)
        FLAGS.balance = 1e-6
        FLAGS.balance_step = 1e-6
        FLAGS.balance_max = 1e-2
        
        if FLAGS.dataset == 'vedai' and FLAGS.pseudo:
            FLAGS.balance_max = 1e-4
        
        # end
        FLAGS.train_test_phase = 'test' if FLAGS.train_test_phase == 'test' else 'train'
        matchnet = MatchNet(FLAGS)
        matchnet.process()
        if FLAGS.train_test_phase == 'train':
            FLAGS.train_test_phase= 'validation'
            matchnet = MatchNet(FLAGS)
            matchnet.process()


def main(_):
    
    working_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
    if FLAGS.experiment == 'multimodal':
        train_multimodal()
    elif FLAGS.experiment == 'multimodalAdditionalLoss':
        train_multimodal_additional_loss()
    elif FLAGS.experiment == '':
        matchnet = MatchNet(FLAGS)
        matchnet.process()
        FLAGS.train_test_phase= 'validation'
        matchnet = MatchNet(FLAGS)
        matchnet.process()
    else:
        print ('unknown experimentation')
        raise NotImplementedError
    os.chdir(working_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.register("type", "bool", lambda v: v.lower() == "true")
    
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
    parser.add_argument("--pseudo", default= False, help='Pseudo or Siamese (default)', action = 'store_true')
    parser.add_argument("--experiment", default= '', help='multimodal')
    
    # specific argument for training matchnet with contrastive loss
    parser.add_argument("--balance", type = float, default = 1e-6, help="balance between two weights, 0 mean do not use contrastive loss")
    parser.add_argument("--balance_step", type = float, default = 1e-6, help="balance between two weights")
    parser.add_argument("--balance_max", type = float, default = 0., help = 'maximum allowed balance between two weights')
    parser.add_argument("--margin", type= float, default = 50, help = 'the upper bound to train contrastive loss, the lower bound is 0')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main= main, argv=[sys.argv[0]] + unparsed)