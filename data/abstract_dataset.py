'''
Created on Dec 22, 2017

@author: en
'''

import numpy as np
import os
from skimage.transform._geometric import AffineTransform
from skimage import transform
import random
import tensorflow as tf
from sklearn.utils import shuffle


class Abstract_dataset():
    '''
    classdocs
    '''
    
    def __init__(self, nb_fake_images, split_ratio):

        self.split_ratio = split_ratio
        self.nb_fake_images = nb_fake_images
        try: os.mkdir('resources')
        except: pass
        random.seed(os.getpid())
        
    def _load(self):
        raise NotImplementedError

    def _transform(self, im):
        
        ims, affines =[im], [[0, 1, 1, 0, 0]]
        
        for _ in range(self.nb_fake_images):
            
            rotation = random.randint(1, 6) * 3
            scale_x = random.randint(8, 10) / 10
            scale_y = random.randint(8, 10) / 10
            tran_x = random.randint(0, 10) - 5
            tran_y = random.randint(0, 10) - 5
            rotation_direction = np.random.randint(0, 100) % 2
            if rotation_direction != 0:
                rotation = -rotation
                
            affine_matrix = AffineTransform(rotation=np.deg2rad(rotation), scale=(scale_x, scale_y), translation = (tran_x, tran_y))
            new_im = transform.warp(im, affine_matrix, mode = 'symmetric')
            new_im = np.array(new_im) * 255
            np.putmask(new_im, new_im > 255, 255)
            new_im = new_im.astype('uint8')
            
            ims.append(new_im)
            affines.append([rotation, scale_x, scale_y, tran_x, tran_y])

        return np.array(ims), np.array(affines, dtype='i1')
    
    def transform_patch(self, left, right):
        left_ims, left_affines = self._transform(left)
        right_ims, right_affines = self._transform(right)
        return left_ims, right_ims, left_affines, right_affines
    

    def calcualte_mean_std(self, left, right):
        if left.shape[0] >= 200000:
            return np.array([np.mean(left[::10].astype('f4')), 
                             np.std(left[::10].astype('f4')), 
                             np.mean(right[::10].astype('f4')), 
                             np.std(right[::10].astype('f4'))])

        left = left.astype('f4')
        right = right.astype('f4')
                
        left_mean, left_std = np.mean(left), np.std(left)
        right_mean, right_std = np.mean(right), np.std(right)
        
        print (left_mean, left_std, right_mean, right_std)
        return np.array([left_mean, left_std, right_mean, right_std])


    def pairing(self, nb_patches, train_test_set):
        pos_matches = np.zeros((nb_patches, 2), dtype='int32')
        neg_matches = np.zeros((nb_patches, 2), dtype='int32')

        index = np.arange(0, nb_patches, 4)

        pos_matches[:, 0] = np.arange(0, nb_patches)
        pos_matches[:, 1] = np.arange(0, nb_patches)
        neg_matches[:, 0] = np.arange(0, nb_patches)

        nb_fake_images = self.nb_fake_images + 1
        
        for i in range(0, pos_matches.shape[0], nb_fake_images):

            tmp_index = np.concatenate((index[np.where(index <256*(i//256))[0]], index[np.where(index >256 + 256*(i//256))[0]])) 
            np.random.shuffle(tmp_index)

            selected = tmp_index[0]
        
            if index.shape[0] >= nb_patches//10:
                index = index[index != selected]

            neg_matches[i:i+4, 1] = np.arange(selected, selected + 4)
            
        if train_test_set == 'train':
            return pos_matches, neg_matches
    
        index = np.arange(nb_patches).reshape((nb_patches //4, 4))
        list(map(np.random.shuffle, index))
        
        index = index[:, 0]
        return pos_matches[index], neg_matches[index]
        
    
    def convert_2_example(self, left, right, label, name_left, name_right, xy_left, xy_right, affine_left, affine_right, index):
        example = tf.train.Example(features=tf.train.Features(feature={
            'left': tf.train.Feature(bytes_list=tf.train.BytesList(value=[left.tostring()])),
            'right': tf.train.Feature(bytes_list=tf.train.BytesList(value=[right.tostring()])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()])),
            'left_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name_left.tostring()])),
            'right_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name_right.tostring()])),
            'xy_left': tf.train.Feature(bytes_list=tf.train.BytesList(value=[xy_left.tostring()])),
            'xy_right': tf.train.Feature(bytes_list=tf.train.BytesList(value=[xy_right.tostring()])),
            'affine_left': tf.train.Feature(bytes_list=tf.train.BytesList(value=[affine_left.tostring()])),
            'affine_right': tf.train.Feature(bytes_list=tf.train.BytesList(value=[affine_right.tostring()])),
            'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            
        }))
        return example
    

    def plot_patches(self, left, right, label, train_test_set, category):
        
        if os.path.exists('./tfrecord/%s/%s.png'%(category, train_test_set)): return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(7, 8,  2*i + 1)
            plt.imshow(left[i], cmap='Greys')
            plt.title('%d_lb:%d'%(i, label[i]))
            plt.subplot(7, 8, 2*i + 2)
            plt.imshow(right[i], cmap='Greys')
            plt.title('%d_lb:%d'%(i, label[i]))
        plt.savefig('./tfrecord/%s/%s.png'%(category, train_test_set))
    
    def convert_np_tf_record(self, left, right, name, xy, left_affine, right_affine, train_test_set, category):
        
        assert left.dtype == np.zeros((), dtype='uint8').dtype
        
        print (left.shape, right.shape, name.shape, xy.shape, left_affine.shape, right_affine.shape)
        assert left.shape[0] == right.shape[0] == name.shape[0] == left_affine.shape[0] == right_affine.shape[0] == xy.shape[0]
        
        step = 64000
        for i in range(0, left.shape[0], step):

            pos, neg = self.pairing(min(step, left[i:i+step].shape[0]), train_test_set)
            if train_test_set == 'train':
                assert pos.shape[0] == min(step, left[i:i+step].shape[0])
#                 pos, neg = shuffle(pos), shuffle(neg)
            else:
                assert pos.shape[0] == min(step, left[i:i+step].shape[0]) //4
            
            assert pos.shape[0] == pos.shape[0] and neg.shape[1] == pos.shape[1] and neg.shape[1] == 2
      
            all_index = np.hstack((pos, neg)).reshape((-1, 2)) + i
            
            batch_left, batch_right = left[all_index[:, 0]], right[all_index[:, 1]]
            batch_name_left, batch_name_right = name[all_index[:, 0]], name[all_index[:, 1]]
            batch_xy_left, batch_xy_right = xy[all_index[:, 0]], xy[all_index[:, 1]]
            batch_affine_left, batch_affine_right = left_affine[all_index[:, 0]], right_affine[all_index[:, 1]]
            
            labels = np.zeros((batch_left.shape[0], 2), dtype='i4')
            labels[::2, 0] = 1
            labels[1::2, 1] = 1
            
            self.plot_patches(batch_left[:50], batch_right[:50], labels[:50, 0], train_test_set, category)
            
            with tf.python_io.TFRecordWriter('./tfrecord/%s/%s_%d.tfrecord'%(category, train_test_set, i)) as tfrecord_writer:
                for j in range(0, labels.shape[0]):
                    example = self.convert_2_example(batch_left[j], batch_right[j], labels[j], 
                                                     batch_name_left[j], batch_name_right[j],
                                                     batch_xy_left[j], batch_xy_right[j],
                                                     batch_affine_left[j], batch_affine_right[j], i + j)
                    
                    tfrecord_writer.write(example.SerializeToString())
                    
    def _combine_std_mean(self, categories):
        mean_std = [np.load('./tfrecord/%s/std_mean.npy'%e) for e in categories]
        mean_std = np.vstack(mean_std)
        mean_std = np.mean(mean_std, axis=0)
        np.save('./tfrecord/std_mean', mean_std)