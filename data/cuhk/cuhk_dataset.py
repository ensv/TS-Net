'''
Created on Nov 17, 2017

@author: en
'''

import os, sys
import numpy as np
from scipy.misc.pilutil import imread  # @UnresolvedImport
import shutil
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')
from data.abstract_dataset import Abstract_dataset

class NCUHK(Abstract_dataset):
    '''
    classdocs
    '''

    def __init__(self, nb_fake_images=3, split_ratio=[0.1, 0.7, 0.2], version = 512):
        '''
        Constructor
        '''
        self.working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        Abstract_dataset.__init__(self, nb_fake_images, split_ratio)
        self.version = version
        self._maybe_download()
        self.correct_name()
        os.chdir(self.working_dir)

    def _maybe_download(self):
        
        if os.path.exists('./data/'):
            return
        
        try: os.mkdir('./data')
        except: pass

        try: os.mkdir('./data/cuhk/')
        except: pass

        os.system('wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/CUHK/training_88/Cropped_Images/CUHK_training_cropped_sketches.zip')
        os.system('wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/CUHK/training_88/Cropped_Images/CUHK_training_cropped_photos.zip')
        os.system('wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/CUHK/testing_100/Cropped_Images/CUHK_testing_cropped_sketches.zip')
        os.system('wget http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/CUHK/testing_100/Cropped_Images/CUHK_testing_cropped_photos.zip')
        os.system('unzip CUHK_training_cropped_sketches.zip')
        os.system('unzip CUHK_training_cropped_photos.zip')
        os.system('unzip CUHK_testing_cropped_sketches.zip')
        os.system('unzip CUHK_testing_cropped_photos.zip')
        os.system('mv ./photos/* ./data/cuhk/')
        os.system('mv ./sketches/* ./data/cuhk/')
        os.system('rm -rf ./photos')
        os.system('rm -rf ./sketches')
        os.remove('CUHK_training_cropped_sketches.zip')
        os.remove('CUHK_training_cropped_photos.zip')
        os.remove('CUHK_testing_cropped_sketches.zip')
        os.remove('CUHK_testing_cropped_photos.zip')
        
            
    def correct_name(self):

        for folder in ['./data/cuhk']:
            filenames = [e for e in os.listdir(folder) if e.startswith('F2') or e.startswith('M2')]
            for f in filenames:
                os.rename('%s/%s'%(folder, f), '%s/%s'%(folder, f.replace('F2', 'f').replace('M2', 'm')))

    def get_patch(self, im_path, name):

        co = imread(im_path.replace('f2', 'f').replace('m2', 'm') + '.jpg', 1).astype('uint8')
        ir = imread(im_path + '-sz1.jpg', 1).astype('uint8')

        patch = []
        
        xy = np.zeros((self.nb_fake_images +1, 2), dtype='uint8')
        
        for i in range(0, co.shape[0], 16):
            for j in range(0, co.shape[1], 16):
                
                if i +64 > co.shape[0] or j + 64 > co.shape[1]: continue
                
                left, right = co[i: i + 64, j:j + 64], ir[i: i + 64, j:j + 64]
#                 print (left.shape, right.shape)
                left, right, left_affine, right_affine = self.transform_patch(left, right)
                xy[:, :] = [i+32, j+32]

                patch.append([left, right, [name] * (self.nb_fake_images +1), xy, left_affine, right_affine])
    
        left, right, name, xy, left_affine, right_affine = zip(*patch)
        assert len(left) == len(right) and len(left[0]) == 4
    
        return [np.vstack(left), np.vstack(right), np.hstack(name), np.vstack(xy), np.vstack(left_affine), np.vstack(right_affine)]

    def process_patches(self, im_names, train_test_set, category):
        patch = [self.get_patch('./data/%s/%s'%(category, e), e) for e in im_names]
        
        left, right, name, xy, left_affine, right_affine = zip(*[e for e in patch if e[0] is not None])
        left, right = np.vstack(left), np.vstack(right)
        name, xy = np.hstack(name), np.vstack(xy)
        left_affine, right_affine = np.vstack(left_affine), np.vstack(right_affine)
        self.convert_np_tf_record(left, right, name, xy, left_affine, right_affine, train_test_set, category)
        return self.calcualte_mean_std(left, right)

    
    def split_train_test(self, category):
        
        self.working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        try: os.mkdir('./tfrecord/')
        except: pass
        
        try: os.mkdir('./tfrecord/%s'%category)
        except: pass

        folder = './data/%s/'%category

        image_files = np.array([e.replace('-sz1.jpg', '') for e in os.listdir(folder) if e.endswith('-sz1.jpg')])
        print ('\n==========================')
        print ('total number of images in ', category, ':', image_files.shape[0])
        print ('nb of images in train set:', image_files.shape[0] * self.split_ratio[1])
        print ('nb of images in test set:', image_files.shape[0] * self.split_ratio[2])
        print ('nb of images in validation set:', image_files.shape[0] * self.split_ratio[0])
        print ('\n==========================')

        indx = np.random.randint(0, len(image_files), 5 * len(image_files))
        train_indx = np.unique(indx)[:int(len(image_files) * self.split_ratio[1])]
        test_indx = np.array([i for i in range(len(image_files)) if i not in train_indx])
        validation_indx = test_indx[:int(image_files.shape[0] * self.split_ratio[0])]
        test_indx = test_indx[int(image_files.shape[0] * self.split_ratio[0]):]
        
        print ('nb of train, test and validation images:', train_indx.shape[0], test_indx.shape[0], validation_indx.shape[0])

        assert train_indx.shape[0] + test_indx.shape[0] + validation_indx.shape[0] == image_files.shape[0]

        for each in train_indx:
            assert each not in test_indx
            assert each not in validation_indx
            
        for each in validation_indx:
            assert each not in test_indx
        
        print ('testing image index', test_indx.shape[0])
        self.train_images = image_files[train_indx]
        self.test_images = image_files[test_indx]
        self.valid_images = image_files[validation_indx]
        
        mean_std_train = self.process_patches(self.train_images, 'train', category)
        mean_std_test = self.process_patches(self.test_images, 'test', category)
        mean_std_valid = self.process_patches(self.valid_images, 'validation', category)

        mean_std = (mean_std_train + mean_std_test + mean_std_valid)/3
        
        np.save('./tfrecord/%s/std_mean'%(category), mean_std)
        os.chdir(self.working_dir)

    def combine_std_mean(self):
        
        self.working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        shutil.copy('./tfrecord/cuhk/std_mean.npy', './tfrecord/')
        os.chdir(self.working_dir)


if __name__ == '__main__':
    
    cuhk = NCUHK()
    cuhk.split_train_test('cuhk')
    cuhk.combine_std_mean()