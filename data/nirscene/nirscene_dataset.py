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

class NNIR(Abstract_dataset):
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

        os.chdir(self.working_dir)

    def _maybe_download(self):
        
        if os.path.exists('./data/'):
            return
        
        os.system('wget http://ivrgwww.epfl.ch/supplementary_material/cvpr11/nirscene1.zip')
        os.system('unzip nirscene1.zip')
        shutil.move('./nirscene1', './data/')
        os.remove('nirscene1.zip')


    def get_patch(self, im_path, name):
        

        try:
            co = imread(im_path + '_rgb.tiff', 1).astype('uint8')
            ir = imread(im_path + '_nir.tiff', 1).astype('uint8')
        except:
            print ('file corrupted', im_path)
            return [None, None, None, None, None, None]

        patch = []
        
        xy = np.zeros((self.nb_fake_images +1, 2), dtype='uint8')
        
        for i in range(co.shape[0] // 64):
            for j in range(co.shape[1] // 64):
                
                left, right = co[i*64: i*64 + 64, j*64:j*64 + 64], ir[i*64: i*64 + 64, j*64:j*64 + 64]
                left, right, left_affine, right_affine = self.transform_patch(left, right)
                xy[:, :] = [i+32, j+32]

                patch.append([left, right, [name] * (self.nb_fake_images +1), xy, left_affine, right_affine])
    
        left, right, name, xy, left_affine, right_affine = zip(*patch)
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

        image_files = np.array([e.replace('_rgb.tiff', '') for e in os.listdir('./data/%s'%category) if e.endswith('_rgb.tiff')])
        print ('\n==========================', category)
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
        
        categories = ['country', 'field', 'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water']
        self.working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self._combine_std_mean(categories)
        os.chdir(self.working_dir)

if __name__ == '__main__':
    
    all_category = ['country', 'field', 'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water']

    for each_cat in all_category:
        nir = NNIR()
        nir.split_train_test(each_cat)
    nir = NNIR()
    nir.combine_std_mean()
