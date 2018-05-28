'''
Created on Jan 15, 2018

@author: en
'''
import os
import numpy as np
from sklearn.externals import joblib  # @Reimport
import os  # @Reimport
import operator



def check_result_valid():
    import os
    import numpy as np
    from sklearn.externals import joblib
    for e in os.listdir('.'):
        if not os.path.isdir(e): continue
        for f in os.listdir(e):
            if not os.path.isdir('%s/%s'%(e, f)): continue
            try:
                re = [int(each.split('model.ckpt-')[1]) for each, _ in joblib.load('%s/%s/result_valid.pkl'%(e, f)).items()]
                print('%s/%s\t%d'%(e, f, np.max(re)))
            except:
                print('No')
                
def check_done():
    import os
    from sklearn.externals import joblib
    nb_good, nb_bad = 0, 0
    for e in os.listdir('.'):
        if not os.path.isdir(e): continue
        for f in os.listdir(e):
            if not os.path.isdir('%s/%s'%(e, f)): continue
            try:
                if open('%s/%s/done.txt'%(e, f), 'r').readlines()[0].startswith('done'): 
                    nb_good += 1
                else:
                    print('%s/%s'%(e, f))
                    nb_bad +=1 
            except:
                print('%s/%s'%(e, f))
                nb_bad += 1
    print ('good = %d, bad = %d'%(nb_good, nb_bad))

# def smooth(x,window_len=11,window='hanning'):
#     
#     if window_len<3:
#         return x
#     
#     s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
#     #print(len(s))
#     if window == 'flat': #moving average
#         w=np.ones(window_len,'d')
#     else:
#         w=eval('np.'+window+'(window_len)')
#     
#     y=np.convolve(w/w.sum(),s,mode='valid')
#     return y

def smooth(x, window_len=5):
    new_x = np.zeros((len(x)), 'f4')
    new_x[:] = x[:]
    half_window = window_len//2
    for i in range(window_len//2, len(x) - window_len//2):
        new_x[i] = np.mean(new_x[i-half_window:i+half_window])
    return new_x

def print_std_mean(folder, use_smooth = False):
    tmp = []
    
    if not os.path.exists(folder): 
        print('0.0, 0.0\t', end='')
        return
    

    for e in os.listdir(folder):

        if not os.path.isdir(folder): continue
        
        if not os.path.exists(folder + '/%s/result_valid.pkl'%e): continue


        re = [v[0.95] for _, v in joblib.load(folder + '/%s/result_valid.pkl'%e).items()]
        if use_smooth:
            smoothed_re = smooth(re)
            index = np.argmin(smoothed_re)
            tmp.append(re[index])
        else:
            tmp.append(np.min(re))

    print('%0.2f, %0.2f \t'%(np.mean(tmp)*100, np.std(tmp)*100), end='')
    return np.mean(tmp)  


def should_correct_cpt(path):
    
    txt = open('%s/checkpoint'%path).readlines()[0]

    tmp = txt.split('"')
    
    if os.path.exists(tmp[1]+'.index'):

        if path == tmp[1].split('/model.ckpt')[0]:
            return False
        else:
            return True
    return True

def correct_path(path):
    path = os.path.abspath(path)
    txt = open('%s/checkpoint'%path).readlines()
    with open('%s/checkpoint'%path, 'w') as f:
        for each in txt:
            first, cpt_path, end = each.split('"')
            fld_path, _ = cpt_path.split('/model.ckpt')
            cpt_path = cpt_path.replace(fld_path, path)
            f.write('%s"%s"%s'%(first, cpt_path, end))


def correct_cpt_path():
    for each in os.listdir('./'):
        if not os.path.isdir(each): continue
        
        for e in os.listdir(each):

            if not os.path.isdir('%s/%s'%(each, e)): continue
    
            if should_correct_cpt(os.path.abspath('%s/%s'%(each, e))):

                correct_path(os.path.abspath('%s/%s'%(each, e)))

def print_matchnet_table(path):
    for optimization in ['sgd', 'sgdm']:
        for lr in [0.01, 0.001]:
            for wd in [0., 0.001]:
                for concat in [True, False]:
                    for dropout in [0.5, 0.]:
                        print_std_mean('%s_32_entropy_%0.4f_0.95_1000_151_%0.2f_0_%0.4f_5_0_%d'%(optimization, lr, dropout, wd, concat))
                print('')
                

def print_matchnetlecun_table(path):
    for wd in [0., 1e-2]:
        for optimization in ['sgd', 'sgdm']:
            for lr in [0.01]:
                for margin in [1, 5, 10, 20, 30, 40, 50, 100, -1]:
                    for concate in [False]:
                        for dropout in [ 0.5, 0.]:
                            for balance_max in [0.01, 0.001, 0.0001]:
                                normalization = margin == -1
                                if normalization:
                                    print_std_mean('%s_32_entropy_%0.4f_0.95_1000_151_%0.2f_0_%0.4f_%d_0_%0.4f_%0.4f_%d_1'%(optimization, lr, dropout, wd, abs(margin), balance_max * 100, balance_max, concate))
                                else:
                                    print_std_mean('%s_32_entropy_%0.4f_0.95_1000_151_%0.2f_0_%0.4f_%d_0_%0.4f_%0.4f_%d'%(optimization, lr, dropout, wd, abs(margin), balance_max * 100, balance_max, concate))
                        print('')
        print('')
        print('')    
        
def print_tsnet_size_table():
    for optimization in ['sgdm']:
        for lr in [0.01]:
            for wd in [0., 1e-2]:
                for conv_filter_factor in [0.4, 0.6, 0.8, 1.0]:
                    for concate in [False, True]:
                        for concate_1 in [False, True]:
                            print_std_mean('%s_32_entropy_0.0100_0.95_1000_201_0.00_0_%0.4f_40_0_0.0000_0.0000_%d_%0.1f_0_%d'%(
                                optimization, wd, concate, conv_filter_factor, concate_1))
                    print('')
                    
def print_tsnet_bottleneck():
    for wd in [1e-2, 1e-3]:
        for dimension in ['1-127-127-1', '32-96-96-32',  '64-64-64-64', '96-32-32-96', '127-1-1-127']:
            print_std_mean('sgdm_32_entropy_0.0100_0.95_1000_201_0.00_0_%0.4f_40_0_0.0000_0.0000_0_1.0_0_0_%s'%(wd, dimension))
        print('')    

def print_tsnetbackup_size_table():
    for optimization in ['sgd', 'sgdm']:
        for concate in [True, False]:
            result = []
            for conv_filter_factor in [0.4, 0.6, 0.8, 1.0]:
                tmp = print_std_mean('%s_32_entropy_0.0100_0.95_1000_201_0.00_0_0.0000_5_0_0.0000_0.0000_%d_%0.1f/'%(
                    optimization, concate, conv_filter_factor))
                result.append(tmp)
            
            print('%0.2f, %0.2f \t'%(np.mean(result)*100, np.std(result)*100), end='')
            print('')


def combine_data():
    for set in ['train', 'test', 'validation']:
        p = joblib.load('pseudo/%s.pkl.pkl'%(set))
        s = joblib.load('siamese/%s.pkl.pkl'%(set))
        
        c = [np.hstack((p[0], s[0])), np.vstack(p[1])]
        joblib.dump(c, '%s.pkl'%set, compress=3)
    
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
    
if __name__ == '__main__':
#     correct_cpt_path()
#     combine_data()

    validation_x, validation_y = joblib.load('./validation.pkl')
    w = 1/4 * np.array([[1, -1], [-1, 1], [1, -1], [-1, 1]], 'f4')
    
    out = validation_x[:, :2] + validation_x[:, 2:] #np.dot(validation_x, w)
    print(out.shape, validation_x.shape, '\n')
    print (validation_x,  '\n')
    print (out, '\n')
    print(_errorRateAt95((validation_y[:, 0], out[:, 0], 0.95)))
    print(_errorRateAt95((validation_y[:, 0], validation_x[:, 0], 0.95)))
    print(_errorRateAt95((validation_y[:, 0], validation_x[:, 2], 0.95)))
    print()
    print(np.mean(validation_x, axis=0))
    print(np.std(validation_x, axis=0))
    
