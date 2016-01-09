### Converts numpy files into .mat files

import glob
import numpy as np
import scipy.io as sio

#subj = 'EC77'
#block = ['B27','B43']

# subj = 'EC82'
# block = 'B91'

data_path = '/Users/johncase/Documents/test/'

fils = [f for f in glob.glob(data_path + '*') if not f[-4:] == '.mat']

for f in fils:

    CFC = np.load(f)
    # new_path = data_path + f.split('/' + typ + '/')[1] + '.mat'
    new_path = f + '.mat'
    sio.savemat(new_path,{'CFC':CFC})
    del CFC
