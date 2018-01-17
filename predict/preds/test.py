import numpy as np
import h5py

h5file = h5py.File("aicha-fsrcn-test-0.4.h5",'r')
preds = np.array(h5file['preds'])
scores = np.array(h5file['scores'])
print scores
