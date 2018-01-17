
# coding: utf-8
import numpy as np
from tqdm import tqdm
import os
import h5py
import scipy.io
from itertools import compress

# Make sure that caffe is on the python path:
caffe_root = '/home/yuliang/code/RMPE-poseflow'  # this file is expected to be in {caffe_root}/examples/mppp
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()


model_def = 'models/VGG_SSD/deploy.prototxt'
model_weights = 'models/VGG_SSD/VGG_MPII_COCO14_SSD_500x500_iter_60000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# ### 2. SSD detection

image_resize = 500
net.blobs['data'].reshape(1,3,image_resize,image_resize)

#root_dir = "/home/yuliang/data/ai_challenger_keypoint_train_20170902/ai_challenger_keypoint_validation_20170911"
root_dir = "/home/yuliang/data/ai_challenger_keypoint/ai_challenger_keypoint_test_a_20170923"
#image_dir = os.path.join(root_dir,'keypoint_validation_images_20170911')
image_dir = os.path.join(root_dir,'keypoint_test_a_images_20170923')

test_list_file = os.path.join(root_dir, 'text_list.txt')
print test_list_file
                   
if not os.path.exists(test_list_file):
    with open(os.path.join(root_dir,'test_list.txt'),'w') as f:
            for _,_,imgs in os.walk(image_dir):
                for img in sorted(imgs):
                    f.write(img+'\n')


configThred = 0.6
NMSThred = 0.45

lines = [line.rstrip('\n') .rstrip('\r') for line in open(os.path.join(root_dir,'test_list.txt'))]

directory = 'examples/ai-cha/aicha-test-'+str(configThred)+'/'
if not os.path.exists(directory):
    os.makedirs(directory)
results = open(directory+"test-bbox_images.txt", 'w')
score_file = open(directory+"score.txt",'w')
index_file = open(directory+"index.txt",'w')

FileLength = len(lines)



num_boxes=0

xminarr=[]
yminarr=[]
xmaxarr=[]
ymaxarr=[]

for i in tqdm(xrange(FileLength)):
#     if ((i%1000) == 0):
#         print i
    picture = lines[i].split("\t")
    p_name = picture[0]

    filename = os.path.join(image_dir, p_name)

    if (os.path.isfile(filename) == False):
        print filename + " does not exist"
        continue
    image = caffe.io.load_image(filename)

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    
    # Forward pass.
    detections = (net.forward()['detection_out'])

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    top_indices = [m for m, conf in enumerate(det_conf) if conf > configThred]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = det_label[top_indices]
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    if(top_conf.shape[0]!=0):
        index_file.write("{} {} ".format(p_name,num_boxes+1))
    for k in xrange(top_conf.shape[0]):
        label = top_labels[k]
        if (label != 1):
            continue
        xmin = int(round(top_xmin[k] * image.shape[1]))
        ymin = int(round(top_ymin[k] * image.shape[0]))
        xmax = int(round(top_xmax[k] * image.shape[1]))
        ymax = int(round(top_ymax[k] * image.shape[0]))
        score = top_conf[k]

        if xmin>=xmax or ymin>=ymax or xmax>image.shape[1] or ymax>image.shape[0]:
         print 'error', p_name
        xminarr.append(xmin);yminarr.append(ymin);xmaxarr.append(xmax);ymaxarr.append(ymax);
        results.write("{}\n".format(p_name))
        score_file.write("{}\n".format(score))

        num_boxes += 1
    
    if(top_conf.shape[0]!=0):
        index_file.write("{}\n".format(num_boxes))

print "Average Boxes per image:", float(num_boxes)/FileLength
results.close()    
score_file.close()
index_file.close()
with h5py.File(directory+'test-bbox.h5', 'w') as hf:
                hf.create_dataset('xmin', data=np.array(xminarr))
                hf.create_dataset('ymin', data=np.array(yminarr))
                hf.create_dataset('xmax', data=np.array(
                    xmaxarr))
                hf.create_dataset('ymax', data=np.array(ymaxarr))
print "Done"

print num_boxes
