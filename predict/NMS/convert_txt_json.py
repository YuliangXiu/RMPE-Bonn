import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from numpy.core.records import fromarrays
from PIL import Image
from itertools import compress
import os, pprint
import json
from tqdm import tqdm


aicha_pred_dir = '/home/yuliang/code/multi-human-pose/predict/NMS'
# aicha_gt = os.path.join(aicha_pred_dir, 'keypoint_validation_annotations_20170911.json')
# aicha_gt = os.path.join(aicha_pred_dir, 'submit.json')
# image_dir = '/home/yuliang/code/multi-human-pose/predict/data/images'
index = np.loadtxt('index.txt', delimiter=' ', dtype='S50,i4,i4')
pose_score = np.loadtxt('scores.txt', dtype='S50'+14*',d')
pose = np.loadtxt('pred.txt', dtype='S50'+28*',i4')

# with open(aicha_gt, 'r') as f:
#     gt = json.load(f)

aicha_results = []

vis = False


for idx, item in enumerate(tqdm(index)):
    
    image_id = item[0][:-4]
    human_num = item[2]-item[1]+1
    image_result = {}
    image_result['image_id'] = image_id
    image_result['keypoint_annotations'] = {}
    
    for human_id in xrange(1, human_num+1):
        image_result['keypoint_annotations']['human'+str(human_id)] = np.hstack((np.array([p for p in pose[item[1]+human_id-2] if type(p) is np.int32])\
                                                                                 .reshape(-1,2),np.ones((14,1)))).flatten().tolist()
    aicha_results.append(image_result)
    
    if vis:
        # prediction          
        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        ax = plt.gca()
        plt.imshow(Image.open(os.path.join(image_dir, item[0])))
        for i in xrange(human_num):
            pose_ = np.array([p for p in pose[item[1]-1+i] if type(p) is np.int32]).reshape(-1,2)
            pose_score_ = np.array([p for p in pose_score[idx+i] if type(p) is np.float64]).reshape(-1,1)
            xmin = np.min(pose_[:,0])
            ymin = np.min(pose_[:,1])
            xmax = np.max(pose_[:,0])
            ymax = np.max(pose_[:,1])
            display_txt = '%.2f'%(np.mean(pose_score_))
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='yellow', linewidth=2))
            ax.text(xmin, ymin, display_txt, bbox={'facecolor':'red', 'alpha':0.5})
        
        # groundtruth
        plt.subplot(1,2,2)
        ax = plt.gca()
        for idx_gt in xrange(len(gt)):
            if gt[idx_gt]['image_id'] == image_id:
                human_num_ = len(gt[idx_gt]['keypoint_annotations'].keys())
                plt.imshow(Image.open(os.path.join(image_dir, item[0])))
                for i in xrange(1,human_num_+1):
                    pose_ = np.array(gt[idx_gt]['keypoint_annotations']['human'+str(i)]).reshape(-1,3)
                    xmin = np.min(pose_[:,0])
                    ymin = np.min(pose_[:,1])
                    xmax = np.max(pose_[:,0])
                    ymax = np.max(pose_[:,1])
                    coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
                    ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='yellow', linewidth=2))
    
#     if idx == 5:
#         break

final = json.dumps(aicha_results)
with open('submit-valid-0.4.json','w') as f:
    f.write(final)
    
