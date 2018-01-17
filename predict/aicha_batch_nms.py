# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:58:09 2016

@author: benjamin
"""

import json
import numpy as np
import h5py
import os
import math
from tqdm import tqdm
import time
import argparse
import pprint

from scipy.io import loadmat, savemat
from numpy.core.records import fromarrays
from PIL import Image
from itertools import compress

def load_annotations(anno_file, return_dict):
    """Convert annotation JSON file."""

    annotations = dict()
    annotations['image_ids'] = set([])
    annotations['annos'] = dict()
    annotations['delta'] = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
                                       0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
                                       0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])
    try:
        annos = json.load(open(anno_file, 'r'))
    except Exception:
        return_dict['error'] = 'Annotation file does not exist or is an invalid JSON file.'
        exit(return_dict['error'])

    for anno in annos:
        annotations['image_ids'].add(anno['image_id'])
        annotations['annos'][anno['image_id']] = dict()
        annotations['annos'][anno['image_id']]['human_annos'] = anno['human_annotations']
        annotations['annos'][anno['image_id']]['keypoint_annos'] = anno['keypoint_annotations']

    return annotations


def load_predictions(prediction_file, return_dict):
    """Convert prediction JSON file."""

    predictions = dict()
    predictions['image_ids'] = []
    predictions['annos'] = dict()
    id_set = set([])

    try:
        preds = json.load(open(prediction_file, 'r'))
    except Exception:
        return_dict['error'] = 'Prediction file does not exist or is an invalid JSON file.'
        exit(return_dict['error'])

    for pred in preds:
        if 'image_id' not in pred.keys():
            return_dict['warning'].append('There is an invalid annotation info, \
                likely missing key \'image_id\'.')
            continue
        if 'keypoint_annotations' not in pred.keys():
            return_dict['warning'].append(pred['image_id']+\
                ' does not have key \'keypoint_annotations\'.')
            continue
        image_id = pred['image_id'].split('.')[0]
        if image_id in id_set:
            return_dict['warning'].append(pred['image_id']+\
                ' is duplicated in prediction JSON file.')
        else:
            id_set.add(image_id)
        predictions['image_ids'].append(image_id)
        predictions['annos'][pred['image_id']] = dict()
        predictions['annos'][pred['image_id']]['keypoint_annos'] = pred['keypoint_annotations']

    return predictions


def compute_oks(anno, predict, delta):
    """Compute oks matrix (size gtN*pN)."""

    anno_count = len(anno['keypoint_annos'].keys())
    predict_count = len(predict.keys())
    oks = np.zeros((anno_count, predict_count))

    # for every human keypoint annotation
    for i in range(anno_count):
        anno_key = anno['keypoint_annos'].keys()[i]
        anno_keypoints = np.reshape(anno['keypoint_annos'][anno_key], (14, 3))
        visible = anno_keypoints[:, 2] == 1
        bbox = anno['human_annos'][anno_key]
        scale = np.float32((bbox[3]-bbox[1])*(bbox[2]-bbox[0]))
        if scale == 0:
            return ['error']
        if np.sum(visible) == 0:
            for j in range(predict_count):
                oks[i, j] = 0
        else:
            # for every predicted human
            for j in range(predict_count):
                predict_key = predict.keys()[j]
                predict_keypoints = np.reshape(predict[predict_key], (14, 3))
                dis = np.sum((anno_keypoints[visible, :2] \
                    - predict_keypoints[visible, :2])**2, axis=1)
                oks[i, j] = np.mean(np.exp(-dis/2/delta[visible]**2/scale))
    return oks


def keypoint_eval(predictions, annotations, return_dict):
    """Evaluate predicted_file and return mAP."""

    oks_all = np.zeros((0))
    oks_num = 0

    # for every annotation in our test/validation set
    for image_id in tqdm(annotations['image_ids']):
        # if the image in the predictions, then compute oks
        if image_id in predictions['image_ids']:
            oks = compute_oks(anno=annotations['annos'][image_id], \
                              predict=predictions['annos'][image_id]['keypoint_annos'], \
                              delta=annotations['delta'])
            if 'error' in oks:
                return_dict['warning'].append(image_id+' bbox is invalid.')
                continue

            # view pairs with max OKSs as match ones, add to oks_all
            oks_all = np.concatenate((oks_all, np.max(oks, axis=0)), axis=0)
            # accumulate total num by max(gtN,pN)
            oks_num += np.max(oks.shape)
        else:
            # otherwise report warning
            return_dict['warning'].append(image_id+' is not in the prediction JSON file.')
            # number of humen in ground truth annotations
            gt_n = len(annotations['annos'][image_id]['human_annos'].keys())
            # fill 0 in oks scores
            oks_all = np.concatenate((oks_all, np.zeros((gt_n))), axis=0)
            # accumulate total num by ground truth number
            oks_num += gt_n

    # compute mAP by APs under different oks thresholds
    average_precision = []
    for threshold in np.linspace(0.5, 0.95, 10):
        average_precision.append(np.sum(oks_all > threshold)/np.float32(oks_num))
    return_dict['score'] = np.mean(average_precision)

    return return_dict



def PCK_match(pick_preds, all_preds,ref_dist):
    dist = np.sqrt(np.sum(np.square(pick_preds[np.newaxis,:]-all_preds),axis=2))
    ref_dist=min(ref_dist,7)
    num_match_keypoints = np.sum(dist/ref_dist <= 1,axis=1)

    face_index = np.zeros(dist.shape)
    for i in range(dist.shape[0]):
        for j in range(5):
            face_index[i][j]=1

    face_match_keypoints = np.sum((dist/10 <= 1) & (face_index == 1),axis=1)
    return num_match_keypoints, face_match_keypoints

def test_parametric_pose_NMS(delta1,delta2,mu,gamma,nms,data,scoreThred):
    scoreThreds = scoreThred
    matchThreds = 1
    
    #prepare data
    h5file = h5py.File("./preds/aicha-fsrcn-{}-{}.h5".format(data,nms),'r')
    preds = np.array(h5file['preds'])
    scores = np.array(h5file['scores'])  
    scores[scores==0] = 1e-5  
    indexs = [line.rstrip(' ').rstrip('\r').rstrip('\n') for line in open("annot/aicha-fsrcn-{}-{}/index.txt".format(data,nms))]
    imgIDs = np.array([(line[0:-5]) for line in open("annot/aicha-fsrcn-{}-{}/test-bbox_images.txt".format(data,nms))])
    scores_proposals = np.loadtxt("annot/aicha-fsrcn-{}-{}/score.txt".format(data,nms))
    
    #get bounding box sizes    
    bbox_file = h5py.File("annot/aicha-fsrcn-{}-{}/test-bbox.h5".format(data,nms),'r')
    xmax=np.array(bbox_file['xmax']); xmin=bbox_file['xmin']; ymax=np.array(bbox_file['ymax']); ymin=bbox_file['ymin']
    widths=xmax-xmin; heights=ymax-ymin;
    alpha = 0.1
    Sizes=alpha*np.maximum(widths,heights)
    
    #set the corresponding dir
    if (os.path.exists("./NMS") == False):
        os.mkdir("./NMS")

    os.chdir("./NMS")
    
    NMS_preds = open("pred.txt",'w')
    NMS_scores = open("scores.txt",'w')
    proposal_scores = open("scores-proposals.txt",'w')
    NMS_index = open("index.txt",'w')
    NMS_ids = open("imgId_file.txt",'w')
    num_human = 0
    
    #loop through every image
    for i in tqdm(xrange(len(indexs))):
        index = indexs[i].split(' '); 
        img_name = index[0]; start = int(index[1])-1; end = int(index[2])-1;
        
        #initialize scores and preds coordinates
        img_preds = preds[start:end+1]; img_scores = np.mean(scores[start:end+1],axis = 1)
        img_ids = np.arange(end-start+1); ref_dists = Sizes[start:end+1];keypoint_scores = scores[start:end+1];
        Ids = imgIDs[start:end+1]; 
        pro_score = scores_proposals[start:end+1]
        
        #do NMS by parametric
        pick = []
        merge_ids = []
        while(img_scores.size != 0):
            
            #pick the one with highest score
            pick_id = np.argmax(img_scores)  
            pick.append(img_ids[pick_id])
            
            #get numbers of match keypoints by calling PCK_match 
            ref_dist=ref_dists[img_ids[pick_id]]
            simi = get_parametric_distance(pick_id,img_preds, keypoint_scores,ref_dist, delta1, delta2, mu)
            num_match_keypoints,face_match_keypoints = PCK_match(img_preds[pick_id],img_preds,ref_dist)

            #delete humans who have more than matchThreds keypoints overlap with the seletced human.
            delete_ids = np.arange(img_scores.shape[0])[simi > gamma | (num_match_keypoints >= matchThreds)]
            if (delete_ids.size == 0):
                delete_ids = pick_id
            merge_ids.append(img_ids[delete_ids])
            img_preds = np.delete(img_preds,delete_ids,axis=0); img_scores = np.delete(img_scores, delete_ids)
            img_ids = np.delete(img_ids, delete_ids); keypoint_scores = np.delete(keypoint_scores,delete_ids,axis=0)
        
        #write the NMS result to files
        pick = [Id+start for Id in pick] 
        merge_ids = [Id+start for Id in merge_ids]
        assert len(merge_ids) == len(pick)
        preds_pick = preds[pick]; scores_pick = scores[pick];pick_ids = imgIDs[pick]
        num_pick = 0
        for j in xrange(len(pick)):
            
            #first compute the average score of a person
            ids = np.arange(14)
            # if (scores_pick[j,0,0] < 0.1): ids = np.delete(ids,0);
            # if (scores_pick[j,5,0] < 0.1): ids = np.delete(ids,5);
            mean_score = np.mean(scores_pick[j,ids,0])
            max_score = np.max(scores_pick[j,ids,0])
            if (max_score < scoreThreds):
                continue
            
            # merge poses
            merge_id = merge_ids[j]  
            pick_id = pick_ids[j]
            score = scores_proposals[pick[j]]
            merge_poses,merge_score = merge_pose(preds_pick[j],preds[merge_id],scores[merge_id],Sizes[pick[j]])
            
            ids = np.arange(14)
            # if (merge_score[0] < 0.1): ids = np.delete(ids,0);
            # if (merge_score[5] < 0.1): ids = np.delete(ids,5);
            mean_score = np.mean(merge_score[ids])
            max_score = np.max(merge_score[ids])
            if (max_score < scoreThreds):
                continue
            
            #add the person to predict
            num_pick += 1
            NMS_preds.write("{}".format(img_name))
            NMS_scores.write("{}".format(img_name))
            NMS_ids.write("{}\n".format((pick_id)))
            proposal_scores.write("{}\n".format(score))
            
            # for point_id in xrange(14):
            #     NMS_preds.write("\t{}\t{}".format(int(merge_poses[point_id,0]),int(merge_poses[point_id,1])))
            #     NMS_scores.write("\t{}".format(merge_score[point_id]))
            # NMS_preds.write("\n")
            # NMS_scores.write("\n")
            for point_id in xrange(14):
#                NMS_preds.write("\t{}\t{}".format(preds_pick[j,point_id,0],preds_pick[j,point_id,1]))
#                NMS_scores.write("\t{}".format(scores_pick[j,point_id,0]))
                if math.isnan(merge_poses[point_id,0]):
                   merge_poses[point_id,0] = 0
                   print point_id
                if math.isnan(merge_poses[point_id,1]):
                   merge_poses[point_id,1] = 1
                NMS_preds.write("\t{}\t{}".format(int(merge_poses[point_id,0]),int(merge_poses[point_id,1])))
                NMS_scores.write("\t{}".format(merge_score[point_id]))
            NMS_preds.write("\n")
            NMS_scores.write("\n")
        NMS_index.write("{} {} {}\n".format(img_name, num_human+1, num_human + num_pick))
        num_human += num_pick
        
    NMS_preds.close();NMS_scores.close();NMS_index.close();NMS_ids.close(); proposal_scores.close()
    
def get_parametric_distance(i,all_preds, keypoint_scores,ref_dist, delta1, delta2, mu):
    pick_preds = all_preds[i]
    pred_scores = keypoint_scores[i]
    dist = np.sqrt(np.sum(np.square(pick_preds[np.newaxis,:]-all_preds),axis=2))/ref_dist
    mask = (dist <= 1)
    # defien a keypoints distances
    score_dists = np.zeros([all_preds.shape[0], 14])
    keypoint_scores = np.squeeze(keypoint_scores)
    if (keypoint_scores.ndim == 1) :
        keypoint_scores = keypoint_scores[np.newaxis,:]
    # the predicted scores are repeated up to do boastcast
    pred_scores = np.tile(pred_scores, [1,all_preds.shape[0]]).T
    score_dists[mask] = np.tanh(pred_scores[mask]/delta1)*np.tanh(keypoint_scores[mask]/delta1)
    # if the keypoint isn't inside the bbox, set the distance to be 10
#    dist[dist>1] = 10
    point_dist = np.exp((-1)*dist/delta2)
    final_dist = np.sum(score_dists,axis=1)+mu*np.sum(point_dist,axis=1)
    return final_dist
    
def merge_pose(refer_pose, cluster_preds, cluster_keypoint_scores, ref_dist):
    dist = np.sqrt(np.sum(np.square(refer_pose[np.newaxis,:]-cluster_preds),axis=2))
    # mask is an nx16 matrix
    mask = (dist <= ref_dist)
    final_pose = np.zeros([14,2]); final_scores = np.zeros(14)
    if (cluster_preds.ndim == 2):
        cluster_preds = cluster_preds[np.newaxis,:,:]
        cluster_keypoint_scores = cluster_keypoint_scores[np.newaxis,:]
    if (mask.ndim == 1):
        mask = mask[np.newaxis,:]
    for i in xrange(14):
        cluster_joint_scores = cluster_keypoint_scores[:,i][mask[:,i]]
        
        # pick the corresponding i's matched keyjoint locations and do an weighed sum.
        cluster_joint_location = cluster_preds[:,i,:][np.tile(mask[:,i,np.newaxis],(1,2))].reshape(np.sum(mask[:,i,np.newaxis]),-1)

        # get an normalized score
        normed_scores = cluster_joint_scores / np.sum(cluster_joint_scores)
        # merge poses by a weighted sum
        final_pose[i,0] = np.dot(cluster_joint_location[:,0], normed_scores)
        final_pose[i,1] = np.dot(cluster_joint_location[:,1], normed_scores)
        final_scores[i] = np.max(cluster_joint_scores)
    return final_pose, final_scores
    

# def get_result():
#     delta1 = 1; mu = 2.5; delta2 = 2.5;
#     gamma = 24; nms = 0.4; data = 'valid';scoreThred = 0.4;
#     test_parametric_pose_NMS(delta1, delta2, mu, gamma, nms, data, scoreThred)

if __name__ == '__main__':

    nms = 0.1; data='valid';
    delta1 = 1; mu = 2.5; delta2 = 2.5;
    gamma = 24; scoreThred = 0.6;
    with open('/home/yuliang/code/multi-human-pose/predict/params_log.txt','w') as fuck:
        for delta2 in np.linspace(0.1,5,5):
            for gamma in np.linspace(20,40,10):
                for mu in np.linspace(0.1,5,5):
                    for scoreThred in np.linspace(0.1,1,10):
                        params = 'delta1:{} delta2:{} mu:{} gamma:{} scoreThred:{} \n'.format(delta1,delta2,mu,gamma,scoreThred)
                        print params
                        fuck.write(params)

                        os.chdir('/home/yuliang/code/multi-human-pose/predict')
                        test_parametric_pose_NMS(delta1, delta2, mu, int(gamma), nms, data, scoreThred)
                    # get_result()

                        aicha_pred_dir = '/home/yuliang/code/multi-human-pose/predict/NMS'
                        os.chdir(aicha_pred_dir)
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
                        with open('submit-valid-0.1.json','w') as f:
                            f.write(final)
                        # Initialize return_dict
                        return_dict = dict()
                        return_dict['error'] = None
                        return_dict['warning'] = []
                        return_dict['score'] = None

                        # Load annotation JSON file
                        # start_time = time.time()
                        annotations = load_annotations(anno_file='keypoint_validation_annotations_20170911.json',
                                                       return_dict=return_dict)
                        # print 'Complete reading annotation JSON file in %.2f seconds.' %(time.time() - start_time)

                        # Load prediction JSON file
                        # start_time = time.time()
                        predictions = load_predictions(prediction_file='submit-valid-0.1.json',
                                                       return_dict=return_dict)
                        # print 'Complete reading prediction JSON file in %.2f seconds.' %(time.time() - start_time)

                        # Keypoint evaluation
                        # start_time = time.time()
                        return_dict = keypoint_eval(predictions=predictions,
                                                    annotations=annotations,
                                                    return_dict=return_dict)
                        # print 'Complete evaluation in %.2f seconds.' %(time.time() - start_time)

                        # Print return_dict and final score
                        # pprint.pprint(return_dict)
                        result_str = 'Score: {} \n'.format(return_dict['score'])
                        print result_str
                        fuck.write(result_str)
                        # print 'delta1:',delta1,' delta2:',delta2, ' mu:',mu, ' gamma:',gamma,' scoreThred:',scoreThred

