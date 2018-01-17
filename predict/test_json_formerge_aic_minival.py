# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:58:09 2016

@author: benjamin
"""

import json
import numpy as np
import h5py
import os
import zipfile
import math

def test_json():
    dataDir="/home/benjamin/HardDisk/coco/"
    dataType="val2014"
    annType="keypoints"
    #resFile='%s/annotations/person_keypoints_%s_fake%s100_results.json'
    resFile='%s/annotations/person_keypoints_train2014.json'
    resFile = resFile%(dataDir)
    with open(resFile) as jsonFile:
        json_data = json.load(jsonFile)
    print json_data['annotations'][0]

def write_nms_json():
    #if not os.path.exists("./NMS"):
    #    os.mkdir("./NMS")
    #os.chdir("./NMS")
    pred_file=[line.rstrip('\n').rstrip(' ') for line in open("pred.txt")]
    score_file=[line.rstrip('\n').rstrip(' ') for line in open("scores.txt")]
#    index_file=[line.rstrip('\n').rstrip(' ') for line in open("index.txt")]
    imgId_file = [line.rstrip('\n').rstrip(' ') for line in open("imgId_file.txt")]
    proposal_scores = np.loadtxt("scores-proposals.txt")
    
    results = {}
    for i in xrange(len(pred_file)):
        xmin = 2000
        ymin = 2000
        xmax = 0
        ymax = 0
        if imgId_file[i] not in results:
            results[imgId_file[i]] = []
        
        keypoints = []
        score=[]
        pred_coordi = pred_file[i].split('\t')
        pred_score = score_file[i].split('\t')
        for n in xrange(14):
            keypoints.append(int(pred_coordi[2*n+1])); 
            keypoints.append(int(pred_coordi[2*n+2]));
            xmin = min(xmin,int(pred_coordi[2*n+1]));
            ymin = min(ymin,int(pred_coordi[2*n+2]));
            xmax = max(xmax,int(pred_coordi[2*n+1]));
            ymax = max(ymax,int(pred_coordi[2*n+2]));
            keypoints.append(float(pred_score[n+1]));
            score.append(float(pred_score[n+1]))
        if (1.5*(xmax-xmin)*1.5*(ymax-ymin) < 40.5*40):
            continue
        results[imgId_file[i]].append(keypoints)
        #result['score'] = 1.0*np.mean(score)+ 0.5* proposal_scores[i] +1.25*np.max(score)
#        result['score'] = proposal_scores[i]
        #results.append(result)
    final_res = []
    for imgId in results.keys():
        res = {}
        res['image_id'] = imgId
        res['keypoint_annotations'] = {}
        for i in range(len(results[imgId])):
            res['keypoint_annotations']['human%d'%(i+1)] = results[imgId][i]
        final_res.append(res)
    cnt=0
    aic=[]
    for i in results.keys():
        dict={}
        dict['image_id']=i
        dict['keypoint_annotations']={}
        for k,j in enumerate(results[i]):
            dict['keypoint_annotations']['human%d'%k]=j
            cnt=cnt+1
        aic.append(dict)
    print(cnt)
    results=aic
    with open("person_keypoints_minival_SJTU-pose_results_forMerge.json",'w') as json_file:
        json_file.write(json.dumps(results))
    Result = zipfile.ZipFile("person_keypoints_minival_SJTU-pose_results_forMerge.zip",'w')
    Result.write("person_keypoints_minival_SJTU-pose_results_forMerge.json") 
    Result.close()

def remove_head_json():
    dataDir="/home/benjamin/HardDisk/coco"
    annFile='%s/annotations/person_keypoints_val5k2014_without_head.json'%(dataDir)
    result_File='%s/annotations/person_keypoints_val5k2014_no_head.json'%(dataDir)
    with open(annFile) as json_file:
        data = json.load(json_file)
    ann = data['annotations']
    for i in xrange(len(ann)):
        if (ann[i]['category_id'] == 1):
            for j in xrange(5):
                ann[i]['keypoints'][3*j+2] = 0
    with open(result_File,'w') as result_json:
        result_json.write(json.dumps(data))

def check_remove():
    result_File = "/home/benjamin/HardDisk/coco/annotations/person_keypoints_val5k2014_no_head.json"
    with open(result_File) as json_file:
        data = json.load(json_file)
    ann = data['annotations']
    for i in xrange(len(ann)):
        if (ann[i]['category_id'] == 1):
            for j in xrange(5):
                if (ann[i]['keypoints'][3*j+2] != 0):
                    print "Shit"

def test_parametric_pose_NMS_json(delta1,delta2,mu,gamma):
    scoreThreds = 4.95

    matchThreds = 1

    #prepare data
    h5file = h5py.File("../annot/aic-val-minival/test-dev0.1.h5",'r')
    preds = np.array(h5file['preds'])
    scores = np.array(h5file['scores'])
    scores[scores==0] = 1e-5
    #print preds.shape, scores.shape
    #for i in xrange(17):
    #   preds[:,i,0] = scores[:,i,0]/(scores[:,i,0]+scores2[:,i,0])*preds[:,i,0] + scores2[:,i,0]/(scores[:,i,0]+scores2[:,i,0])*preds2[:,i,0]
    #   preds[:,i,1] = scores[:,i,0]/(scores[:,i,0]+scores2[:,i,0])*preds[:,i,1] + scores2[:,i,0]/(scores[:,i,0]+scores2[:,i,0])*preds2[:,i,1]
    #   scores[:,i,0] = scores[:,i,0]/(scores[:,i,0]+scores2[:,i,0])*scores[:,i,0] + scores2[:,i,0]/(scores[:,i,0]+scores2[:,i,0])*scores2[:,i,0]
    indexs = [line.rstrip(' ').rstrip('\r').rstrip('\n') for line in open("../annot/aic-val-minival/index.txt")]
    imgIDs = np.array([(line[0:-5]) for line in open("../annot/aic-val-minival/test-dev_images.txt")])
    #np.loadtxt(,delimiter = '\n',dtype=str)
    #print imgIDs[0]
    scores_proposals = np.loadtxt("../annot/aic-val-minival/score-proposals.txt")
    
    
    #get bounding box sizes    
    bbox_file = h5py.File("../annot/aic-val-minival/test-dev.h5",'r')
    xmax=np.array(bbox_file['xmax']); xmin=bbox_file['xmin']; ymax=np.array(bbox_file['ymax']); ymin=bbox_file['ymin']
    widths=xmax-xmin; heights=ymax-ymin;
    alpha = 0.1
    Sizes=alpha*np.maximum(widths,heights)
    
    #set the corresponding dir
    if (os.path.exists("./aic-val-minival/") == False):
        os.mkdir("./aic-val-minival/")
#    if (os.path.exists("/home/benjamin/HardDisk/data/MPII_bbox_info/test/parametric-NMS/delta1{}".format(delta1)) == False):
#        os.mkdir("/home/benjamin/HardDisk/data/MPII_bbox_info/test/parametric-NMS/delta1{}".format(delta1))
    os.chdir("./aic-val-minival/")
    
    NMS_preds = open("pred.txt",'w')
    NMS_scores = open("scores.txt",'w')
    proposal_scores = open("scores-proposals.txt",'w')
    NMS_index = open("test.txt",'w')
    NMS_ids = open("imgId_file.txt",'w')
    num_human = 0
    
    #loop through every image

    for i in xrange(len(indexs)):
#    for i in [2,119]:
        index = indexs[i].split(' '); 
        img_name = index[0]; start = int(index[1])-1; end = int(index[2])-1;
        
        #initialize scores and preds coordinates
        img_preds = preds[start:end+1]; img_scores = np.mean(scores[start:end+1],axis = 1)
        #print np.mean(scores[start:end+1],axis = 1).shape,np.array(scores_proposals[start:end+1,np.newaxis]).shape
        img_ids = np.arange(end-start+1); ref_dists = Sizes[start:end+1];keypoint_scores = scores[start:end+1];
        Ids = imgIDs[start:end+1]; 
        pro_score = np.array(scores_proposals[start:end+1,np.newaxis])
        
        #do NMS by parametric
        pick = []
        merge_ids = []
        while(img_scores.size != 0):
            
            #pick the one with highest score
            pick_id = np.argmax(img_scores)  
            pick.append(img_ids[pick_id])
            
            #get numbers of match keypoints by calling PCK_match 
            #print ref_dists
            ref_dist=ref_dists[img_ids[pick_id]]
            #print ref_dist
            simi = get_parametric_distance(pick_id,img_preds, keypoint_scores,ref_dist, delta1, delta2, mu)
            num_match_keypoints,face_match_keypoints = PCK_match(img_preds[pick_id],img_preds,ref_dist)

            #delete humans who have more than matchThreds keypoints overlap with the seletced human.
            delete_ids = np.arange(img_scores.shape[0])[(simi > gamma) | (num_match_keypoints >= matchThreds)]
            if (delete_ids.size == 0):
                delete_ids = pick_id
            merge_ids.append(img_ids[delete_ids])
            img_preds = np.delete(img_preds,delete_ids,axis=0); img_scores = np.delete(img_scores, delete_ids); pro_score = np.delete(pro_score,delete_ids);
            img_ids = np.delete(img_ids, delete_ids); keypoint_scores = np.delete(keypoint_scores,delete_ids,axis=0)

        #write the NMS result to files
        pick = [Id+start for Id in pick] 
        merge_ids = [Id+start for Id in merge_ids]
        assert len(merge_ids) == len(pick)
        preds_pick = preds[pick]; scores_pick = scores[pick]; pick_ids = imgIDs[pick]
        num_pick = 0
        for j in xrange(len(pick)):
            
            #first compute the average score of a person
            #filtering out if the human has no ankles
            ids = np.arange(14)
            #if (scores_pick[j,16,0] < 0): ids = np.delete(ids,16);
            #if (scores_pick[j,15,0] < 0): ids = np.delete(ids,15);
            max_score = np.max(scores_pick[j,ids,0])
       # print mean_score
            if (max_score < scoreThreds):
                continue
            
            
            # merge poses
            merge_id = merge_ids[j]  
            pick_id = pick_ids[j]
            score = scores_proposals[pick[j]]
            merge_poses,merge_score = merge_pose(preds_pick[j],preds[merge_id],scores[merge_id],Sizes[pick[j]])
            
            ids = np.arange(14)
            #if (merge_score[16] < 0.1): ids = np.delete(ids,16);
            #if (merge_score[15] < 0.1): ids = np.delete(ids,15);
            max_score = np.max(merge_score[ids])
            if (max_score < scoreThreds):
                continue
            
            #add the person to predict
            num_pick += 1
            NMS_preds.write("{}".format(img_name))
            NMS_scores.write("{}".format(img_name))
            NMS_ids.write("{}\n".format((pick_id)))
            proposal_scores.write("{}\n".format(score))
            
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
    ref_dist=min(ref_dist,15)
    # mask is an nx17 matrix
    mask = (dist <= ref_dist)
    final_pose = np.zeros([14,2]); final_scores = np.zeros(14)
    if (cluster_preds.ndim == 2):
        cluster_preds = cluster_preds[np.newaxis,:,:]
        cluster_keypoint_scores = cluster_keypoint_scores[np.newaxis,:]
    if (mask.ndim == 1):
        mask = mask[np.newaxis,:]
    for i in xrange(14):
#        print "cluster_keypoint_scores.shape:", cluster_keypoint_scores.shape
#        print "mask.shape:",mask.shape
        cluster_joint_scores = cluster_keypoint_scores[:,i][mask[:,i]]
        
        # pick the corresponding i's matched keyjoint locations and do an weighed sum.
#        print cluster_preds.shape
#        print cluster_preds[:,i,:]
        cluster_joint_location = cluster_preds[:,i,:][np.tile(mask[:,i,np.newaxis],(1,2))].reshape(np.sum(mask[:,i,np.newaxis]),-1)

        # get an normalized score
        normed_scores = cluster_joint_scores / np.sum(cluster_joint_scores)
        # merge poses by a weighted sum
        final_pose[i,0] = np.dot(cluster_joint_location[:,0], normed_scores)
        final_pose[i,1] = np.dot(cluster_joint_location[:,1], normed_scores)
#        final_scores[i] = np.average(cluster_joint_scores)
        final_scores[i] = np.dot(cluster_joint_scores.T, normed_scores)
        #final_scores[i] = np.max(cluster_joint_scores)
    return final_pose, final_scores
    
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

def PCK_match_ratio(pick_preds, all_preds, keypoint_scores, ref_dist):
    dist = np.sqrt(np.sum(np.square(pick_preds[np.newaxis,:]-all_preds),axis=2))
    keypoint_scores = np.sum(keypoint_scores,axis=2)
    ref_dist=min(ref_dist,7)
    num_match_keypoints_ratio = np.sum((dist/ref_dist <= 1) & (keypoint_scores>=0.2),axis=1)/(np.sum((keypoint_scores>=0.2),axis=1)+1)
    return num_match_keypoints_ratio

def score_PCK_pose_NMS_json(matchThreds,scoreThreds):
    
    #prepare data
    h5file = h5py.File("/home/benjamin/HardDisk/coco/valid-500/test-try1.h5",'r')
    preds = np.array(h5file['preds'])
    scores = np.array(h5file['scores'])    
    indexs = [line.rstrip(' ').rstrip('\r').rstrip('\n') for line in open("/home/benjamin/HardDisk/coco/valid-500/index.txt")]
    imgIDs = np.loadtxt("/home/benjamin/HardDisk/coco/valid-500/id.txt")   
    
    #get bounding box sizes    
    bbox_file = h5py.File("/home/benjamin/HardDisk/coco/valid-500/test.h5",'r')
    xmax=np.array(bbox_file['xmax']); xmin=bbox_file['xmin']; ymax=np.array(bbox_file['ymax']); ymin=bbox_file['ymin']
    widths=xmax-xmin; heights=ymax-ymin;
    alpha = 0.1
    Sizes=alpha*np.maximum(widths,heights)
    
    
    if (os.path.exists("/home/benjamin/HardDisk/coco/valid-500/PCK") == False):
        os.mkdir("/home/benjamin/HardDisk/coco/valid-500/PCK")
    os.chdir("/home/benjamin/HardDisk/coco/valid-500/PCK")
    NMS_preds = open("pred.txt",'w')
    NMS_scores = open("scores.txt",'w')
    NMS_index = open("index.txt",'w')
    NMS_ids = open("imgId_file.txt",'w')
    num_human = 0
    
    #loop through images index
    for i in xrange(len(indexs)):
#    for i in [21]:
        index = indexs[i].split(' '); 
        img_name = index[0]; start = int(index[1])-1; end = int(index[2])-1;
        
        #initialize scores and preds coordinates
        img_preds = preds[start:end+1]; img_scores = np.mean(scores[start:end+1],axis = 1)
        img_ids = np.arange(end-start+1); ref_dists = Sizes[start:end+1]
        
        #do NMS by PCK
        pick = []
        while(img_scores.size != 0):
            
            #pick the one with highest score
            pick_id = np.argmax(img_scores)  
            pick.append(img_ids[pick_id])
            
            #get numbers of match keypoints by calling PCK_match 
            ref_dist=ref_dists[img_ids[pick_id]]
            num_match_points = PCK_match(img_preds[pick_id],img_preds,ref_dist)
            
            #delete humans who have more than matchThreds keypoints overlap with the seletced human.
            delete_ids = np.arange(img_scores.shape[0])[num_match_points > matchThreds]
            img_preds = np.delete(img_preds,delete_ids,axis=0); img_scores = np.delete(img_scores, delete_ids)
            img_ids = np.delete(img_ids, delete_ids); 
        
        #write the NMS result to files
        pick = [Id+start for Id in pick]        
        preds_pick = preds[pick]; scores_pick = scores[pick];pick_ids = imgIDs[pick]
        num_pick = 0
        for j in xrange(len(pick)):
            
            #first compute the average score of a person
            #filtering out if the human has no ankles
            ids = np.arange(14)
            if (scores_pick[j,8,0] < 0.1): ids = np.delete(ids,0);
            if (scores_pick[j,11,0] < 0.1): ids = np.delete(ids,5);
            mean_score = np.mean(scores_pick[j,ids,0])
            if (mean_score < scoreThreds):
                continue
            
            #add the person to predict
            num_pick += 1
            pick_id = pick_ids[j]
            NMS_preds.write("{}".format(img_name))
            NMS_scores.write("{}".format(img_name))
            NMS_ids.write("{}\n".format(int(pick_id)))
            for point_id in xrange(14):
                NMS_preds.write("\t{}\t{}".format(int(preds_pick[j,point_id,0]),int(preds_pick[j,point_id,1])))
                NMS_scores.write("\t{}".format(scores_pick[j,point_id,0]))
            NMS_preds.write("\n")
            NMS_scores.write("\n")
        NMS_index.write("{} {} {}\n".format(img_name, num_human+1, num_human + num_pick))
        num_human += num_pick
        
    NMS_preds.close();NMS_scores.close();NMS_index.close(); NMS_ids.close()


def write_bbox():
    os.chdir("/home/benjamin/HardDisk/coco/val5k/valid-5k/")
    with h5py.File("valid5k.h5") as bbox_file:
        xmin = np.array(bbox_file['xmin'])
        ymin = np.array(bbox_file['ymin'])
        xmax = np.array(bbox_file['xmax'])
        ymax = np.array(bbox_file['ymax'])
    index_file = [line.rstrip(' ').rstrip('\r').rstrip('\n') for line in open("index.txt")]
    id_file = np.loadtxt("id.txt")
    score_file = np.loadtxt("score-proposals.txt")
    results = []
    for i in xrange(len(index_file)):
        image = index_file[i].split(' ')
        start = int(image[1])-1; end = int(image[2])-1
        img_id = id_file[start]
        for j in xrange(end-start+1):
            result = {}
            result['image_id'] = int(img_id)
            result['category_id'] = 1
            result['bbox'] = [xmin[start+j], ymin[start+j], xmax[start+j]-xmin[start+j], ymax[start+j]-ymin[start+j]]
            result['score'] = score_file[start+j]
            results.append(result)
    with open("detections_val5k2014_sjtu_eval.json",'w') as json_file:
        json_file.write(json.dumps(results))

def write_val5k_gt_json():
    if not os.path.exists("/home/benjamin/HardDisk/coco/val5k/valid-5k-gtbox/"):
        os.mkdir("/home/benjamin/HardDisk/coco/val5k/valid-5k-gtbox/")
    os.chdir("/home/benjamin/HardDisk/coco/val5k/valid-5k-gtbox/")
    h5_file = h5py.File("val5k-basedgt.h5",'r')
    pred_file = h5_file['preds']
    score_file = h5_file['scores']
#    index_file=[line.rstrip('\n').rstrip(' ') for line in open("index.txt")]
    imgId_file = [line.rstrip('\n').rstrip(' ') for line in open("val5k2014-gt_id.txt")]
#    proposal_scores = np.loadtxt("scores-proposals.txt")
    
    results = []
    for i in xrange(pred_file.shape[0]):
        result = {}
        result['image_id'] = int(imgId_file[i])
        result['category_id'] = 1;
        keypoints = []
        score = []
        pred_coordi = pred_file[i]
        pred_score = score_file[i]
        for n in xrange(14):
#            keypoints.append(int(pred_coordi[2*n+1])); 
#            keypoints.append(int(pred_coordi[2*n+2]));
            keypoints.append(int(pred_coordi[n,0]))
            keypoints.append(int(pred_coordi[n,1]))
            keypoints.append(1);
            score.append(float(pred_score[n]))
        result['keypoints'] = keypoints
        result['score'] = np.mean(score)
#        result['score'] = proposal_scores[i]
        results.append(result)
    with open("person_keypoints_val5k2014_SJTU-pose_results.json",'w') as json_file:
        json_file.write(json.dumps(results))
    Result = zipfile.ZipFile("person_keypoints_val5k2014_SJTU-pose_results.zip",'w')
    Result.write("person_keypoints_val5k2014_SJTU-pose_results.json")
    Result.close()

def get_result_json():
    delta1 = 1; mu = 2.5; delta2 = 2.5;
    gamma = 24; nms = 0.4; data = 'test'
    test_parametric_pose_NMS_json(delta1, delta2, mu, gamma, nms, data)
    write_nms_json()
  
#write_val5k_gt_json()  
#write_bbox()
#delta1 = 0.01; mu = 2.08; delta2 = 2.08;
#gamma = 22.48;
#test_parametric_pose_NMS_json(delta1, delta2, mu, gamma)
###score_PCK_pose_NMS_json(5,0.0)
#write_nms_json()
#test_json()
get_result_json()
