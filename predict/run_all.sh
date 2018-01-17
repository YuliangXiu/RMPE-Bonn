#python aicha-human_detection.py
#cd annot
#ln -s ~/code/RMPE-poseflow/examples/ai-cha/aicha-test-0.6
#cd ..
#th main_aicha.lua predict-test
python aicha_batch_nms.py 
cd NMS
python convert_txt_json.py 
python keypoint_eval.py --submit submit-valid-0.4.json --ref keypoint_validation_annotations_20170911.json



