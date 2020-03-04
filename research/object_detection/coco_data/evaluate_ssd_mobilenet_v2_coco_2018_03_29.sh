#
# References
#  - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/oid_inference_and_evaluation.md
#

#SPLIT=validation  # or test
#TF_RECORD_FILES=$(ls -1 ${SPLIT}_tfrecords/* | tr '\n' ',')
#TF_RECORD_FILES=$(ls -1 ./object_detection/coco_data/coco_val.* | tr '\n' ',')
#TF_RECORD_FILES=coco_val.record-00000-of-00010

#python3 ./object_detection/inference/infer_detections.py \
#    --input_tfrecord_paths=$TF_RECORD_FILES \
#    --output_tfrecord_path=./object_detection/coco_data/coco_detection.record-00000-of-00010 \
#    --inference_graph=./object_detection/coco_data/ssd_mobilenet_v3_large_coco_2019_08_14/frozen_inference_graph.pb \
#    --discard_image_pixels
#

#
# References
#  - https://www.gitmemory.com/issue/tensorflow/models/6636/505741335
#

#python3 ./object_detection/legacy/eval.py  \
#  --checkpoint_dir=./object_detection/coco_data/ssd_mobilenet_v2_coco_2018_03_29/  \
#  --eval_dir=./object_detection/coco_data/output_eval_ssd_mobilenet_v2_coco_2018_03_29/ \
#  --pipeline_config_path=./object_detection/coco_data/ssd_mobilenet_v2_coco_2018_03_29/pipeline_for_eval.config
#

python3 ./object_detection/model_main.py \
  --run_once=True \
  --checkpoint_dir=./object_detection/coco_data/ssd_mobilenet_v2_coco_2018_03_29 \
  --pipeline_config_path=./object_detection/coco_data/ssd_mobilenet_v2_coco_2018_03_29/pipeline_for_eval.config
  #--eval_training_data=True \
  #--sample_1_of_n_eval_on_train_examples=1 \
