#
# download http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz
# https://github.com/tensorflow/models/pull/8057/commits/4563c282d3d664853eae3e99b6fd3453aacc39b0
#

python3 ./object_detection/model_main.py \
  --run_once=True \
  --checkpoint_dir=./object_detection/coco_data/ssd_mobilenet_v3_large_coco_2020_01_14 \
  --pipeline_config_path=./object_detection/coco_data/ssd_mobilenet_v3_large_coco_2020_01_14/pipeline_for_eval.config \
  2>&1 | tee ./coco_data/ssd_mobilenet_v3_large_coco_2020_01_14/evaluation.log
