
python3 ./object_detection/model_main.py \
  --run_once=True \
  --checkpoint_dir=./object_detection/coco_data/ssd_inception_v2_coco_2018_01_28 \
  --pipeline_config_path=./object_detection/coco_data/ssd_inception_v2_coco_2018_01_28/pipeline_for_eval.config
