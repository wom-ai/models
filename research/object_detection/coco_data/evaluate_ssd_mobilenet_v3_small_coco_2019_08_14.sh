
python3 ./object_detection/model_main.py \
  --run_once=True \
  --checkpoint_dir=./object_detection/coco_data/ssd_mobilenet_v3_small_coco_2019_08_14 \
  --pipeline_config_path=./object_detection/coco_data/ssd_mobilenet_v3_small_coco_2019_08_14/pipeline_for_eval.config \
  2>&1 | tee ./coco_data/ssd_mobilenet_v3_small_coco_2019_08_14/evaluation.log
