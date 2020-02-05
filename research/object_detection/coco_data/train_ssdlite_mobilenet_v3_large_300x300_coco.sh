# run this on tensorflow/models/research
python3 ./object_detection/model_main.py \
  --model_dir=./object_detection/coco_data/model_dir_ssdlite_mobilenet_v3_large_300x300_coco \
  --pipeline_config_path=./object_detection/coco_data/ssdlite_mobilenet_v3_large_300x300_coco.config
