INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=./object_detection/coco_data/ssdlite_mobilenet_v3_large_300x300_coco.config
TRAINED_CKPT_PREFIX=./object_detection/coco_data/model_dir_ssdlite_mobilenet_v3_large_300x300_coco/model.ckpt-16497
EXPORT_DIR=./object_detection/coco_data/frozen_model_ssdlite_mobilenet_v3_large_300x300_coco
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR} \
    --input_shape="1,-1,-1,3"
