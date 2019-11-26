INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=./object_detection/ball_data/ssdlite_mobilenet_v2_ball_1_anchor_layer.config
TRAINED_CKPT_PREFIX=./object_detection/ball_data/model_dir/model.ckpt-200000
#TRAINED_CKPT_PREFIX=./object_detection/ball_data/model_dir/model.ckpt-1000000
EXPORT_DIR=./object_detection/ball_data/frozen_model
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR} \
    --input_shape="1,-1,-1,3"
