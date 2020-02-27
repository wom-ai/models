#!/bin/sh

MODEL_NAME=ssdlite_mobilenet_v2_ball_2020_02_26
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=./object_detection/ball_data/ssdlite_mobilenet_v2_ball.config
TRAINED_CKPT_DIR_PATH=./object_detection/ball_data/model_dir_${MODEL_NAME}

# find the latest mkpt file and get step number
#CKPT_FILE=$(ls -t ${TRAINED_CKPT_DIR_PATH}/model.ckpt-* | head -1 | awk -F"[.-]" "{ print $3}")
CKPT_FILE_PATH=$(ls -t ${TRAINED_CKPT_DIR_PATH}/model.ckpt-* | head -1)
CKPT_FILE_NAME=$(basename ${CKPT_FILE_PATH})
CKPT_STEP=$(echo "${CKPT_FILE_NAME}" | awk -F"[.-]" '{ print $3 }')
TRAINED_CKPT_PREFIX=${TRAINED_CKPT_DIR_PATH}/model.ckpt-${CKPT_STEP}

EXPORT_DIR=./object_detection/ball_data/frozen_model_${MODEL_NAME}

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "MODEL_NAME=${MODEL_NAME}"
echo "TRAINED_CKPT_DIR_PATH=${TRAINED_CKPT_DIR_PATH}"
echo "CKPT_FILE_PATH=${CKPT_FILE_PATH}"
echo "CKPT_FILE_NAME=${CKPT_FILE_NAME}"
echo "latest ckpt step is $CKPT_STEP"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"


if [ -d ${EXPORT_DIR} ]
then
  rm -v ${EXPORT_DIR}/* -rf
fi

python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR} \
    --input_shape="1,-1,-1,3"

cd ./object_detection/ball_data/ && \
tar czvf frozen_model_${MODEL_NAME}.tar.gz ./frozen_model_${MODEL_NAME} && \
cp -v frozen_model_${MODEL_NAME}.tar.gz /data/
