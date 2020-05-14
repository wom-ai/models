
CHECKPOINT_DIR=$1
echo "########################################################################"
echo " Evaluation"
echo "  - checkpoint_dir=${CHECKPOINT_DIR}"
echo "########################################################################"
start=`date +%s`

python3 ./object_detection/model_main.py \
  --run_once=True \
  --checkpoint_dir=${CHECKPOINT_DIR} \
  --pipeline_config_path=${CHECKPOINT_DIR}/pipeline_for_eval.config \
  2>&1 | tee ${CHECKPOINT_DIR}/evaluation.log
sleep 30
end=`date +%s`
runtime=$((end-start))
echo "Elapsed Time: $runtime seconds"
echo "Elapsed Time: $runtime seconds" >> ${CHECKPOINT_DIR}/evaluation.log
