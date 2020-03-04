#
# download http://images.cocodataset.org/zips/val2014.zip
# download http://images.cocodataset.org/annotations/annotations_trainval2014.zip
#

python3 object_detection/dataset_tools/create_coco_tf_record_eval.py \
  --label_map_path=./object_detection/coco_data/mscoco_label_map.pbtxt \
  --val_image_dir=/data/coco/val2014 \
  --val_annotations_file=/data/coco/annotations/instances_val2014.json \
  --output_dir=./object_detection/coco_data/
