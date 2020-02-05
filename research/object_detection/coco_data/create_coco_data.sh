python3 object_detection/dataset_tools/create_coco_tf_record.py \
  --label_map_path=./object_detection/coco_data/mscoco_label_map.pbtxt \
  --train_image_dir=/data/coco/train2017 \
  --val_image_dir=/data/coco/val2017 \
  --test_image_dir=/data/coco/val2017 \
  --train_annotations_file=/data/coco/annotations/instances_train2017.json \
  --val_annotations_file=/data/coco/annotations/instances_val2017.json \
  --testdev_annotations_file=/data/coco/annotations/instances_val2017.json \
  --output_dir=./object_detection/coco_data/
