#!/usr/bin/env python
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

use_gpu = True
if not use_gpu:
  os.environ['CUDA_VISIBLE_DEVICES'] = ''

# ## Env setup

# In[2]:


# This is needed to display the images.
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[17]:


# Lot's of paths to configure here!

# model_to_test = 'ssdlite_shufflenet_v2_pet' # 11 ms
model_to_test = 'ssdlite_mobilenet_v2_pet' # 7 ms
# model_to_test = 'ssdlite_mobilenet_v2_pet__withdeviceinfo' # 7 ms + device info = 9 ms
# model_to_test = 'ssdlite_mobilenet_v2_coco__model_zoo'

if model_to_test == 'ssdlite_shufflenet_v2_pet':
    #PATH_TO_FROZEN_GRAPH = '/temp/pet/saved-model-out/' + '/frozen_inference_graph.pb'
    PATH_TO_FROZEN_GRAPH = '/temp/pet/saved-model-out2/' + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('data', 'pet_label_map.pbtxt')
elif model_to_test == 'ssdlite_mobilenet_v2_pet':
    PATH_TO_FROZEN_GRAPH = '/temp/bogusnet/pet_data/frozen_model/' + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('data', 'pet_label_map.pbtxt')
elif model_to_test == 'ssdlite_mobilenet_v2_pet__withdeviceinfo':
    PATH_TO_FROZEN_GRAPH = '/temp/bogusnet/pet_data/frozen_model_withdeviceinfo/' + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('data', 'pet_label_map.pbtxt')
elif model_to_test == 'ssdlite_mobilenet_v2_coco__model_zoo':
    # What model to download.
    #MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


# ## Download Model

# In[ ]:


###   opener = urllib.request.URLopener()
###   opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
###   tar_file = tarfile.open(MODEL_FILE)
###   for file in tar_file.getmembers():
###     file_name = os.path.basename(file.name)
###     if 'frozen_inference_graph.pb' in file_name:
###       tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[18]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
    # print('=' * 40)
    # print('=' * 40)
    # print('is: od_graph_def')

    # for node in od_graph_def.node:
    #   print(node.name)

    # print('is: od_graph_def')
    # print('=' * 40)
    # print('=' * 40)

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[19]:


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# ## Helper code

# In[7]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection (inaccurate, see below)

# In[8]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 9) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

print(TEST_IMAGE_PATHS)


# In[ ]:


###   def run_inference_for_single_image(image, graph):
###     with graph.as_default():
###       with tf.Session() as sess:
###         # Get handles to input and output tensors
###         ops = tf.get_default_graph().get_operations()
###         all_tensor_names = {output.name for op in ops for output in op.outputs}
###         tensor_dict = {}
###         for key in [
###             'num_detections', 'detection_boxes', 'detection_scores',
###             'detection_classes', 'detection_masks'
###         ]:
###           tensor_name = key + ':0'
###           if tensor_name in all_tensor_names:
###             tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
###                 tensor_name)
###         if 'detection_masks' in tensor_dict:
###           # The following processing is only for single image
###           detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
###           detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
###           # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
###           real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
###           detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
###           detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
###           detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
###               detection_masks, detection_boxes, image.shape[1], image.shape[2])
###           detection_masks_reframed = tf.cast(
###               tf.greater(detection_masks_reframed, 0.5), tf.uint8)
###           # Follow the convention by adding back the batch dimension
###           tensor_dict['detection_masks'] = tf.expand_dims(
###               detection_masks_reframed, 0)
###         image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
###   
###         # Run inference
###         import time
###         start = time.process_time()
###         output_dict = sess.run(tensor_dict,
###                                feed_dict={image_tensor: image})
###         print(time.process_time() - start, 'sec')
###   
###         # all outputs are float32 numpy arrays, so convert types as appropriate
###         output_dict['num_detections'] = int(output_dict['num_detections'][0])
###         output_dict['detection_classes'] = output_dict[
###             'detection_classes'][0].astype(np.int64)
###         output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
###         output_dict['detection_scores'] = output_dict['detection_scores'][0]
###         if 'detection_masks' in output_dict:
###           output_dict['detection_masks'] = output_dict['detection_masks'][0]
###     return output_dict


# In[ ]:


###   for image_path in TEST_IMAGE_PATHS:
###     image = Image.open(image_path)
###     # the array based representation of the image will be used later in order to prepare the
###     # result image with boxes and labels on it.
###     image_np = load_image_into_numpy_array(image)
###     # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
###     image_np_expanded = np.expand_dims(image_np, axis=0)
###     # Actual detection.
###     get_ipython().run_line_magic('timeit', 'output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)')
###     for cls, scr in zip(output_dict['detection_classes'], output_dict['detection_scores']):
###       print(category_index[cls], '-->', scr)
###     # Visualization of the results of a detection.
###     vis_util.visualize_boxes_and_labels_on_image_array(
###         image_np,
###         output_dict['detection_boxes'],
###         output_dict['detection_classes'],
###         output_dict['detection_scores'],
###         category_index,
###         instance_masks=output_dict.get('detection_masks'),
###         use_normalized_coordinates=True,
###         line_thickness=8)
###     plt.figure(figsize=IMAGE_SIZE)
###   plt.imshow(image_np)


# # Detection (better)

# Copied from: [https://github.com/tensorflow/models/issues/3270](https://github.com/tensorflow/models/issues/3270)

# In[20]:


import time
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represents confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    durations = []
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
#       image = image.resize((300, 300))
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)

#      options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#      run_metadata = tf.RunMetadata()
      # Actual detection.
      start = time.time()
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded},
#          options=options,
#          run_metadata=run_metadata
          )
      duration = time.time() - start
      durations.append(duration)
      print("Duration: {0:.2f}ms".format(duration*1000))
    #print('\n'.join([x.name for x in sess.graph.get_operations()]))

# Calculate distribution
durations = durations[1:]
avg = np.average(durations)
std = np.std(durations)
print('Average: %.3fms +/- %.3fms' % (avg*1000, std*1000))
print('(Results for model: %s)' % model_to_test)


# Output Chrome trace
try:
    run_metadata
except:
    pass
else:
    from tensorflow.python.client import timeline
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()

    with open('Experiment_mobilenet_' + ('gpu' if use_gpu else 'cpu') + '.json', 'w') as f:
        f.write(chrome_trace)
        print('Chrome trace generated.')

