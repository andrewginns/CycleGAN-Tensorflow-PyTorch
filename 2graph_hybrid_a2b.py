from __future__ import absolute_import, division, print_function

import argparse
from glob import glob
import os

import image_utils as im
import utils
import tensorflow as tf
from tensorflow.python.platform import gfile
import time
import numpy as np

""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--graph1', dest='graph_file1', default='C:/Users/ag17634/Desktop/optimized-graph.pb',
                    help='path of first graph to use')
parser.add_argument('--graph1', dest='graph_file2', default='C:/Users/ag17634/Desktop/optimized-graph.pb',
                    help='path of second graph to use')
parser.add_argument('--dataset', dest='dataset', default='C:/Users/ag17634/Desktop/test',
                    help='path of images to process')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=600, help='then crop to this size')
args = parser.parse_args()

graph_file1 = args.graph_file1
graph_file2 = args.graph_file2

dataset = args.dataset
crop_size = args.crop_size

""" Restore the graph """
with tf.Graph().as_default() as graph1:  # Set default graph as graph

    with tf.Session() as sess:

        # Parse the protobuff file to obtain an unserialized GraphDef
        with gfile.FastGFile(graph_file1, 'rb') as f:

            print("\nLoading Images...\n")
            print("Images will be cropped to:", (crop_size, crop_size))
            a_list = glob(dataset + '/*.jpg')

            print("Creating destination")
            a_save_dir = dataset + '/inference_results'
            utils.mkdir([a_save_dir])

            # Set the graph as the default graph
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            # Import the GraphDef as the current default graph
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="",
                op_dict=None,
                producer_op_list=None
            )
            print("\nGraph successfully loaded")

            # Print the name of the ops and their attributes
            for op in graph1.get_operations():
                print("\nOperation name :", op.name)  # Operation name
                print("Tensor details :", str(op.values()))  # Tensor name

            # Assign input and output tensors
            a_input = graph1.get_tensor_by_name('inputA:0')  # Input Tensor
            a_output = graph1.get_tensor_by_name('a2b_generator/Conv_7/Relu:0')  # Output Tensor

            # Initialize_all_variables
            tf.global_variables_initializer()

            start = time.time()
            # Inference
            for i in range(len(a_list)):
                # Define shapes for images fed to the graph
                a_feed = im.imresize(im.imread(a_list[i]), [crop_size, crop_size])
                a_feed.shape = 1, crop_size, crop_size, 3

                # Feed in images to the graph
                a2b_result = sess.run(a_output, feed_dict={a_input: a_feed})
                print(type(a2b_result))
                print(a2b_result.shape)

                # Create and save the output image
                output = tf
                # a_img_opt = a2b_result
                # img_name = os.path.basename(a_list[i])
                #
                # output = im.immerge(a_img_opt, 1, 1)
                #
                # im.imwriteShow(output, a_save_dir + '/' + img_name)
                #
                # print('Save %s' % (a_save_dir + '/' + img_name))

                if i == 100:
                    end = time.time()
            end2 = time.time()

            # print("Time to process first 100 images:", end - start)
            print("Time to process all %d images: %f" % (i + 1, end2 - start))

""" Restore the graph """
with tf.Graph().as_default() as graph2:  # Set default graph as graph

    with tf.Session() as sess:

        # Parse the protobuff file to obtain an unserialized graph_drf
        with gfile.FastGFile(graph_file2, 'rb') as f:

            print("\nLoading Images...\n")
            print("Images will be cropped to:", (crop_size, crop_size))
            a_list = glob(dataset + '/*.jpg')

            print("Creating destination")
            a_save_dir = dataset + '/inference_results'
            utils.mkdir([a_save_dir])

            # Set the graph as the default graph
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            # Import the graph_def as the current default graph
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="",
                op_dict=None,
                producer_op_list=None
            )
            print("\nGraph successfully loaded")

            # Print the name of the ops and their attributes
            for op in graph.get_operations():
                print("\nOperation name :", op.name)  # Operation name
                print("Tensor details :", str(op.values()))  # Tensor name

            # Assign input and output tensors
            a_input = graph.get_tensor_by_name('a2b_generator/Conv_7/Relu:0')  # Input Tensor
            a_output = graph.get_tensor_by_name('a2b_generator/output_image')  # Output Tensor

            # Initialize_all_variables
            tf.global_variables_initializer()

            start = time.time()
            # Inference
            for i in range(len(a_list)):
                # Define shapes for images fed to the graph
                inputShape = (150, 150, 256)
                inputTensor = np.zeros(inputShape)
                a_feed = inputTensor

                # Feed in images to the graph
                a2b_result = sess.run(a_output, feed_dict={a_input: a_feed})
                print(type(a2b_result))
                print(a2b_result.shape)

                # Create and save the output image
                a_img_opt = a2b_result
                img_name = os.path.basename(a_list[i])

                output = im.immerge(a_img_opt, 1, 1)

                im.imwriteShow(output, a_save_dir + '/' + img_name)

                print('Save %s' % (a_save_dir + '/' + img_name))

                if i == 100:
                    end = time.time()
            end2 = time.time()

            # print("Time to process first 100 images:", end - start)
            print("Time to process all %d images: %f" % (i + 1, end2 - start))
