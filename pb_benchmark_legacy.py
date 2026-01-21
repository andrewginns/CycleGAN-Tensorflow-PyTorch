#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import argparse
import time
from glob import glob

import numpy as np
import tensorflow as tf
from PIL import Image


def _imread(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    return arr / 127.5 - 1.0


def _imresize(image, size):
    # image in [-1, 1], float
    uint8_img = ((image + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(uint8_img)
    resized = pil.resize((int(size), int(size)), Image.BILINEAR)
    arr = np.asarray(resized)
    return (arr / 127.5 - 1.0).astype(image.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True, help="path of .pb graph")
    p.add_argument("--dataset", required=True, help="path of images folder")
    p.add_argument("--crop_size", type=int, default=600, help="resize images to this square size")
    p.add_argument("--limit", type=int, default=0, help="limit number of images (0 = all)")
    args = p.parse_args()

    image_paths = sorted(glob(args.dataset.rstrip("/") + "/*.jpg"))
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        raise SystemExit("No .jpg files found in %s" % (args.dataset,))

    tf.reset_default_graph()

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            with tf.gfile.GFile(args.graph, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            tf.import_graph_def(graph_def, name="")

            a_input = graph.get_tensor_by_name("inputA:0")
            a_output = graph.get_tensor_by_name("a2b_generator/output_image:0")

            warm = _imresize(_imread(image_paths[0]), args.crop_size)
            warm.shape = (1, args.crop_size, args.crop_size, 3)
            sess.run(a_output, feed_dict={a_input: warm})

            start = time.time()
            first_100_end = None

            for i, path in enumerate(image_paths):
                a_feed = _imresize(_imread(path), args.crop_size)
                a_feed.shape = (1, args.crop_size, args.crop_size, 3)
                sess.run(a_output, feed_dict={a_input: a_feed})
                if i == 99:
                    first_100_end = time.time()

            end_all = time.time()

    total = len(image_paths)
    all_s = end_all - start
    print("Images: %d" % total)

    if first_100_end is not None:
        s_100 = first_100_end - start
        print("First 100: %.4fs (%.4f fps)" % (s_100, 100.0 / s_100))

    print("All: %.4fs (%.4f fps)" % (all_s, total / all_s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
