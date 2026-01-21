#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from glob import glob

import numpy as np
import tensorflow as tf

import image_utils as im


tf.compat.v1.disable_eager_execution()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="path of .pb graph")
    parser.add_argument("--dataset", required=True, help="path of images folder")
    parser.add_argument(
        "--crop_size",
        type=int,
        default=600,
        help="resize images to this square size before inference",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="limit number of images (0 = all)",
    )
    args = parser.parse_args()

    image_paths = sorted(glob(args.dataset.rstrip("/") + "/*.jpg"))
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        raise SystemExit(f"No .jpg files found in {args.dataset}")

    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session() as sess:
            gfile = getattr(getattr(tf, "io", None), "gfile", None)
            gopen = getattr(gfile, "GFile", None) if gfile is not None else None
            if gopen is None:
                gopen = tf.gfile.GFile

            with gopen(args.graph, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())

            tf.import_graph_def(graph_def, name="")

            a_input = graph.get_tensor_by_name("inputA:0")
            a_output = graph.get_tensor_by_name("a2b_generator/output_image:0")

            # tiny warmup to avoid measuring graph setup
            warm = im.imresize(im.imread(image_paths[0]), [args.crop_size, args.crop_size])
            warm.shape = 1, args.crop_size, args.crop_size, 3
            _ = sess.run(a_output, feed_dict={a_input: warm})

            start = time.time()
            first_100_end = None

            for i, path in enumerate(image_paths):
                a_feed = im.imresize(im.imread(path), [args.crop_size, args.crop_size])
                a_feed.shape = 1, args.crop_size, args.crop_size, 3
                _ = sess.run(a_output, feed_dict={a_input: a_feed})

                if i == 99:
                    first_100_end = time.time()

            end_all = time.time()

    total = len(image_paths)
    all_s = end_all - start
    print(f"Images: {total}")

    if first_100_end is not None:
        s_100 = first_100_end - start
        print(f"First 100: {s_100:.4f}s ({100 / s_100:.4f} fps)")

    print(f"All: {all_s:.4f}s ({total / all_s:.4f} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
