#!/usr/bin/env python

import h5py
import sys

if __name__ == "__main__":

    if len(sys.argv) < 1:
        print("please give embedding filename")
        exit()

    with h5py.File(sys.argv[1], "r") as h5py_file:

        num_lines = len(h5py_file) - 1

        # This is the embedding for the n'th line in the file
        # 
        # The shape is (num_words, 1024)
        # Each word in the sentence gets its own embedding
        # Each word embedding is of dim 1024
        embedding = h5py_file.get("0")

        """
        Exxample: line 0

        >>> embedding[0]
        array([-0.28788558, -0.04935288, -0.18455128, ..., -0.2811056 ,
                0.18217532,  0.26742932], dtype=float32)
        """

