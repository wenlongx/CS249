#!/usr/bin/env python

import sys
import pandas as pd

def newline_to_space(string):
    return string.replace('\r', ' ').replace('\n', ' ')

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("please give input and output file names as arguments")
        exit()

    df = pd.read_csv(sys.argv[1])

    # Convert newlines within each line to spaces, so that the document works with allennlp
    df["question_text"] = df["question_text"].apply(newline_to_space)

    df.to_csv(sys.argv[2], columns=["question_text"], index=False, header=False, sep='\n')
