#!/usr/bin/env python

# Copyright 2017 houjingyong@gmail.com 
#
# MIT licence
import os
import sys

CLASSES = 'unknown, yes, no, up, down, left, right, on, off, stop, go'.split(', ')
CLASS_TO_IDX = {CLASSES[i]: str(i) for i in range(len(CLASSES))}

if __name__=='__main__':
    if len(sys.argv) < 3:
        print("USAGE:python %s text label\n"%(sys.argv[0]))
        exit(1)
    f = open(sys.argv[2], 'w')    
    for x in open(sys.argv[1]).readlines():
        wav_id, c = x.strip().split()
        if c in CLASS_TO_IDX:
            f.writelines(wav_id + " " + CLASS_TO_IDX[c] +"\n")
        else:
            f.writelines(wav_id + " " + CLASS_TO_IDX["unknown"] + "\n")
    f.close()

