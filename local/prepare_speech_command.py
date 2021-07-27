#!/usr/bin/env python
import os
import sys

if __name__=='__main__':
    if len(sys.argv) < 3:
        print("USAGE: python %s wav.list local_dir"%sys.argv[0])
        exit(1)
    root_dir = sys.argv[2]
    f_wav_scp = open(os.path.join(root_dir,"wav.scp"), "w")
    f_text = open(os.path.join(root_dir, "text"), "w")
    f_utt2spk = open(os.path.join(root_dir, "utt2spk"), "w")
    with open(sys.argv[1]) as f:
        for line in f.readlines():
            keyword, file_name = line.strip().split("/")[-2:]
            #print(keyword)
            #print(file_name)
            file_name_new = file_name.split(".")[0]
            wav_id = "_".join([keyword, file_name_new])
            file_dir = line.strip()
            f_wav_scp.writelines(wav_id + " " + file_dir + "\n")
            f_text.writelines(wav_id + " " + keyword + "\n")
            f_utt2spk.writelines(wav_id + " " + wav_id + "\n")
    f_wav_scp.close()
    f_text.close()
    f_utt2spk.close()

            
