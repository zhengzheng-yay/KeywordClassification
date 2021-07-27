#/bin/bash

echo "$0 $@"
stage=0
stop_stage=3

data_dir=/home/disk1/jyhou/my_data
file_name=speech_commands_v0.02.tar.gz
speech_command_dir=$data_dir/speech_commands_v2
audio_dir=$data_dir/speech_commands_v2/audio
url=http://download.tensorflow.org/data/$file_name

if [ ! -f $data_dir/$file_name ]; then
    echo "downloading $URL...\n"
    wget -O $data_dir $url 
else
    echo "$file_name exist in $data_dir, skip download it"
fi

if [ ! -f $speech_command_dir/.extracted ]; then
    mkdir -p $audio_dir
    tar -xzvf $data_dir/$file_name -C $audio_dir
    touch $speech_command_dir/.extracted
else
    echo "$speech_command_dir/.exatracted exist in $speech_command_dir, skip exatraction"
fi

if [ $stage -le -1 ]; then
    echo "splitting the dataset into train, validation and test sets..."
    python local/split_dataset.py $data_dir/speech_commands_v2
fi

sleep 1
# Prepare kaldi format files 
if [ $stage -le 0 ]; then
    echo "Start preparing Kaldi format files"
    for x in train test valid;
    do
        data_local=data_v2/local/$x
        data=data_v2/$x
        mkdir -p $data_local
        mkdir -p $data
        # make wav.scp utt2spk text file
        find $speech_command_dir/$x -name *.wav | grep -v "_background_noise_" > $data_local/wav.list  
        #echo "python prepare_speech_command.py $data_local/wav.list $data_local"
        python local/prepare_speech_command.py $data_local/wav.list $data_local
        for y in wav.scp utt2spk text;
        do 
            sort $data_local/$y > $data/$y
        done
        cat $data/utt2spk > $data/spk2utt
    done
fi

