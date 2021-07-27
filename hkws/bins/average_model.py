import os
import argparse
import glob
import numpy as np

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')
    parser.add_argument('--val_best',
                        action="store_true",
                        help='averaged model')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')
    args = parser.parse_args()
    print(args)

    checkpoints = []
    val_scores = []
    num = args.num
    path_list = glob.glob('{}/[!avg][!final]*.pt'.format(args.src_path))
    if len(path_list) < num:
        print("Error: not enough models for average")
        exit(1)
    if args.val_best:
        # select num best checkpoint
        for path in path_list:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            val_scores.append(checkpoint['val_loss'])
            checkpoints.append(path)
        val_scores = np.array(val_scores)
        sort_idx = np.argsort(val_scores)
        selected_idx = sort_idx[:num]
        selected_val_scores = val_scores[selected_idx]
        path_list = [checkpoints[x] for x in list(selected_idx)]
        print("best val scores = " + str(selected_val_scores))
        print("selected epochs = " + str(path_list))
    else:
        # select last num checkpoint
        path_list = sorted(path_list, key=os.path.getmtime)
        path_list = path_list[-args.num:]
        print("selected epochs = " + str(path_list))
    avg = None
    assert num == len(path_list)
    for path in path_list:
        # print('Processing {}'.format(path))
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model_state_dict = checkpoint['model']
        if avg is None:
            avg = model_state_dict
        else:
            for k in avg.keys():
                avg[k] += model_state_dict[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(args.dst_model))
    torch.save({'model': avg}, args.dst_model) 

	
