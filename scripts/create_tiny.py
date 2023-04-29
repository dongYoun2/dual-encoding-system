import argparse
import os

import utils


def create_tiny_cap_file(cap_file, cap_num_tiny):
    cap_ids, cap_text = utils.read_captions(cap_file)
    orig_path, ext = os.path.splitext(cap_file)
    tiny_cap_file = f'{orig_path}_tiny{ext}'

    cnt = 0
    prev_vid_id = ''
    with open(tiny_cap_file, 'w') as f:
        for c_id, c_text in zip(cap_ids, cap_text):
            curr_vid_id = utils.vid_id_from_cap_id(c_id)
            if prev_vid_id != curr_vid_id:
                line = f'{c_id} {c_text}\n'
                f.write(line)
                cnt += 1
                prev_vid_id = curr_vid_id

                if cnt >= cap_num_tiny:
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cap_file', type=str, default='data/msrvtt10k/TextData/msrvtt10ktrain.caption.txt', help='train caption file path. read captions from there and create tiny caption files in same dir. "_tiny" will be added to the end.')
    parser.add_argument('--val_cap_file', type=str, default='data/msrvtt10k/TextData/msrvtt10kval.caption.txt', help='validation caption file path. read captions from there and create tiny caption files in same dir.')
    parser.add_argument('--test_cap_file', type=str, default='data/msrvtt10k/TextData/msrvtt10ktest.caption.txt', help='test caption file path. read captions from there and create tiny caption files in same dir.')
    parser.add_argument('--cap_num_tiny', type=int, default=8, help='# of tiny captions to make for different video each.')

    args = parser.parse_args()

    create_tiny_cap_file(args.train_cap_file, args.cap_num_tiny)
    create_tiny_cap_file(args.val_cap_file, args.cap_num_tiny)
    create_tiny_cap_file(args.test_cap_file, args.cap_num_tiny)