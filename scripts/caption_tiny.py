import argparse
import os

import utils


def create_tiny_cap_file(cap_file, tiny_cap_file, cap_num_tiny):
    cap_ids, cap_text = utils.read_captions(cap_file)

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
    parser.add_argument('--train_cap_file', type=str, help='train caption file path to create tiny caption file from')
    parser.add_argument('--tiny_cap_file', type=str, help='target tiny caption file path')
    parser.add_argument('--cap_num', type=int, help='# of tiny captions to make for different video each.')

    args = parser.parse_args()

    create_tiny_cap_file(args.train_cap_file, args.tiny_cap_file, args.cap_num)
