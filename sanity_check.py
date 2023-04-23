import utils


def sc_read_captions():
    path = 'data/msrvtt10k/TextData/msrvtt10kval.caption.txt'
    max_lens = [-1, 0, 10]

    for m_len in max_lens:
        ids, captions = utils.read_captions(path, m_len)

        assert len(ids) == len(captions)
        if m_len != -1: assert len(ids) == m_len

    print("sc_read_captions() passed!")

if __name__ == '__main__':
    sc_read_captions()