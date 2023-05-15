import torch
from model import HybridDualEncoding


def compatible(model_path, vid_rnn_b, text_rnn_b):
    model_dict = torch.load(model_path,  map_location=torch.device('cpu'))
    model_dict['args']['vid_rnn_bidirectional'] = vid_rnn_b
    model_dict['args']['text_rnn_bidirectional'] = text_rnn_b

    model = HybridDualEncoding(**model_dict['args'])
    model.load_state_dict(model_dict['state_dict'])

    # resave_path = path.join(path.dirname(model_path), 'c_' + path.basename(model_path))
    model.save(model_path)

    model: HybridDualEncoding = HybridDualEncoding.load(model_path, 'cpu')
    print(model.video_encoder.rnn_bidirectional)
    print(model.text_encoder.rnn_bidirectional)

compatible('models_hybrid/05-06-13:53:19.bin', True, True)  # baseline 1
compatible('models_hybrid/05-13-07:56:45.bin', False, True) # unidirectional GRU on video encoder
compatible('models_hybrid/05-13-18:17:58.bin', False, False)    # unidirectional GRU on both vid & text encoder
compatible('models_hybrid/05-14-07:38:12.bin', True, True)  # baseline 2