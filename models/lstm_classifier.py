import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):

    def __init__(self, dimension=128, emb_inp_size=300):
        super(LSTM, self).__init__()

        #self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=emb_inp_size,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 2)

    def forward(self, inp_emb, inp_len):

        #text_emb = self.embedding(text)

        print('inp_shaep',inp_emb.shape)
        packed_input = pack_padded_sequence(inp_emb, inp_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        print(f"Output - {output.shape}")
        out_forward = output[range(len(output)), inp_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        print(f"Output forward - {out_forward.shape}")
        print(f"Output Reverse - {out_reverse.shape}")
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        print(f"Out Reduced - {out_reduced.shape}")
        text_fea = self.drop(out_reduced)
        print("Before fc", text_fea.shape)
        text_fea = self.fc(text_fea)
        #text_fea = torch.squeeze(text_fea, 1)
        print('After fc',text_fea.shape)
        text_out = torch.softmax(text_fea)

        return text_out