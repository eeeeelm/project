import torch
from torch import nn
from torch.nn import functional as F

class MyModel(nn.Module):
    def __init__(self, enc_input_size, num_hiddens, num_layers, dec_input_size, dec_output_size, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Seq2SeqEncoder(enc_input_size, num_hiddens, num_layers, dropout)
        self.decoder = Seq2SeqDecoder(dec_input_size, num_hiddens, dec_output_size, num_layers, dropout)
    
    def forward(self, enc_X, dec_X):
        enc_output = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_output)
        return self.decoder(dec_X, dec_state)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def foward(self, enc_X, dec_X, *args):
        enc_output = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_output, *args)
        return self.decoder(dec_X, dec_state)
    
class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size, num_hiddens, num_layers, dropout=0, **kwarg) -> None:
        super(Seq2SeqEncoder, self).__init__(**kwarg)
        self.dense = nn.Linear(input_size, num_hiddens)
        self.rnn = nn.GRU(num_hiddens, num_hiddens, num_layers, dropout=dropout)
    
    def forward(self, X, *args):
        #embed后X为batchsize, timestep, embedsize
        #X = self.embedding(X)
        #adjust it to (timestep, batchsize, embedsize)
        X = self.dense(X)
        X = X.permute(1, 0, 2)
        out_put, state = self.rnn(X)
        #the shape of output is timestep, batchsize, num_hiddens
        #the shape of state is numlayers, batchsize, num_hiddens
        return out_put, state

class Seq2SeqDecoder(nn.Module):
    def __init__(self, dec_input_size, num_hiddens, dec_out_put_size, num_layers, dropout=0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(dec_input_size, num_hiddens)
        self.rnn = nn.GRU(num_hiddens + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense2 = nn.Linear(num_hiddens, dec_out_put_size)
    
    def init_state(self, enc_output, *args):
        return enc_output[1]

    def forward(self, X: torch.Tensor, state):
        X = self.dense1(X)
        X = X.permute(1, 0, 2)
        #X.shape is num_step, batch_size, embed_size
        #the shape of state is num_layers, batch_size, num_hiddens
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), dim=2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense2(output).permute(1, 0, 2)
        #state.shape is num_layers, batch_size, num_hiddens
        #output_shape is batch_size, num_steps, vocab_size
        return output, state

def train_seq2seq(net, data_iter, lr, num_epochs, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_uniform_(m._parameters[param])
        
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    net.train()
    for epoch in num_epochs:
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            #X.shape is batch_size, num_steps(same as Y)
            dec_input = Y[:, :60]
            expected_Y = Y[:, :-60]
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            grad_clip(net, 1)
            optimizer.step()

def predict_Seq2Seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weight=False):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    #turn the shape into batchsize, num_steps.(batchsize = 1)
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weights = [], []
    for _ in num_steps:
        Y, dec_state = net.decoder(dec_X, dec_state)
        #shape of Y is still batchsize, num_step, embed_size    
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weight:
            attention_weights.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ''.join(tgt_vocab.to_tokens(output_seq)), attention_weights