import numpy as np
from torch.nn import Module, Embedding, Linear, GRUCell
from models.modules import *
from util.hparams import *


class Encoder(Module):
    def __init__(self, K, conv_dim):
        super(Encoder, self).__init__()
        # 처음 input 값에 대한 embedding 진행 (70, 256) ?
        self.embedding = Embedding(symbol_length, embedding_dim)
        # prenet(fully connected 2번)
        self.prenet = prenet(embedding_dim)

        self.cbhg = CBHG(K, conv_dim)
        
    def forward(self, enc_input, sequence_length, is_training):
        # enc_input은 텍스트를 tensor로 변환한 데이터
        x = self.embedding(enc_input)
        # embedding 과정을 거친 데이터를 prenet에 적용
        x = self.prenet(x, is_training=is_training)
        
        x = x.transpose(1, 2)

        # cbhg 과정 진행
        x = self.cbhg(x, sequence_length)
        return x

    
class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.prenet = prenet(mel_dim)
        self.attention_rnn = GRUCell(encoder_dim, decoder_dim)
        self.attention = LuongAttention()
        self.proj1 = Linear(decoder_dim * 2, decoder_dim)
        self.dec_rnn1 = GRUCell(decoder_dim, decoder_dim)
        self.dec_rnn2 = GRUCell(decoder_dim, decoder_dim)
        self.proj2 = Linear(decoder_dim, mel_dim * reduction)
        
    def forward(self, batch, dec_input, enc_output, mode):
        if mode == 'train':
            # transpose(0,1) 그냥 transpose
            dec_input = dec_input.transpose(0, 1)
            attn_rnn_state = torch.zeros(batch, decoder_dim).cuda()
            dec_rnn_state1 = torch.zeros(batch, decoder_dim).cuda()
            dec_rnn_state2 = torch.zeros(batch, decoder_dim).cuda()
        else:
            # test 환경은 직렬 구조 cpu 유리
            attn_rnn_state = torch.zeros(batch, decoder_dim)
            dec_rnn_state1 = torch.zeros(batch, decoder_dim)
            dec_rnn_state2 = torch.zeros(batch, decoder_dim)
        
        iters = dec_input.shape[0] if mode == 'train' else max_iter+1
        print(iters)
        
        for i in range(iters):
            inp = dec_input[i] if mode == 'train' else dec_input

            # Prenet (첫 번째)
            x = self.prenet(inp, is_training=True)


            # Attention RNN (두 번째)
            attn_rnn_state = self.attention_rnn(x, attn_rnn_state)
            # 차원 추가 [[1,2,3],[4,5,6]] → [[[1,2,3]], [4,5,6]]
            attn_rnn_state = attn_rnn_state.unsqueeze(1)
            # LuongAttention
            context, align = self.attention(attn_rnn_state, enc_output)

            # Decoder RNN의 input
            dec_rnn_input = self.proj1(context)
            # 차원 제거
            dec_rnn_input = dec_rnn_input.squeeze(1)

            # Decoder RNN (세 번째)
            dec_rnn_state1 = self.dec_rnn1(dec_rnn_input, dec_rnn_state1)
            # 이전의 state 결과를 input에 포함
            dec_rnn_input = dec_rnn_input + dec_rnn_state1
            dec_rnn_state2 = self.dec_rnn2(dec_rnn_input, dec_rnn_state2)
            dec_rnn_output = dec_rnn_input + dec_rnn_state2

            dec_out = self.proj2(dec_rnn_output)

            dec_out = dec_out.unsqueeze(1)
            attn_rnn_state = attn_rnn_state.squeeze(1)

            if i == 0:
                # 첫 트레이닝의 경우 dec_out을 활용
                mel_out = torch.reshape(dec_out, [batch, -1, mel_dim])
                # attention에서 뽑힌 alignment
                alignment = align
            else:
                # 이전의 mel_out과 새로운 dec_out 합치기
                mel_out = torch.cat([mel_out, torch.reshape(dec_out, [batch, -1, mel_dim])], dim=1)
                # alignment도 동일하게 합치기
                alignment = torch.cat([alignment, align], dim=-1)
                
            if mode == 'inference':
                # 합성의 경우 
                print(mel_out)
                dec_input = mel_out[:, reduction * (i+1) - 1, :]
                print(dec_input)
        # mel_out은 pred 결과값
        return mel_out, alignment
    
    
class Tacotron(Module):
    # K 16 넘겨주고, conv_dim은 128, 128
    def __init__(self, K, conv_dim):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(K, conv_dim)
        self.decoder = Decoder()
        
    def forward(self, enc_input, sequence_length, dec_input, is_training, mode):
        batch = dec_input.shape[0]
        # text와 text 길이
        x = self.encoder(enc_input, sequence_length, is_training)
        # dec 데이터와 그 크기, 
        x = self.decoder(batch, dec_input, x, mode)
        return x
    

class post_CBHG(Module):
    def __init__(self, K, conv_dim):
        super(post_CBHG, self).__init__()
        self.cbhg = CBHG(K, conv_dim)
        self.fc = Linear(256, n_fft // 2 + 1)
        
    def forward(self, mel_input):
        x = self.cbhg(mel_input.transpose(1, 2), None)
        x = self.fc(x)
        return x
    