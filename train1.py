import os, argparse, traceback, glob, random, itertools, time, torch, threading, queue
import numpy as np
import torch.optim as optim
from models.tacotron import Tacotron
from torch.nn import L1Loss
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from util.text import sequence_to_text
from util.plot_alignment import plot_alignment
from util.hparams import *


data_dir = './data'
text_list = sorted(glob.glob(os.path.join(data_dir + '/text', '*.npy')))
mel_list = sorted(glob.glob(os.path.join(data_dir + '/mel', '*.npy')))
dec_list = sorted(glob.glob(os.path.join(data_dir + '/dec', '*.npy')))

fn = os.path.join(data_dir + '/mel_len.npy')
# mel_len에 대한 정보가 없을 때
if not os.path.isfile(fn):
    # 직접 mel_len을 만들어내서 저장해
    mel_len_list = []
    for i in range(len(mel_list)):
        mel_length = np.load(mel_list[i]).shape[0]
        mel_len_list.append([mel_length, i])
    mel_len = sorted(mel_len_list)
    np.save(os.path.join(data_dir + '/mel_len.npy'), np.array(mel_len))

text_len = np.load(os.path.join(data_dir + '/text_len.npy'))
mel_len = np.load(os.path.join(data_dir + '/mel_len.npy'))

    
def DataGenerator():
    while True:
        # 랜덤으로 인덱스 batch*batch 만큼 뽑고 sort, 랜덤
        idx_list = np.random.choice(len(mel_list), batch_group, replace=False)
        idx_list = sorted(idx_list)
        idx_list = [idx_list[i : i + batch_size] for i in range(0, len(idx_list), batch_size)]
        random.shuffle(idx_list)

        for idx in idx_list:
            random.shuffle(idx)

            # 텐서 형태로 변환하여 저장
            text = [torch.from_numpy(np.load(text_list[mel_len[i][1]])) for i in idx]
            dec = [torch.from_numpy(np.load(dec_list[mel_len[i][1]])) for i in idx]
            mel = [torch.from_numpy(np.load(mel_list[mel_len[i][1]])) for i in idx]

            text_length = torch.tensor([text_len[mel_len[i][1]] for i in idx], dtype=torch.int32)
            text_length, _ = text_length.sort(descending=True)

            # padding을 통해 길이가 같지 않은 항목에 추가 후 일치
            text = pad_sequence(text, batch_first=True)
            dec = pad_sequence(dec, batch_first=True)
            mel = pad_sequence(mel, batch_first=True)

            yield [text, dec, mel, text_length]
            
            
class Generator(threading.Thread):
    def __init__(self, generator):
        threading.Thread.__init__(self)
        # queue, generator 생성
        self.queue = queue.Queue(8)
        self.generator = generator
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        # que에 있는 아이템을 가져옴
        next_item = self.queue.get()
        if next_item is None:
             raise StopIteration
        return next_item


def train(args):
    train_loader = Generator(DataGenerator())

    # Tacotron 불러오기
    model = Tacotron(K=16, conv_dim=[128, 128]).cuda()

    # 옵티마이저는 아담
    optimizer = optim.Adam(model.parameters())

    step, epochs = 0, 0
    # 체크포인트가 있을 경우 torch.load와 load_state_dict로 model과 optimizer, step, epoch 불러옴
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt['step'],
        step = step[0]
        epoch = ckpt['epoch']
        print('Load Status: Epoch %d, Step %d' % (epoch, step))

    # True면 cudnn이 다중 conv 알고리즘을 벤치마킹해 가장 빠른 알고리즘 선택
    torch.backends.cudnn.benchmark = True

    try:
        # itertools는 반복물을 효율적으로 만드는 함수
        for epoch in itertools.count(epochs):
            for _ in range(batch_group):
                start = time.time()
                # 이번 차례의 학습 데이터 가져오기
                text, dec, target, text_length = train_loader.next()

                # 쿠다로 작동
                text = text.cuda()
                dec = dec.float().cuda()
                target = target.float().cuda()

                # 모델에 문장, dec 등 정보를 트레이닝 형식으로 보내어 pred와 정렬 결과값 return
                pred, alignment = model(text, text_length, dec, is_training=True, mode='train')
                # loss는 L1을 활용
                loss = L1Loss()(pred, target)

                # backpropagation 전에 gradients를 zero로 만들고 시작 (무조건)
                # zero_grad 이유 -> backward 하면 파라미터의 .grad 값에 변화도 저장
                # zero_grad 없을 경우 이전 루프의 저장 값이 다음 루프에 간섭하여 원하는 학습 X
                # 따라서 zero_grad로 .grad 값을 0으로 초기화 시키고 학습 진행
                model.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                print('step: {}, loss: {:.5f}, {:.3f} sec/step'.format(step, loss, time.time() - start))

                # step이 체크포인트 때마다 ckpt에 저장
                if step % checkpoint_step == 0:
                    save_dir = './ckpt/' + args.name + '/1'
                    
                    # 학습 외에는 cpu를 활용하여 직렬 구조를 더 빠르게 실행
                    input_seq = sequence_to_text(text[0].cpu().numpy())
                    input_seq = input_seq[:text_length[0].cpu().numpy()]

                    # 저장 경로, 이름 설정
                    alignment_dir = os.path.join(save_dir, 'step-{}-align.png'.format(step))
                    plot_alignment(alignment[0].detach().cpu().numpy(), alignment_dir, input_seq)
                    # 경로에 현재 모델, 옵티마이저, 스탭, 에포크 등 담기
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'epoch': epoch
                    }, os.path.join(save_dir, 'ckpt-{}.pt'.format(step)))

    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', default=None)
    parser.add_argument('--name', '-n', required=True)
    args = parser.parse_args()
    save_dir = os.path.join('./ckpt/' + args.name, '1')
    os.makedirs(save_dir, exist_ok=True)
    train(args)
