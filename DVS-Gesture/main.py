
from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import *
from mask import *
import argparse
from datetime import datetime
import logging
from timm.utils import *
from dvs128_gesture import DVS128Gesture
import random
from PIL import Image,ImageFilter

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
names = 'spiking_model'
DATA_DIR = '/data0/datasets/'

train_transform = transforms.Compose([lambda x: torch.tensor(x),
                                          transforms.RandomCrop(128, padding=16),
                                          # transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(5)])
train_datasets = DVS128Gesture(os.path.join(DATA_DIR, 'DVS/DVS_Gesture'), train=True, transform=train_transform,
                                   data_type='frame', split_by='number', frames_number=64)
test_datasets = DVS128Gesture(os.path.join(DATA_DIR, 'DVS/DVS_Gesture'), train=False,
                                  data_type='frame', split_by='number', frames_number=64)

train_loader = torch.utils.data.DataLoader(
        dataset=train_datasets,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=8
)
test_loader = torch.utils.data.DataLoader(
        dataset=test_datasets,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2
)


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(0) 

_logger = logging.getLogger('train')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"),'nn','60-2'])
output_dir = get_outdir('./', 'train', exp_name)
setup_default_logging(log_path=os.path.join(output_dir, 'log.txt'))

best_acc = 0  # best test accuracy
best=0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
pruning=0.5
rate_decay = 600
epoch_prune = 1
NUM=0

snn = SCNN()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

m = Mask(snn,18)

for epoch in range(num_epochs):
        
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):

        snn.zero_grad()
        optimizer.zero_grad()
        images = images.float().to(device)
        outputs= snn(images, epoch=epoch, train=1)

        labels_ = torch.zeros(batch_size, 11).scatter_(1, labels.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                  % (epoch + 1, num_epochs, i + 1, len(train_datasets) // batch_size, running_loss))
            running_loss = 0
            print('Time elasped:', time.time() - start_time)
        
    correct = 0
    total = 0
    # if epoch %30 == 0 and epoch >1:
    #     pruning = pruning - 0.05
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

    if epoch==0:
        m.init_length()
    if epoch>0:
        m.model = snn
        m.init_mask_dsd()
        if epoch>18:
            matt=m.do_mask_dsd()
        if epoch>36:
            m.if_zero()
            m.do_growth_ww(epoch)
            matt=m.do_pruning_dsd(epoch)
        snn = m.model
    cc=m.if_zero()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs)
            labels_ = torch.zeros(batch_size, 11).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx % 100 == 0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)

    print('Iters:', epoch, '\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if best_acc < acc:
        best_acc = acc

    _logger.info('*** epoch: {0} (prun {1},acc:{2})'.format(epoch, cc,acc))
    if epoch % 10 == 0:
        print(best_acc)
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + names + '.t7')
    
        




