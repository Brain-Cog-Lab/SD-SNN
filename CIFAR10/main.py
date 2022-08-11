import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import sys
sys.path.append('..')

from braincog.base.node.node import *
from braincog.utils import *
from braincog.base.utils.criterions import *
from braincog.datasets.datasets import *
from braincog.model_zoo.resnet import *
from braincog.model_zoo.convnet import *
from braincog.utils import save_feature_map, setup_seed

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import ImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"),'c10','1024-5'])
output_dir = get_outdir('./', 'train', exp_name)
log = open(os.path.join(output_dir, '.txt'), 'w')

def print_log(print_string, log):
    #print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

# from ptflops import get_model_complexity_info
print("sss")
model='cifar_convnet'
dataset='cifar10'
num_classes=10
step=8
encode='direct'
node_type='PLIFNode'
thresh=0.5
tau=2.0
torch.backends.cudnn.benchmark = True
devicee=0
seed=0
torch.cuda.set_device('cuda:%d' % devicee)
torch.manual_seed(seed)

model = create_model(
    model,
    pretrained=False,
    num_classes=num_classes,
    dataset=dataset,
    step=step,
    encode_type=encode,
    node_type=eval(node_type),
    threshold=thresh,
    tau=tau,
    spike_output=True
)
model = model.cuda()
print(model)
channels = 2
lr=5e-3
batch_size=50
epochs=600
linear_scaled_lr = lr * batch_size/ 1024.0
config_parser = cfg = argparse.ArgumentParser(description='Training Config', add_help=False)
cfg.opt='adamw'
cfg.lr=linear_scaled_lr
cfg.weight_decay=0.01
cfg.momentum=0.9
cfg.epochs=epochs
cfg.sched='cosine'
cfg.min_lr=1e-5
cfg.warmup_lr=1e-6
cfg.warmup_epochs=5
cfg.cooldown_epochs=10
cfg.decay_rate=0.1
optimizer = create_optimizer(cfg, model)
lr_scheduler, num_epochs = create_scheduler(cfg, optimizer)

loader_train, loader_eval, mixup_active, mixup_fn = eval('get_%s_data' % dataset)(batch_size=batch_size, step=step)
print(len(loader_train), len(loader_eval))
train_loss_fn = UnilateralMse(1.)
validate_loss_fn = UnilateralMse(1.)

eval_metric='top1'
best_test = 0
best_testepoch = 0
best_testprun = 0
best_testepochprun = 0

from maskup import *
m = Mask(model,18)
epoch_prune = 1

    
def train_epoch(
        epoch, model, loader, optimizer, loss_fn,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (inputs, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        inputs, target = inputs.type(torch.FloatTensor).cuda(), target.cuda()
        output = model(inputs)


        loss = loss_fn(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_m.update(loss.item(), inputs.size(0))
        top1_m.update(acc1.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx %40 == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            print(epoch,loss.data,losses_m.avg,acc1, top1_m.avg,lr)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
            #print(inputs.size())
            # inputs = inputs.type(torch.float64)
            last_batch = batch_idx == last_idx
            inputs = inputs.type(torch.FloatTensor).cuda()
            target = target.cuda()

            output = model(inputs)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), output.size(0))
            if last_batch or batch_idx %100 == 0:
                print(loss.data,losses_m.avg,acc1, top1_m.avg)

            batch_time_m.update(time.time() - end)
            end = time.time()


    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


for epoch in range(epochs):

    train_metrics= train_epoch(
        epoch, model, loader_train, optimizer, train_loss_fn,
        lr_scheduler=lr_scheduler)

    if epoch==0:
        m.init_length()
    if epoch>0:
        m.model = model
        m.init_mask_dsd()
        if epoch>5:
            matt=m.do_mask_dsd()
        if epoch>36:
            # m.if_zero()
            m.do_growth_ww(epoch)
            matt=m.do_pruning_dsd(epoch)
        model = m.model
    cc=m.if_zero()

    eval_metrics = validate(model, loader_eval, validate_loss_fn)
    top1=eval_metrics['top1']
    if top1 > best_testprun:
        best_testprun = top1
        best_testepochprun =epoch
    if epoch%40==0:
        print('pruning:',best_testprun,best_testepochprun)
    if epoch>4:
        print_log('*** epoch: {0} (acc:{1},cc:{2})'.format(epoch,top1,cc), log)
    if lr_scheduler is not None:
        lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
