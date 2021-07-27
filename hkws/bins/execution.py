import sys
import datetime
import torch

sys.path.append('./')
from hkws.utils.utils import AverageMeter
from hkws.models.loss import acc_frame

criterion = torch.nn.CrossEntropyLoss()
def _one_epoch(epoch, args, model, device, data_loader, optimizer=None, scheduler=None, is_train=False):
    """one epoch."""
    if is_train:
        tag="Train"
    else:
        tag="Val"
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    total_step = len(data_loader)
    for batch_idx, (utt_ids, act_lens, inputs, targets) in enumerate(data_loader):
        inputs, act_lens = inputs.to(device), act_lens.to(device)
        # Forward pass
        batch_size = inputs.shape[0]
        outputs = model(inputs, act_lens)
        targets = torch.LongTensor(targets).to(device)
        loss = criterion(outputs, targets)
        acc =  acc_frame(outputs, targets)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        acc_meter.update(acc, len(utt_ids))
        loss_meter.update(loss.item(), len(utt_ids))
        if batch_idx % args.log_interval == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Lr: {:.6f} {} Loss: {:.6f}, {} Acc: {:.6f}%  Time: {}'
                  .format(epoch, args.max_epochs, batch_idx+1, total_step, 
                      optimizer.param_groups[0]['lr'] if optimizer != None else 0.0, tag, loss_meter.cur, 
                          tag, acc_meter.cur, datetime.datetime.now()))

    print('Epoch: [{}/{}], Average {} Loss: {:.6f}, Average {} Acc: {:.6f}% Time: {}'
          .format(epoch, args.max_epochs, 
                  tag, loss_meter.avg, 
                  tag, acc_meter.avg, datetime.datetime.now()))
    return float(loss_meter.avg)

def train(args, model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    avg_loss = _one_epoch(epoch, args, model, device, 
                        train_loader, optimizer, scheduler, is_train=True)
    return avg_loss
    
def validate(args, model, device, dev_loader, epoch):
    """Cross validate the model."""
    model.eval()
    with torch.no_grad():
        avg_loss = _one_epoch(epoch, args, model, device, 
                        dev_loader, optimizer=None, is_train=False)
    return avg_loss

def test(args, model, device, test_loader, output_file):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for batch_idx, (utt_ids, act_lens, data, target) in enumerate(test_loader):
            data = data.to(device)
            act_lens = act_lens.to(device)
            batch_size = data.shape[0]
            output = model(data, act_lens)
            target = torch.LongTensor(target).to(device)
            loss = criterion(output, target)
            loss_meter.update(loss.item(), len(utt_ids))
            acc = acc_frame(output, target) 
            acc_meter.update(acc, len(utt_ids))
    print("Done, Time: {}".format(datetime.datetime.now()))
    tag="Test"
    print('\033[34mAverage {} Loss: {:.6f}, Average {} Acc: {:.6f}%'
          .format(tag, loss_meter.avg, 
                  tag, acc_meter.avg))
    print('\033[0m')
