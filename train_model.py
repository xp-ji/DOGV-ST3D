import os, json
import shutil
import argparse
import torch
from tqdm import tqdm
from torch.utils import data
from torch.optim import lr_scheduler
from models.model import generate_model
from dataloader.data_loader import depth_dataset
from tensorboardX import SummaryWriter
from utils.utils import AverageMeter, calculate_accuracy_percent, read_json_param


def read_json_param(filename):
    with open(filename, 'r') as lcf:
        config_param = json.load(lcf)
        return config_param

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(log_name, filename))
    if is_best:
        shutil.copyfile(os.path.join(log_name, filename), os.path.join(log_name, 'model_best.pth.tar'))

def validate(val_loader, model, criterion, opt, tb_writer):
    # switch to evaluation
    model.eval()

    losses = AverageMeter()
    accuracy = AverageMeter()

    bar_format = '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
    with tqdm(total=len(val_loader), bar_format=bar_format) as _tqdm:
        _tqdm.set_description('Eval')

        with torch.no_grad():
            for i, (input, target, _) in  enumerate(val_loader):
                
                if opt.cuda is not None:
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                # compute output
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                predicted_accuracy, n_correct_elems = calculate_accuracy_percent(output, target)
                accuracy.update(predicted_accuracy.item(), input.size(0))
                losses.update(loss.item(), input.size(0))

                n_iter = opt.epoch * len(val_loader) + i
                postfix = {'loss': '{:.4f}'.format(losses.avg), 'acc': '{:.2f}%'.format(accuracy.avg)}
                _tqdm.set_postfix(postfix)
                _tqdm.update(1)

                if i % opt.print_freq == 0:
                    tb_writer.add_scalar('iteration/test_loss', losses.avg, n_iter)
                    tb_writer.add_scalar('iteration/test_acc', accuracy.avg, n_iter)

    return accuracy.avg, losses.avg
    

def train_epoch(train_loader, model, criterion, optimizer, opt, tb_writer):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    accuracy = AverageMeter()
    bar_format = '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
    with tqdm(total=len(train_loader), bar_format=bar_format) as _tqdm:
        _tqdm.set_description('Train')
        
        for i, (input, target, _) in enumerate(train_loader):
            if opt.cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            optimizer.zero_grad()
            
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure metrics, eg., accuracy and record loss
            predicted_accuracy, _ = calculate_accuracy_percent(output.data, target)

            accuracy.update(predicted_accuracy.item(), input.size(0))
            losses.update(loss.item(), input.size(0))

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()

            n_iter = opt.epoch * len(train_loader) + i
            
            postfix = {'loss': '{:.4f}'.format(losses.avg), 'acc': '{:.2f}%'.format(accuracy.avg)}
            _tqdm.set_postfix(postfix)
            _tqdm.update(1)

            if i % opt.print_freq == 0:
                tb_writer.add_scalar('iteration/train_loss', losses.avg, n_iter)
                tb_writer.add_scalar('iteration/train_acc', accuracy.val, n_iter)
                tb_writer.add_scalar('iteration/lr', optimizer.param_groups[0]['lr'], n_iter)
    return accuracy.avg, losses.avg

def main_worker(opt):
    # loading dataset
    config = read_json_param(opt.config)

    train_set = depth_dataset(opt, config, subset='train')
    val_set = depth_dataset(opt, config, subset='test')

    train_loader = data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)
    val_loader = data.DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)

    sample_data, _, _ = train_set.__getitem__(0)
    data_shape = sample_data.shape # CxFxHxW

    opt.num_classes = config['num_classes']
    opt.in_channel = data_shape[0] #CxF
    opt.num_segments = data_shape[1]

    # load model
    model = generate_model(opt)

    print('=> Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
    
    torch.backends.cudnn.benchmark = True

    # SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    # Cross entropy loss
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = -1

    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_steps, gamma=opt.lr_decay)

    for epoch in range(opt.start_epoch, opt.epochs):
        opt.epoch = epoch

        # adjust the learning rate
        scheduler.step()

        print('==> Epoch: [{:d}/{:d}]'.format(opt.epoch, opt.epochs))
        # train for one epoch
        train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer, opt, tb_writer)

        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, criterion, opt, tb_writer)

        # remember best acc and save checkpoint
        is_better = val_acc > best_acc

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec': val_acc,
            'arch' : opt.arch,
            'exp': opt.exp,
            'intype': opt.intype,
            'dataset': opt.dataset
        }, is_better)

        tb_writer.add_scalar('Loss/train', train_loss, epoch)
        tb_writer.add_scalar('Loss/val', val_loss, epoch)
        tb_writer.add_scalar('Acc/train', train_acc, epoch)
        tb_writer.add_scalar('Acc/val', val_acc, epoch)

def get_options():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', type=str, choices=['Resnet3d', 'TSN', 'P3D', 'Resnet3d-18', 'TSN-resnet101', 'P3D-18'])
    parser.add_argument('--dataset', type=str, default='NTU_RGBD', choices=['NTU_RGBD', 'NTU_120', 'PKU_MMD', 'UOW_Combined3D'])
    parser.add_argument('--intype', type=str, default='DOGV', choices=['RDMs', 'DOGV'])
    parser.add_argument('--exp', type=str, default='X-Sub')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = get_options()
    opt.log_path = 'logs'

    opt.cuda = True
    opt.batch_size = 64
    opt.workers = 8

    opt.print_freq = 20
    opt.start_epoch = 0
    opt.epochs = 100

    opt.learning_rate = 0.01
    opt.momentum = 0.9
    opt.weight_decay = 1e-5
    opt.lr_steps = 30
    opt.lr_decay = 0.1
    opt.evaluate = False

    opt.config = 'configs/{dataset:s}_{exp:s}.json'.format(dataset=opt.dataset, exp=opt.exp)

    seed = 2
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    log_name = '{:s}/{:s}_{:s}_{:s}_{:s}'.format(opt.log_path, opt.dataset, opt.arch, opt.exp, opt.intype)
    os.makedirs(log_name, exist_ok=True)
    
    print(log_name)
    tb_writer = SummaryWriter(log_name)
    main_worker(opt)
