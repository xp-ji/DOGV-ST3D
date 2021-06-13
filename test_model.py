import os
import csv
import argparse
import torch
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils import data
from models.model import generate_model
from dataloader.data_loader import depth_dataset
from utils.utils import AverageMeter, calculate_accuracy_percent, read_json_param
from sklearn.metrics import confusion_matrix

def test_model(val_loader, model, opt):
    accuracy = AverageMeter()
    samples_count = AverageMeter()
    samples_right = AverageMeter()

    # switch to evaluation
    model.eval()

    bar_format = '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
    with tqdm(total=len(val_loader), bar_format=bar_format) as _tqdm:
        _tqdm.set_description('Test')

        with torch.no_grad():
            y_true = []
            y_pred = []
            output_list = []
            fids = []

            for i, (input, target, fid) in  enumerate(val_loader):
                
                if opt.cuda is not None:
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                # compute output
                output = model(input)
                output_list.append(output)

                fids.extend(fid.tolist())

                y_true.extend(target.tolist())
                _, pred = output.topk(1, 1, True, True)
                y_pred.extend(pred.t().tolist()[0])

                # measure accuracy and record loss
                predicted_accuracy, n_correct_elems = calculate_accuracy_percent(output, target)
                accuracy.update(predicted_accuracy.item(), input.size(0))
                samples_count.update(input.size(0))
                samples_right.update(n_correct_elems.item())

                postfix = {'acc': '{:.2f}%'.format(accuracy.avg)}
                _tqdm.set_postfix(postfix)
                _tqdm.update(1)

            output_score = F.softmax(torch.cat(output_list, dim=0), dim=1)
            error_fids = []
            for i, y_t in enumerate(y_true):
                if y_t != y_pred[i]:
                    error_fids.append(i)

            save_predict_result(
                {'output_score': output_score,
                'true_label': y_true,
                'pred_label': y_pred},
                log_name,
                opt.action_names
            )

    print("=> Test result has been stored in '{:s}'".format(log_name))
    return accuracy.avg, error_fids


def save_predict_result(state, output_file_path, action_names_file):
    output_pth_name = "{:s}/test_result.pth.tar".format(output_file_path)
    torch.save(state, output_pth_name)

    # write result to csv
    txt_file_name = output_pth_name.replace(".pth.tar", ".txt")
    csv_file_name = output_pth_name.replace(".pth.tar", ".csv")
    true_label = state["true_label"]
    pred_label = state["pred_label"]

    cnf_matrix = confusion_matrix(true_label, pred_label, labels=None, sample_weight=None, normalize='true')

    with open(txt_file_name, "w") as f:
        for i in range(len(pred_label)):
            line = "{:d}\t{:d}\t{:d}\n".format(i, true_label[i], pred_label[i])
            f.write(line)
        f.close()

    action_names = []
    with open(action_names_file, "r") as f:
        for line in f.readlines():
            action_name = line.strip("\n")
            action_names.append(action_name)

    temp = cnf_matrix.diagonal()
    with open(csv_file_name,'w')as f:
        f_csv = csv.writer(f)
        for i, val in enumerate(temp):
            f_csv.writerow(['{:d}'.format(i), '{:s}'.format(action_names[i]), '{:.2f}'.format(val*100)])

    return


def main_worker(opt):
    # loading dataset
    config = read_json_param(opt.config)

    test_set = depth_dataset(opt, config, subset='test')
    test_loader = data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)

    sample_data, _, _ = test_set.__getitem__(0)
    data_shape = sample_data.shape # CxFxHxW

    opt.num_classes = config['num_classes']
    opt.in_channel = data_shape[0] #CxF
    opt.num_segments = data_shape[1]
    opt.action_names = config['action_names']

    # load model
    model = generate_model(opt)
    torch.backends.cudnn.benchmark = True

    test_model(test_loader, model, opt)

    return


def get_options():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', type=str, choices=['Resnet3d', 'TSN', 'P3D', 'Resnet3d-18', 'TSN-resnet101', 'P3D-18'])
    parser.add_argument('--dataset', type=str, default='NTU_RGBD', choices=['NTU_RGBD', 'NTU_120', 'PKU_MMD', 'UOW_Combined3D'])
    parser.add_argument('--intype', type=str, default='DOGV', choices=['RDMs', 'DOGV'])
    parser.add_argument('--exp', type=str, default='X-Sub')
    parser.add_argument('--weights', type=str, default="logs/NTU_RGBD_Resnet3d_X-Sub_RDMs/model_best.pth.tar")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = get_options()
    opt.log_path = 'logs'

    opt.cuda = True
    opt.batch_size = 64
    opt.workers = 8

    opt.print_freq = 20
    opt.evaluate = True

    opt.config = 'configs/{dataset:s}_{exp:s}.json'.format(dataset=opt.dataset, exp=opt.exp)
    
    seed = 2
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    log_name = '{:s}/Test_{:s}_{:s}_{:s}_{:s}'.format(opt.log_path, opt.dataset, opt.arch, opt.exp, opt.intype)
    os.makedirs(log_name, exist_ok=True)
    
    print('----' * 20)
    print('=> {:s}'.format(log_name))
    main_worker(opt)
