import os
import torch

def generate_model(opt):
    num_classes = opt.num_classes
    in_channel = opt.in_channel
    num_segments = opt.num_segments

    if 'resnet3d' in str.lower(opt.arch):
        if '-' in str.lower(opt.arch):
            num_layer = int(opt.arch.split('-')[-1])
        else:
            num_layer = 18
        from models.Resnet3D import Resnet3D
        model = Resnet3D(num_layer=num_layer, num_classes=num_classes, in_channel=in_channel, num_segments=num_segments)
    
    elif 'tsn' in str.lower(opt.arch):
        if '-' in str.lower(opt.arch):
            base_model = opt.arch.split('-')[-1]
        else:
            base_model = 'resnet101'
        from models.TSN import TSN
        model = TSN(base_model=base_model, num_classes=num_classes, in_channel=in_channel, num_segments=num_segments)
    
    elif 'p3d' in str.lower(opt.arch):
        if '-' in str.lower(opt.arch):
            num_layer = int(opt.arch.split('-')[-1])
        else:
            num_layer = 18
        from models.P3D import P3D
        model = P3D(num_layer=num_layer, num_classes=num_classes, in_channel=in_channel, num_segments=num_segments)
    
    else:
        raise RuntimeError('Unknown model: {:s}'.format(opt.arch))

    if opt.cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    
    if opt.evaluate:
        assert os.path.exists(opt.weights), "The file '{:s}' does not exist!"
        print('=> Loading weights from {}'.format(opt.weights))

        checkpoint = torch.load(opt.weights)
        
        arch = checkpoint['arch']
        exp = checkpoint['exp']
        intype = checkpoint['intype']
        assert arch == opt.arch, "The arch '{:s}' of checkpoint is not consistent with current model!".format(arch)
        assert exp == opt.exp, "The protocol '{:s}' of checkpoint is not consistent with current model!".format(exp)
        assert intype == opt.intype, "The input type '{:s}' of checkpoint is not consistent with current model!".format(intype)
        
        print('checkpoint accuracy: {:.2f}%'.format(checkpoint['best_prec']))

        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
    return model