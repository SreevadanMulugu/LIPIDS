import os
import torch
import torchvision.utils as vutils
import numpy as np
from models import model_utils
from utils import eval_utils, time_utils 
import imageio

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

file_path = '/media/aalok/Zone_B/ashish/PS_Datasets/DiLiGenT102/objects.txt'
objs = read_file(file_path)
objs = [i.strip() for i in objs]

def get_itervals(args, split):
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    return disp_intv, save_intv

def test(args, split, loader, model, log, epoch, recorder):
    model.eval()
    print('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv = get_itervals(args, split)
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, timer, split)
            input = model_utils.getInput(args, data)
            out_var = None
            if args.model == 'LSNET_run':
                out_var = model.ps_fcn(input)
            elif args.model != 'LSNET' and args.model != 'SampleLS':
                out_var = model(input)
            else:
                out_var = model(input,30)
            timer.updateTime('Forward')
            acc = {'n_err_mean': 0}
            if args.evaluate:
                acc = eval_utils.calNormalAcc(data['tar'].data, out_var.data, data['m'].data) 
            else:
                out_var_magnitude = torch.norm(out_var.data, dim=1, keepdim=True)
                out_var = out_var.data / (out_var_magnitude + 1e-8)
                out_var.data = ((out_var.data + 1) / 2)*255 #
                out_var *= data['m'].data.expand_as(out_var)
                out_var.data *= data['m'].data.expand_as(out_var.data)
                normal = out_var.data.cpu().numpy()
                normal = normal.astype(np.uint8) #
                normal = np.transpose(normal, (0, 2, 3, 1))[0]
                # import scipy.io as sio
                # sio.savemat(args.save_result+f'{objs[i]}.mat', {'Normal_est': normal})
                imageio.imwrite(args.save_result+f'/normal{i}_{args.out_img}.png', normal)  
                
                if args.error_map:
                    gt_normal = data['tar'].data
                    gt_normal_magnitude = torch.norm(gt_normal, dim=1, keepdim=True)
                    gt_normal = gt_normal / (gt_normal_magnitude + 1e-8)
                    gt_normal_magnitude = torch.norm(gt_normal, dim=1, keepdim=True)
                    gt_normal = ((gt_normal + 1) / 2)*255
                    gt_normal *= data['m'].data.expand_as(gt_normal)
                    gt_normal = gt_normal.data.cpu().numpy()
                    gt_normal = gt_normal.astype(np.uint8)
                    gt_normal = np.transpose(gt_normal, (0, 2, 3, 1))[0]
                    
                    normal = normal.astype(np.float32)
                    gt_normal = gt_normal.astype(np.float32)
                    normal = 2*normal/255 - 1
                    gt_normal = 2*gt_normal/255 - 1
                    error = np.arccos(np.sum(normal*gt_normal, axis=2).clip(-1,1))/np.pi
                    error = (error)*255*3
                    error = error.clip(0,255)
                    error = error*data['m'].data[0,0].cpu().numpy()
                    error = error.astype(np.uint8)
                    error = np.expand_dims(error, axis=2)
                    error = np.concatenate((error, np.zeros((error.shape[0], error.shape[1], 2), dtype=np.uint8)), axis=2)
                    imageio.imwrite(args.save_result+f'/error_{args.out_img}_{i}.png', error)
                
                              
            recorder.updateIter(split, acc.keys(), acc.values())

            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                        'timer':timer, 'recorder': recorder}
                log.printItersSummary(opt)

            if iters % save_intv == 0:
                pred = (out_var.data + 1) / 2
                masked_pred = pred * data['m'].data.expand_as(out_var.data)
                log.saveNormalResults(masked_pred, split, epoch, iters)

    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

