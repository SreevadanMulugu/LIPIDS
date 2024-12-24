from models import model_utils
from utils  import time_utils 
import torch

def train(args, loader, model, criterion, optimizer, log, epoch, recorder):

    model.train()
    print('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync)

    for i, sample in enumerate(loader):
        data  = model_utils.parseData(args, sample, timer, 'train')
        input = model_utils.getInput(args, data)
        if args.model!='LSNET'and args.model!='SampleLS':
            out_var = model(input); 
        else :
            out_var = model(input,epoch)
        timer.updateTime('Forward')

        optimizer.zero_grad()
        loss = criterion.forward(out_var, data['tar']); timer.updateTime('Crit');
        criterion.backward(); timer.updateTime('Backward')
        # if args.model =='LSNET':
        #     model.ps_fcn.zero_grad()

        recorder.updateIter('train', loss.keys(), loss.values())
        
        if args.model == 'OLSNET':
            model.ps_fcn.zero_grad()
        # print(model.weight.grad)
        if args.model == 'SampleLS':
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e10)
            model.weight.grad = 1e-6*(model.weight.grad/torch.max(abs(model.weight.grad)))
        # print(torch.max(model.weight.grad))
        # exit(0)
        optimizer.step(); timer.updateTime('Solver')

        iters = i + 1
        if iters % args.train_disp == 0:
            opt = {'split':'train', 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                    'timer':timer, 'recorder': recorder}
            log.printItersSummary(opt)
            # if args.model=='LSNET':
            #     print(model.weight)

    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)
