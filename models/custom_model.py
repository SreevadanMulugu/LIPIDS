from . import model_utils
from descritization import conversion

def buildModel(args):
    print('Creating Model %s' % (args.model))
    in_c = model_utils.getInputChanel(args)
    other = {'img_num': args.in_img_num, 'in_light': args.in_light}
    if args.model == 'NRNET': 
        from models.NRNET import NRNET
        model = NRNET(args.fuse_type, args.use_BN, in_c, other)
    elif args.model == 'NRNET_run':
        from models.NRNET_run import NRNET
        model = NRNET(args.fuse_type, args.use_BN, in_c, other)
    elif args.model == 'LSNET' or args.model == 'LSNET_run':
        from models.LSNET import LSNET
        model = LSNET(conversion.total_bins, int(args.out_img),args.fuse_type, args.use_BN,in_c,other)
    elif args.model == 'OLSNET' or args.model == 'OLSNET_run':
        from models.OLSNET import OLSNET
        model = OLSNET(conversion.total_bins, int(args.out_img),args.fuse_type, args.use_BN,in_c,other)
    elif args.model =='LS':
        from models.LS import LS
        model = LS()
    elif args.model =='SampleLS':
        from models.SampleLS import SampleLS
        model = SampleLS(conversion.total_bins, int(args.out_img))
    else:
        raise Exception("=> Unknown Model '{}'".format(args.model))
    
    if args.cuda: 
        model = model.cuda()

    if args.retrain and args.model!='LS': 
        print("=> using pre-trained model %s" % (args.retrain))
        model_utils.loadCheckpoint(args.retrain, model, cuda=args.cuda)

    if args.resume:
        print("=> Resume loading checkpoint %s" % (args.resume))
        model_utils.loadCheckpoint(args.resume, model, cuda=args.cuda)
    print(model)
    print("=> Model Parameters: %d" % (model_utils.get_n_params(model)))
    return model