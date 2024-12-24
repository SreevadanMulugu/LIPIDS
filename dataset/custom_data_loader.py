import torch.utils.data

def customDataloader(args):
    print("=> fetching img pairs in %s" % (args.data_dir))
    if args.dataset == 'PS_Synth_Dataset':
        from dataset.PS_Synth_Dataset import PS_Synth_Dataset
        train_set = PS_Synth_Dataset(args, args.data_dir, 'train')
        val_set   = PS_Synth_Dataset(args, args.data_dir, 'val')
    elif args.dataset == 'PS_SampleNet_Dataset':
        from dataset.PS_SampleNet_Dataset import PS_SampleNet_Dataset
        train_set = PS_SampleNet_Dataset(args, args.data_dir, 'train')
        val_set   = PS_SampleNet_Dataset(args, args.data_dir, 'val')
    else:
        raise Exception('Unknown dataset: %s' % (args.dataset))

    if args.concat_data:
        print('****** Using cocnat data ******')
        print("=> fetching img pairs in %s" % (args.data_dir2))
        if args.dataset == 'PS_Synth_Dataset':
            from dataset.PS_Synth_Dataset import PS_Synth_Dataset
            train_set2 = PS_Synth_Dataset(args, args.data_dir2, 'train')
            val_set2   = PS_Synth_Dataset(args, args.data_dir2, 'val')
        elif args.dataset == 'PS_SampleNet_Dataset':
            from dataset.PS_SampleNet_Dataset import PS_SampleNet_Dataset
            train_set2 = PS_SampleNet_Dataset(args, args.data_dir2, 'train')
            val_set2   = PS_SampleNet_Dataset(args, args.data_dir2, 'val')
            
        train_set  = torch.utils.data.ConcatDataset([train_set, train_set2])
        val_set    = torch.utils.data.ConcatDataset([val_set,   val_set2])

    print('\t Found Data: %d Train and %d Val' % (len(train_set), len(val_set)))
    print('\t Train Batch %d, Val Batch: %d' % (args.batch, args.val_batch))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch,
                        num_workers=args.workers, pin_memory=args.cuda, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(val_set , batch_size=args.val_batch,
                        num_workers=args.workers, pin_memory=args.cuda, shuffle=False)
    return train_loader, test_loader

def benchmarkLoader(args):
    print("=> fetching img pairs in data/%s" % (args.benchmark))
    if args.benchmark == 'DiLiGenT_main':
        from dataset.DiLiGenT_main import DiLiGenT_main
        test_set  = DiLiGenT_main(args, 'test')
    elif args.benchmark == 'DiLiGenT_SampleNet':
        from dataset.DiLiGenT_SampleNet import DiLiGenT_SampleNet
        test_set = DiLiGenT_SampleNet(args, 'test')
    elif args.benchmark == 'Scultpure':
        from dataset.PS_SampleNet_Dataset import PS_SampleNet_Dataset
        test_set   = PS_SampleNet_Dataset(args, './data/datasets/PS_Blobby_Dataset', 'val')
    elif args.benchmark == 'Bunny':
        from dataset.Bunny import Bunny
        test_set = Bunny(args,'/media/aalok/Zone_B/ashish/PS_Datasets/Blobby_Sculpture_Bunny/PS_Sphere_Bunny','test')
    elif args.benchmark == 'Bunny_DH':
        from dataset.Bunny_DH import Bunny_DH
        test_set = Bunny_DH(args,'/media/aalok/Zone_B/ashish/PS_Datasets/Blobby_Sculpture_Bunny/PS_Sphere_Bunny','test')
    elif args.benchmark == 'Apple_Gourd':
        from dataset.Apple_Gourd import Apple_Gourd
        test_set = Apple_Gourd(args,'/media/aalok/Zone_B/ashish/PS_Datasets/Apple_Gourd')
    elif args.benchmark == 'Light_Stage_Gallery':
        from dataset.Apple_Gourd import Apple_Gourd
        test_set = Apple_Gourd(args,'/media/aalok/Zone_B/ashish/PS_Datasets/Light_Stage_Gallery')
    elif args.benchmark == 'DiLiGenT102':
        from dataset.Apple_Gourd import Apple_Gourd
        test_set = Apple_Gourd(args,'/media/aalok/Zone_B/ashish/PS_Datasets/DiLiGenT102')
    elif args.benchmark == 'DiLiGenT_DH':
        from dataset.DiLiGenT_DH import DiLiGenT_DH
        test_set = DiLiGenT_DH(args, 'test')
    elif args.benchmark == 'DiLiGenT102_DH':
        from dataset.Apple_Gourd_DH import Apple_Gourd_DH
        test_set = Apple_Gourd_DH(args,'/media/aalok/Zone_B/ashish/PS_Datasets/DiLiGenT102')
    else:
        raise Exception('Unknown benchmark')

    print('\t Found Benchmark Data: %d samples' % (len(test_set)))
    print('\t Test Batch %d' % (args.test_batch))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch,
                        num_workers=args.workers, pin_memory=args.cuda, shuffle=False)
    return test_loader
