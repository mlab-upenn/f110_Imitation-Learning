import argparse
import time

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *

#      0.109      0.297       0.15      0.126       7.04      1.666      4.062     0.1845       42.6       3.34      12.61      8.338     0.2705      0.001         -4        0.9     0.0005   320 giou + best_anchor False
#      0.223      0.218      0.138      0.189       9.28      1.153      4.376    0.08263      24.28       3.05      20.93      2.842     0.2759   0.001357     -5.036     0.9158  0.0005722   mAP/F1 - 50/50 weighting
#      0.231      0.215      0.135      0.191       9.51      1.432      3.007    0.06082      24.87      3.477      24.13      2.802     0.3436   0.001127     -5.036     0.9232  0.0005874
#      0.246      0.194      0.128      0.192       8.12      1.101      3.954     0.0817      22.83      3.967      19.83      1.779     0.3352   0.000895     -5.036     0.9238  0.0007973
# 0.242	0.296	0.196	0.231	5.67	0.8541	4.286	0.1539	21.61	1.957	22.9	2.894	0.3689	0.001844	-4	0.913	0.000467  # ha
# 0.298	0.244	0.167	0.247	4.99	0.8896	4.067	0.1694	21.41	2.033	25.61	1.783	0.4115	0.00128	    -4	0.950	0.000377  # hb
# 0.268	0.268	0.178	0.240	4.36	1.104	5.596	0.2087	14.47	2.599	16.27	2.406	0.4114	0.001585	-4	0.950	0.000524  # hc
# 0.161	0.327	0.190	0.193	7.82	1.153	4.062	0.1845	24.28	3.05	20.93	2.842	0.2759	0.001357	-4	0.916	0.000572  # hd 320 --epochs 2


# Training hyperparameters a
hyp = {'giou': 0.8541,  # giou loss gain
       'xy': 4.062,  # xy loss gain
       'wh': 0.1845,  # wh loss gain
       'cls': 21.61,  # cls loss gain
       'cls_pw': 1.957,  # cls BCELoss positive_weight
       'obj': 22.9,  # obj loss gain
       'obj_pw': 2.894,  # obj BCELoss positive_weight
       'iou_t': 0.3689,  # iou target-anchor training threshold
       'lr0': 0.001844,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.913,  # SGD momentum
       'weight_decay': 0.000467}  # optimizer weight decay


# Training hyperparameters d
# hyp = {'giou': 1.153,  # giou loss gain
#        'xy': 4.062,  # xy loss gain
#        'wh': 0.1845,  # wh loss gain
#        'cls': 24.28,  # cls loss gain
#        'cls_pw': 3.05,  # cls BCELoss positive_weight
#        'obj': 20.93,  # obj loss gain
#        'obj_pw': 2.842,  # obj BCELoss positive_weight
#        'iou_t': 0.2759,  # iou target-anchor training threshold
#        'lr0': 0.001357,  # initial learning rate
#        'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
#        'momentum': 0.916,  # SGD momentum
#        'weight_decay': 0.000572}  # optimizer weight decay


def train(cfg,
          data_cfg,
          img_size=416,
          epochs=100,  # 500200 batches at bs 16, 117263 images = 273 epochs
          batch_size=16,
          accumulate=4):  # effective bs = batch_size * accumulate = 8 * 8 = 64
    # Initialize
    init_seeds()
    weights = 'weights' + os.sep
    last = weights + 'last.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device()
    multi_scale = opt.multi_scale

    if multi_scale:
        img_size_min = round(img_size / 32 / 1.5)
        img_size_max = round(img_size / 32 * 1.5)
        img_size = img_size_max * 32  # initiate with maximum multi_scale size

    # Configure run
    data_dict = parse_data_cfg(data_cfg)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Darknet(cfg).to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = 0.0
    if opt.resume or opt.transfer:  # Load previously saved model
        if opt.transfer:  # Transfer learning
            nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
            chkpt = torch.load(weights + 'yolov3-spp.pt', map_location=device)
            model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                                  strict=False)

            for p in model.parameters():
                p.requires_grad = True if p.shape[0] == nf else False

        else:  # resume from last.pt
            if opt.bucket:
                os.system('gsutil cp gs://%s/last.pt %s' % (opt.bucket, last))  # download from bucket
            chkpt = torch.load(last, map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])

        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        if chkpt['training_results'] is not None:
            with open('results.txt', 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    else:  # Initialize model with backbone (optional)
        if '-tiny.cfg' in cfg:
            cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
        else:
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

        # Remove old results
        for f in glob.glob('*_batch*.jpg') + glob.glob('results.txt'):
            os.remove(f)

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in (0.8, 0.9)], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  rect=opt.rect)  # rectangular training

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank

        model = torch.nn.parallel.DistributedDataParallel(model)
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    # Mixed precision training https://github.com/NVIDIA/apex
    mixed_precision = True
    if mixed_precision:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
        except:  # not installed: install help: https://github.com/NVIDIA/apex/issues/259
            mixed_precision = False

    # Start training
    model.hyp = hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model_info(model, report='summary')  # 'full' or 'summary'
    nb = len(dataloader)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n' + '%10s' * 9) %
              ('Epoch', 'gpu_mem', 'GIoU/xy', 'wh', 'obj', 'cls', 'total', 'targets', 'img_size'))

        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        freeze_backbone = False
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        # # Update image weights (optional)
        # w = model.class_weights.cpu().numpy() * (1 - maps)  # class weights
        # image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
        # dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # random weighted index

        mloss = torch.zeros(5).to(device)  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Multi-Scale training TODO: short-side to 32-multiple https://github.com/ultralytics/yolov3/issues/358
            if multi_scale:
                if (i + nb * epoch) / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.choice(range(img_size_min, img_size_max + 1)) * 32
                    # print('img_size = %g' % img_size)
                scale_factor = img_size / max(imgs.shape[-2:])
                imgs = F.interpolate(imgs, scale_factor=scale_factor, mode='bilinear', align_corners=False)

            # Plot images with bounding boxes
            if epoch == 0 and i == 0:
                plot_images(imgs=imgs, targets=targets, paths=paths, fname='train_batch%g.jpg' % i)

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = hyp['lr0'] * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model, giou_loss=not opt.xywh)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            pbar.set_description(s)  # print(s)

        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if not (opt.notest or (opt.nosave and epoch < 10)) or epoch == epochs - 1:
            with torch.no_grad():
                results, maps = test.test(cfg, data_cfg, batch_size=batch_size, img_size=opt.img_size, model=model,
                                          conf_thres=0.1)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        # Update best map
        fitness = results[2]
        if fitness > best_fitness:
            best_fitness = fitness

        # Save training results
        save = (not opt.nosave) or ((not opt.evolve) and (epoch == epochs - 1))
        if save:
            with open('results.txt', 'r') as file:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': file.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)
            if opt.bucket:
                os.system('gsutil cp %s gs://%s' % (last, opt.bucket))  # upload to bucket

            # Save best checkpoint
            if best_fitness == fitness:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

    # Report time
    print('%g epochs completed in %.3f hours.' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    return results


def print_mutation(hyp, results):
    # Write mutation results
    a = '%11s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%11.4g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%11.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if opt.bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt
        with open('evolve.txt', 'a') as f:  # append result
            f.write(c + b + '\n')
        os.system('gsutil cp evolve.txt gs://%s' % opt.bucket)  # upload evolve.txt
    else:
        with open('evolve.txt', 'a') as f:
            f.write(c + b + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--accumulate', type=int, default=4, help='number of batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco_64img.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='train at (1/1.5)x - 1.5x sizes')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--xywh', action='store_true', help='use xywh loss instead of GIoU loss')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    opt = parser.parse_args()
    print(opt)

    if opt.evolve:
        opt.notest = True  # only test final epoch
        opt.nosave = True  # only save final checkpoint

    # Train
    results = train(opt.cfg,
                    opt.data_cfg,
                    img_size=opt.img_size,
                    epochs=opt.epochs,
                    batch_size=opt.batch_size,
                    accumulate=opt.accumulate)

    # Evolve hyperparameters (optional)
    if opt.evolve:
        gen = 1000  # generations to evolve
        print_mutation(hyp, results)  # Write mutation results

        for _ in range(gen):
            # Get best hyperparameters
            x = np.loadtxt('evolve.txt', ndmin=2)
            fitness = x[:, 2] * 0.5 + x[:, 3] * 0.5  # fitness as weighted combination of mAP and F1
            x = x[fitness.argmax()]  # select best fitness hyps
            for i, k in enumerate(hyp.keys()):
                hyp[k] = x[i + 5]

            # Mutate
            init_seeds(seed=int(time.time()))
            s = [.15, .15, .15, .15, .15, .15, .15, .15, .15, .00, .05, .10]  # fractional sigmas
            for i, k in enumerate(hyp.keys()):
                x = (np.random.randn(1) * s[i] + 1) ** 2.0  # plt.hist(x.ravel(), 300)
                hyp[k] *= float(x)  # vary by 20% 1sigma

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay']
            limits = [(1e-4, 1e-2), (0.00, 0.70), (0.60, 0.95), (0, 0.01)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train(opt.cfg,
                            opt.data_cfg,
                            img_size=opt.img_size,
                            epochs=opt.epochs,
                            batch_size=opt.batch_size,
                            accumulate=opt.accumulate)

            # Write mutation results
            print_mutation(hyp, results)

            # # Plot results
            # import numpy as np
            # import matplotlib.pyplot as plt
            # a = np.loadtxt('evolve_1000val.txt')
            # x = a[:, 2] * a[:, 3]  # metric = mAP * F1
            # weights = (x - x.min()) ** 2
            # fig = plt.figure(figsize=(14, 7))
            # for i in range(len(hyp)):
            #     y = a[:, i + 5]
            #     mu = (y * weights).sum() / weights.sum()
            #     plt.subplot(2, 5, i+1)
            #     plt.plot(x.max(), mu, 'o')
            #     plt.plot(x, y, '.')
            #     print(list(hyp.keys())[i],'%.4g' % mu)
