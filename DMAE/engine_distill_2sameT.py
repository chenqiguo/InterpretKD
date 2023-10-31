import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch_2sameT(model: torch.nn.Module, 
                    model_teacher_1: torch.nn.Module, model_teacher_2: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    model_teacher_1.eval()
    model_teacher_2.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    print(len(data_loader))
    
    # newly modified by Chenqi:
    
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)): # for other datasets
    #for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)): # for ImageNet-LT and CIFAR-LT
        # Note: for our custom dataloader_LT, the samples is a list contain: sample, label, path
        """
        print('****** debug 1 !')
        print(len(samples)) # 3
        print(samples)
        assert(False)
        """
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        if isinstance(samples, list):
            imgs = samples[0].to(device, non_blocking=True)
            heatmaps = samples[1].to(device, non_blocking=True)
        else:
            imgs = samples.to(device, non_blocking=True)
            heatmaps = None
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                latents_teacher_1, mask_1, ids_restore_1, ids_keep_1 = \
                    model_teacher_1.module.forward_encoder_customized(imgs, args.mask_ratio)
                teacher_prediction_1 = model_teacher_1.module.forward_decoder(latents_teacher_1[-1],
                                                                            ids_restore_1)  
                latents_teacher_2, mask_2, ids_restore_2, ids_keep_2 = \
                    model_teacher_2.module.forward_encoder_customized(imgs, args.mask_ratio)
                teacher_prediction_2 = model_teacher_2.module.forward_decoder(latents_teacher_2[-1],
                                                                              ids_restore_2)  
            """
            # for debug:
            print('********** debug 1 begin:')
            print('len(latents_teacher_1) = ' + str(len(latents_teacher_1)))
            print('latents_teacher_1[0].shape = ' + str(latents_teacher_1[0].shape))
            print('latents_teacher_1[1].shape = ' + str(latents_teacher_1[1].shape))
            print('mask_1.shape = ' + str(mask_1.shape))
            print('ids_restore_1.shape = ' + str(ids_restore_1.shape))
            print('ids_keep_1.shape = ' + str(ids_keep_1.shape))
            print('********** debug 1 end!')
            assert(False)
            """
            ## here we can use different strategies to "feature concate" for distillation:
            # ***** strategy 1:
            # for S, forward with T1 and T2 respectively,
            # then the loss should be the mean of their losses.
            
            loss_1, loss_distillation_embedding_1, _, _ = model(imgs, ids_keep_1, ids_restore_1, mask_1, teacher_prediction_1,
                                                            args.target_sum_weights, latents_teacher_1)
            loss_2, loss_distillation_embedding_2, _, _ = model(imgs, ids_keep_2, ids_restore_2, mask_2, teacher_prediction_2,
                                                            args.target_sum_weights, latents_teacher_2)
            
            loss_value_1 = loss_1.item()
            for loss_1_k, loss_1_v in loss_distillation_embedding_1.items():
                loss_1 += loss_1_v
            
            loss_value_2 = loss_2.item()
            for loss_2_k, loss_2_v in loss_distillation_embedding_2.items():
                loss_2 += loss_2_v
            
            loss_value = (loss_value_1+loss_value_2)/2
            loss = (loss_1+loss_2)/2
            
            """
            # for debug:
            print('********** debug 2 begin:')
            print('loss_distillation_embedding_1 = ' + str(loss_distillation_embedding_1))
            print('loss_distillation_embedding_2 = ' + str(loss_distillation_embedding_2))
            print('********** debug 2 end!')
            # conclusion: they have same key!:
            # loss_distillation_embedding_1 = {'align_block8': tensor(0.0339, device='cuda:0', grad_fn=<_DDPSinkBackward>)}
            assert(False)
            """


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if args.aligned_blks_indices is not None:
            loss_total = loss.item()
            loss_total_value_reduce = misc.all_reduce_mean(loss_total)
        
        # newly modified by Chenqi:
        if args.aligned_blks_indices is not None:
            loss_distillation_embedding = dict()
            for loss_k, loss_1_v in loss_distillation_embedding_1.items():
                loss_2_v = loss_distillation_embedding_2[loss_k]
                loss_v = (loss_1_v + loss_2_v) / 2
                loss_distillation_embedding[loss_k] = misc.all_reduce_mean(loss_v)
            

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            if args.aligned_blks_indices is not None:
                log_writer.add_scalar('train_loss_total', loss_total_value_reduce, epoch_1000x)
                for key, value in loss_distillation_embedding.items():
                    log_writer.add_scalar(f'distillation_loss/{key}', value, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}