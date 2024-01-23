import torch

import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
if __name__ == "__main__":

       
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
   
    
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
   

    #shot_SR=torch.zeros(1,1151,1641)
    #shot_HR=torch.zeros(1,1151,1641)
    #shot_LR=torch.zeros(1,1151,1641)
    #shot_INF=torch.zeros(1,1151,1641)
    #shot_gap1=torch.zeros(1,1151,1641)
    #shot_gap2=torch.zeros(1,1151,1641)
 
    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':

            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
           
        elif phase == 'val':
            print(phase)
            val_set = Data.create_dataset(dataset_opt, phase)
            
            val_batchszie=dataset_opt['batch_size']
            me=dataset_opt['overlap']+1
            overlap=dataset_opt['overlap']
            nny=dataset_opt['nny']
            nnx=dataset_opt['nnx']
            shot_SR=torch.zeros(me,dataset_opt['nny'],dataset_opt['nnx'])
            shot_HR=torch.zeros(me,dataset_opt['nny'],dataset_opt['nnx'])
            shot_LR=torch.zeros(me,dataset_opt['nny'],dataset_opt['nnx'])
            shot_INF=torch.zeros(me,dataset_opt['nny'],dataset_opt['nnx'])
            shot_gap1=torch.zeros(me,dataset_opt['nny'],dataset_opt['nnx'])
            shot_gap2=torch.zeros(me,dataset_opt['nny'],dataset_opt['nnx'])
            ddy=dataset_opt['nyy']//128
            ddx=dataset_opt['nxx']//128
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
      
    logger.info('Initial Dataset Finished')
    

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    to_pil = transforms.ToPILImage(mode='L')

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):

                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                logs = diffusion.get_current_log()
                if current_step % opt['train']['print_freq'] == 0:
                    
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)
                for k, v in logs.items():                        
                  f=open('./train_loss.txt','a') 
                  f.write('\n'+str(v))
                
                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    times=0
                    os.makedirs(result_path, exist_ok=True)
          
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        
                        for i in range(0,ddy):
                            for j in range(0,1):
                              ec2=(times//(ddy*ddx))
                              ec=(times//(ddy*ddx))%overlap
                              ec=ec+1
                              ec2=ec2+1
                              co = np.loadtxt('./coe_{}.txt'.format(ec2),dtype=np.float32)
                              x=(times//ddy)%ddx

                              p1=ddy*x+i
                              p=ddy*j+i
                              print("times=",times,"x=",x,"ec=",ec,'ec2=',ec2,"p=",p)                              
                            
                              
                              
                              siz=32
                              shot_SR[ec,ec*siz*1+128*i:ec*siz*1+128*(i+1),ec*siz*1+128*(x)+128*j:ec*siz*1+128*(x)+128*(j+1)]=visuals['SR'][p,:,:]
                              
                              shot_HR[ec,ec*siz*1+128*i:ec*siz*1+128*(i+1),ec*siz*1+128*(x)+128*j:ec*siz*1+128*(x)+128*(j+1)]=visuals['HR'][p,:,:]
                              shot_LR[ec,ec*siz*1+128*i:ec*siz*1+128*(i+1),ec*siz*1+128*(x)+128*j:ec*siz*1+128*(x)+128*(j+1)]=visuals['LR'][p,:,:]
                              shot_INF[ec,ec*siz*1+128*i:ec*siz*1+128*(i+1),ec*siz*1 +128*(x)+128*j:ec*siz*1+128*(x)+128*(j+1)]=visuals['INF'][p,:,:]  
                              times=times+1
                    file = open('{}/whole_{}.dat'.format(result_path,ec), 'wb')
          
                    sumSR_img = Metrics.tensor2img(torch.sum(shot_SR,dim=0)/(ec+1))
                    sumINF_img = Metrics.tensor2img(torch.sum(shot_INF,dim=0)/(ec+1))
                    
                    Metrics.save_img(
                          sumSR_img, '{}/sumSR.png'.format(result_path))
                    Metrics.save_img(
                          sumINF_img, '{}/sumINF.png'.format(result_path))
                    tem=(torch.sum(shot_SR,dim=0)/(ec+1)).numpy()
                    tempr=np.copy(tem)
                    file.write(tempr)
                    print(tempr.shape)     
                    #sr_img = Metrics.tensor2img(shot_SR)  # uint8
                    #hr_img = Metrics.tensor2img(shot_HR)  # uint8
                    #lr_img = Metrics.tensor2img(shot_LR)  # uint8
                    #fake_img = Metrics.tensor2img(shot_INF)  # uint8
                    #gap_img1 = Metrics.tensor2img(torch.cat([shot_HR,shot_HR-shot_SR],dim=2))
                    #gap_img2 = Metrics.tensor2img(torch.cat([shot_INF,shot_INF-shot_SR],dim=2)) # uint8
                    #print(visuals['SR'].shape,visuals['HR'].shape,visuals['INF'].shape)
                    #print(sr_img.shape,hr_img.shape,lr_img.shape,fake_img.shape)
                    
                    # generation
                    
                    #Metrics.save_img(
                    #    hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                    #Metrics.save_img(
                    #    sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                    #Metrics.save_img(
                    #    lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                   # Metrics.save_img(
                    #   fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                  # Metrics.save_img(
                     #   gap_img1, '{}/{}_{}_gap1.png'.format(result_path, current_step, idx))
                   # Metrics.save_img(
                    #    gap_img2, '{}/{}_{}_gap2.png'.format(result_path, current_step, idx))
                        #tb_logger.add_image(
                            #'Iter_{}'.format(current_step),
                            #np.expand_dims(np.concatenate((fake_img, sr_img, hr_img), axis=1),axis=0),idx)
                        #tb_logger.add_image(
                            #'Iter_{}'.format(current_step),
                           # np.transpose(np.concatenate((fake_img, sr_img, hr_img), axis=1), [2,0, 1]),idx)                           
                    avg_psnr += Metrics.calculate_psnr( sumSR_img,  sumINF_img)

                    #if wandb_logger:
                    #    wandb_logger.log_image(
                    #        f'validation_{idx}', 
                    #        np.concatenate((fake_img, sr_img, hr_img), axis=1)
                   #     )
                    #print(idx)
                    avg_psnr = avg_psnr / idx
                        
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    with open('./train_acc.txt','a') as train_acc:
                       train_acc.write('\n'+str(avg_psnr))
                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 1
        ec=0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        current_step=0
        times=0
     
        for _,  val_data in enumerate(val_loader):
            
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals()
          
         
            for k in range(0,1):
              for i in range(0,ddy):
                 for j in range(0,1):
                    
                    ec1=(times//(ddy*ddx))%overlap
                    ec2=(times//(ddy*ddx))
                    ec2=ec2+1
                    ec1=ec1+1
                  
                    co = np.loadtxt('./coe_{}.txt'.format(ec2),dtype=np.float32)
                    x=(times//ddy)%ddx
                    y=(times%ddy)              
                    p=ddy*j+i
                    p1=ddy*x+i
                    print("times=",times,"x=",x,"y=",y,"ec1=",ec1)
                 
                    
                    siz=8
              
                    #shot_SR[ec,ec*32*1+128*i:ec*32*1+128*(i+1),ec*32*1+128*(x)+128*j:ec*32*1+128*(x)+128*(j+1)]=co[p1]*visuals['SR'][p,:,:]
                      
                    #shot_HR[ec,ec*32*1+128*i:ec*32*1+128*(i+1),ec*32*1+128*(x)+128*j:ec*32*1+128*(x)+128*(j+1)]=co[p1]*visuals['HR'][p,:,:]
                    #shot_LR[ec,ec*32*1+128*i:ec*32*1+128*(i+1),ec*32*1+128*(x)+128*j:ec*32*1+128*(x)+128*(j+1)]=co[p1]*visuals['LR'][p,:,:]
                    #shot_INF[ec,ec*32*1+128*i:ec*32*1+128*(i+1),ec*32*1 +128*(x)+128*j:ec*32*1+128*(x)+128*(j+1)]=co[p1]*visuals['INF'][p,:,:]
                    times=times+1
                    print(visuals['SR'][p,:,:].shape)                    
                    #shot_SR[ec,ec*32*1+128*i:ec*32*1+128*(i+1),ec*32*1+128*(x)+128*j:ec*32*1+128*(x)+128*(j+1)]=visuals['SR'][p,:,:]
                    shot_SR[ec1,siz*ec1+128*y:siz*ec1+128*(y)+128,siz*ec1+128*x:siz*ec1+128*(x)+128]=co[p1]*visuals['SR'][p,:,:]
                    shot_HR[ec1,siz*ec1+128*y:siz*ec1+128*(y)+128,siz*ec1+128*x:siz*ec1+128*(x)+128]=co[p1]*visuals['HR'][p,:,:]
                    shot_LR[ec1,siz*ec1+128*y:siz*ec1+128*(y)+128,siz*ec1+128*x:siz*ec1+128*(x)+128]=co[p1]*visuals['LR'][p,:,:]
                    shot_INF[ec1,siz*ec1+128*y:siz*ec1+128*(y)+128,siz*ec1+128*x:siz*ec1+128*(x)+128]=co[p1]*visuals['INF'][p,:,:]  
          
                #print(visuals['SR'].shape,visuals['HR'].shape,visuals['INF'].shape)
                #print(sr_img.shape,hr_img.shape,lr_img.shape,fake_img.shape)
                  
                # generation
               
        
        
          # uint8
            if idx%(overlap*(ddx))==0:
           
      
              file = open('{}/SR_{}.dat'.format(result_path,idx//(overlap*ddx)), 'wb')
              
              sumSR_img = Metrics.tensor2img(torch.sum(shot_SR,dim=0)/(ec+1))
              sumINF_img = Metrics.tensor2img(torch.sum(shot_INF,dim=0)/(ec+1))
              sumHR_img = Metrics.tensor2img(torch.sum(shot_HR,dim=0)/(ec+1))
              sumLR_img = Metrics.tensor2img(torch.sum(shot_HR,dim=0)/(ec+1))
              Metrics.save_img(
              sumSR_img, '{}/sumSR.png'.format(result_path))
              Metrics.save_img(
              sumINF_img, '{}/sumINF.png'.format(result_path))
              Metrics.save_img(
              sumHR_img, '{}/sumHR.png'.format(result_path))
              Metrics.save_img(
              sumLR_img, '{}/sumLR.png'.format(result_path))                     
              tempr=np.ascontiguousarray(np.transpose((torch.sum(shot_SR[:,128:128+nny-128*3,128:128+nnx-128*3],dim=0)/(ec+1)).numpy(),axes=(1,0)))
              
              file.write(tempr)
              file = open('{}/INF_{}.dat'.format(result_path,idx//(overlap*ddx)), 'wb')
              tempr=np.ascontiguousarray(np.transpose((torch.sum(shot_INF[:,128:128+nny-128*3,128:128+nnx-128*3],dim=0)/(ec+1)).numpy(),axes=(1,0)))
              file.write(tempr)
              shot_SR[:,:,:]=0
              shot_HR[:,:,:]=0
              shot_LR[:,:,:]=0
              shot_INF[:,:,:]=0
            idx += 1
                    #file = open('./tempdata/sr_16_128'+ '/slices_' + str(3)+'patch_'+str(7*64+i*128)+'_'+str(j*128)+'.dat', 'wb')
                  #file.write(visuals['SR'][p,:,0:128,0:128].numpy())
            #hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            #lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            #fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
            #file = open('./tempdata/whole.dat', 'wb')
            #file.write(shot_SR.numpy())
            #sr_img_mode = 'grid'
            #if sr_img_mode == 'single':
                # single img series
            #sr_img = Metrics.tensor2img(visuals['SR'])   # uint8
            #    sample_num = sr_img.shape[0]
            #    for iter in range(0, sample_num):
            #        Metrics.save_img(
            #            Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            #else:
                # grid img
            #    sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            #    Metrics.save_img(
            #        sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
            #    Metrics.save_img(
            #        Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
            #Metrics.save_img(
                      #sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
            #Metrics.save_img(
                #hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            #Metrics.save_img(
                #lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            #Metrics.save_img(
                #fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            #eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            #eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            #avg_psnr += eval_psnr
            #avg_ssim += eval_ssim

            #if wandb_logger and opt['log_eval']:
            #    wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        #avg_psnr = avg_psnr / idx
        #avg_ssim = avg_ssim / idx

        # log
       # logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        #logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        #logger_val = logging.getLogger('val')  # validation logger
       # logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
        #    current_epoch, current_step, avg_psnr, avg_ssim))

        #if wandb_logger:
            #if opt['log_eval']:
             #   wandb_logger.log_eval_table()
           # wandb_logger.log_metrics({
            #    'PSNR': float(avg_psnr),
           #     'SSIM': float(avg_ssim)
            #})
