import os, sys
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        #self.automatic_optimization = False
        self.save_hyperparameters()

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams['hparams'].chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams['hparams'].chunk],
                            self.hparams['hparams'].N_samples,
                            self.hparams['hparams'].use_disp,
                            self.hparams['hparams'].perturb,
                            self.hparams['hparams'].noise_std,
                            self.hparams['hparams'].N_importance,
                            self.hparams['hparams'].chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]
        
        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams['hparams'].dataset_name]
        kwargs = {'root_dir': self.hparams['hparams'].root_dir,
                  'img_wh': tuple(self.hparams['hparams'].img_wh),
                  'num_images': self.hparams['hparams'].num_images}
        if self.hparams['hparams'].dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams['hparams'].spheric_poses
            kwargs['val_num'] = self.hparams['hparams'].num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams['hparams'], self.models)
        scheduler = get_scheduler(self.hparams['hparams'], self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams['hparams'].batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        ## grab batch that corresponds to view2 from self
        ## if batch_nb == 0
        ## and plot
        """
        if batch_nb == 0:
            W, H = self.hparams['hparams'].img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)
                                               """

        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        targets = {'rgb': rgbs, 'depth': rays[:, 8], 'dirs': rays[:, 3:6], 'depth_uncertainty': rays[:, 9]}
        
        #anneal learning rate after 500 steps, reaching its max after another 5000 steps
        
        lambda_blend = min(((self.global_step > 1000)*(self.global_step-1000)/500), 1)
        #else: lambda_depth = 0
        loss_dict = self.loss(results, targets)
        loss = self.hparams['hparams'].lambda_rgb * loss_dict['rgb'] + \
            lambda_blend * (
               self.hparams['hparams'].lambda_orientation * loss_dict['orientation'] + \
               self.hparams['hparams'].lambda_depth * loss_dict['depth'] + \
               self.hparams['hparams'].lambda_distortion * loss_dict['distortion']
        )
                             
        typ = 'fine' if 'rgb_fine' in results else 'coarse'        
        self.log('train/loss',loss)

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            self.log('train/psnr', psnr_, prog_bar=True)
            self.log('train/orientation',loss_dict['orientation'])
            self.log('train/depth', loss_dict['depth'])
            self.log('train/rgb', loss_dict['rgb'])
            self.log('train/distortion', loss_dict['distortion'])
            self.log('lr',get_learning_rate(self.optimizer))   

        return loss

        
    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)

        targets = {'rgb': rgbs, 'depth': rays[:, 8], 'dirs': rays[:, 3:6], 'depth_uncertainty': rays[:, 9]}
        loss_dict = self.loss(results, targets)

        typ = 'fine' if 'rgb_fine' in results else 'coarse'        
        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)            

        if batch_nb == 0:
            W, H = self.hparams['hparams'].img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        loss_dict['val/loss'] = loss_dict['rgb']
        loss_dict['val/psnr'] = psnr_
        return loss_dict

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['rgb'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val/psnr'] for x in outputs]).mean()
        mean_depth = torch.stack([x['depth'] for x in outputs]).mean()
        mean_rgb = torch.stack([x['rgb'] for x in outputs]).mean()
        #mean_orientation = torch.stack([x['orientation'] for x in outputs]).mean()
        #mean_distortion = torch.stack([x['distortion'] for x in outputs]).mean()

        self.log('val/loss',mean_loss, prog_bar=True)
        self.log('val/psnr',mean_psnr, prog_bar=True)
        self.log('val/depth', mean_depth)
        self.log('val/rgb', mean_rgb)
        #self.log('val/orientation', mean_orientation)
        #self.log('train/distortion', mean_distortion)

        return {'val/loss': mean_loss}


if __name__ == '__main__':
    seed_everything(42424, workers=True) #9458 #42424
    hparams = get_opts()
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                                          monitor='val/loss',
                                          filename='epoch{epoch:02d}-val_loss{val/loss:.4f}',
                                          auto_insert_metric_name=False,                                          
                                          mode='min',
                                          save_top_k=5,)
    #cback_earlystopping = EarlyStopping()

    logger = TensorBoardLogger(
        save_dir="logs",
        name=hparams.exp_name
    )
    callbacks = [checkpoint_callback]
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      gpus=hparams.num_gpus,
                      num_sanity_val_steps=0,
                      benchmark=True,
                      val_check_interval=1000)

    trainer.fit(system)