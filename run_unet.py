import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torch.optim import Adam
import numpy as np
from pathlib import Path
from diffusion_scripts import *
from datasets import load_dataset
from torch.utils.data import DataLoader

import dhn
import os
from torchsummary import summary
import wandb
import time
PYTORCH_ENABLE_MPS_FALLBACK=1
class Manager:
    def __init__(self, epochs) -> None:
        dataset = load_dataset("fashion_mnist")
        self.image_size = 28
        self.channels = 1
        self.batch_size = 64
        self.epochs = epochs
        self.transformed_dataset = dataset.with_transform(transforms).remove_columns("label")
        self.dataloader = DataLoader(self.transformed_dataset["train"], batch_size=self.batch_size, num_workers=4, shuffle=True)

        self.device = get_device()
        
        self.lr = 1e-3
        hn_mult = 2

        x = torch.randn(1, 1, 28, 28)
        self.model = Unet_Model(
            dim=self.image_size,
            channels=self.channels,
            dim_mults=(1,2,4,)
        )
        summary(self.model)
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)

        self.results_folder = Path("./testing/unet/train_out")
        self.test_out = Path("./testing/unet/test_out")
        self.results_folder.mkdir(exist_ok = True)
        self.test_out.mkdir(exist_ok=True)
        self.save_and_sample_every = 100000

        watermark = "{}_lr{}".format("et", self.lr)
        wandb.init(project="diffusion_testing",
        name=watermark)
        wandb.watch(self.model)
    
    def train(self, epochs=None):
        if epochs == None:
            epochs = self.epochs
        samples_seen = 0
        best_loss = None
        milestone = 0
        for epoch in range(epochs):
            start = time.time()
            print("epoch: {}".format(str(epoch)))
            current_loss = None
            total_loss = 0
            for step, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                batch_size = batch["pixel_values"].shape[0]
                batch = batch["pixel_values"].to(self.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, timesteps, (batch_size,), device=self.device).long()

                loss = p_losses(self.model, batch, t, loss_type="huber")
                current_loss = loss.item()
                total_loss += current_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                # save generated images
                #if samples_seen > self.save_and_sample_every:
                #i had image out here
                
                progress_bar(step, len(self.dataloader), 'Loss: %.3f' % (total_loss/(step+1),))
                samples_seen += batch_size

            wandb.log({'epoch': epoch, 'train_loss': total_loss/(samples_seen/self.batch_size), "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time": time.time()-start})
            
            samples_seen = 0
            milestone += 1
            print("saving images")
            all_images = torch.Tensor(sample(self.model, image_size=self.image_size,batch_size=1, channels=self.channels))
            all_images = (all_images + 1) * 0.5
            all_images = all_images.squeeze(1)
            save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = 10)
                
            if best_loss==None or best_loss>current_loss:
                best_loss = current_loss
                print('Saving..')
                state = {"model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()}
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './testing/unet/checkpoint/unet-1ckpt.t7')
            self.scheduler.step()
    def test(self):
        '''
        hard coded, fix it
        '''
        all_images = torch.Tensor(sample(self.model, image_size=self.image_size,batch_size=1, channels=self.channels))
        all_images = (all_images + 1) * 0.5
        all_images = all_images.squeeze(1)
        save_image(all_images, str(self.test_out / f'test_out.png'), nrow = 10)
    
    def load_ckpt(self):
        '''
        this is hard coded, fix it
        '''
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./testing/unet/checkpoint/{}'.format("unet-1ckpt.t7"))
        self.model.load_state_dict(checkpoint['model'])
if __name__ == '__main__':
    manager = Manager(epochs=50)
    #manager.load_ckpt()
    manager.train()
    #manager.test()