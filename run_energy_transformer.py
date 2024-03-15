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
import argparse
import dhn
import os
from torchsummary import summary
import wandb
import time

#PYTORCH_ENABLE_MPS_FALLBACK=1

class Manager:
    def __init__(self, args) -> None:
        dataset = load_dataset("imagenet-1k", split='train', streaming=args.stream_data, use_auth_token=True)
        self.image_size = 128
        self.batch_size = args.bs
        self.epochs = args.epochs
        self.num_training_images = 1281167
        self.num_batches = self.num_training_images//self.batch_size
        if args.stream_data:
            if args.bw:
                self.channels = 1
                self.transformed_dataset = CustomImageDataset(dataset=dataset, transform=transform, bw=True)
            else:
                self.channels = 3
                self.transformed_dataset = CustomImageDataset(dataset=dataset, transform=transform_color, bw=False)
        else:
            if args.bw:
                self.channels = 1
                self.transformed_dataset = dataset.with_transform(transforms).remove_columns("label")
            else:
                self.channels = 3
                self.transformed_dataset = dataset.with_transform(transforms_color).remove_columns("label")
        #self.transformed_dataset = dataset.map(transforms, remove_columns=["label"], batched=True)
        self.dataloader = DataLoader(dataset=self.transformed_dataset,batch_size=self.batch_size, num_workers=args.nworkers if not args.stream_data else min(args.nworkers,dataset.n_shards), shuffle=False)

        self.device = get_device(args.xla)
        
        self.lr = args.lr
        hn_mult = args.hn_mult
        patch_size = args.patch_size
        tkn_dim = args.tkn_dim
        qk_dim = args.qk_dim
        nheads = args.nheads
        out_dim = None
        time_steps = 1
        blocks = 1

        x = torch.randn(1, self.channels, self.image_size, self.image_size)
        patch_fn = Patch(dim=patch_size, n=self.image_size)
        self.model = ET(
            x,
            patch_fn,
            out_dim,
            tkn_dim,
            qk_dim,
            nheads,
            hn_mult,
            time_steps=time_steps,
            blocks=blocks
        )
        summary(self.model)
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)

        self.results_folder = Path("./testing/energy_transformer/train_out")
        self.test_out = Path("./testing/energy_transformer/test_out")
        self.results_folder.mkdir(exist_ok = True)
        self.test_out.mkdir(exist_ok=True)
        self.save_and_sample_every = 100000

        watermark = "{}_lr{}_heads{}_hnmult{}".format("energy_transformer", args.lr, args.nheads, args.hn_mult)
        wandb.init(project="imagenet1-k_energy",
                name=watermark)
        wandb.config.update(args)
    
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
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                # save generated images
                #if samples_seen > self.save_and_sample_every:
                #i had image out here
                
                #progress_bar(step, len(self.dataloader), 'Loss: %.3f' % (total_loss/(step+1),))
                progress_bar(step, self.num_batches, 'Loss: %.3f' % (total_loss/(step+1),))
                samples_seen += batch_size

            wandb.log({'epoch': epoch, 'train_loss': total_loss/(samples_seen/self.batch_size), "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time": time.time()-start})
            
            samples_seen = 0
            milestone += 1
            print("saving images")
            self.save_img(str(self.results_folder / f'sample-{milestone}.png'), everyn=100)
                
            if best_loss==None or best_loss>current_loss:
                best_loss = current_loss
                print('Saving..')
                state = {"model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()}
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './testing/energy_transformer/checkpoint/et-patch_{}-hnmult_{}.t7'.format(args.patch_size, args.hnmult))
            self.scheduler.step()
    
    def save_img(self, path, nrow=10, everyn=0):
        all_images = torch.Tensor(sample(self.model, image_size=self.image_size,batch_size=1, channels=self.channels))
        all_images = (all_images + 1) * 0.5
        all_images = all_images.squeeze(1)
        if everyn>0:
            all_images = all_images[::everyn]
        save_image(all_images, path, nrow = nrow)

    def test(self):
        '''
        hard coded, fix it
        '''
        self.save_img(str(self.test_out / f'test_out.png'), everyn=100)
    
    def load_ckpt(self, ckpt):
        '''
        this is hard coded, fix it
        '''
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./testing/energy_transformer/checkpoint/{}'.format(ckpt))
        self.model.load_state_dict(checkpoint['model'])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Energy Transformer Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--bs', type=int, default='32')
    parser.add_argument('--bw', action='store_true', help="black&white or RGB")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--hn_mult", default=4, type=int, help="hopfield multiplier")
    parser.add_argument("--patch_size", default=4, type=int, help="number of pixles per dimension of patch")
    parser.add_argument("--tkn_dim", default=64, type=int, help="Token dim for Energy Transformer")
    parser.add_argument("--qk_dim", default=64, type=int, help="QK dim for Energy Transformer")
    parser.add_argument("--nheads", default=4, type=int, help="num of heads for Energy Transformer")
    parser.add_argument("--ckpt", default="ckpt.t7", help="Name of checkpoint file")
    parser.add_argument("--test", action="store_true", help="If set, run test() instead of train(), must be used in conjunction with --resume")
    parser.add_argument("--stream_data", action="store_true", help="stream hugging face dataset")
    parser.add_argument("--nworkers", default=0, type=int, help="num of workers for dataloader, if streaming, should be set to num shards")
    parser.add_argument("--shuffle", action="store_true", help="shuffle dataset, only works if not streaming")
    parser.add_argument("--xla", action="store_true", help="Enable XLA")
    args = parser.parse_args()

    manager = Manager(args=args)
    if args.resume:
        manager.load_ckpt()
    if args.test:
        manager.test()
    else:
        manager.train()