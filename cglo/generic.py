import torch
import random
import logging
import numpy as np
from math import log, ceil

import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.utils import save_image

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from cglo.utils.dataset import BasicDataset
from cglo.utils.misc import timer, export
from cglo.utils.hypersphere import projected_normal_sampling as pnsamp
from cglo.utils.hypersphere import projector as proj
from cglo.models.decoder import Decoder

DEVICE = torch.device('cuda')

def set_seed(seed:int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

class LossLog():

    def __init__(self):
        self.names = ['DEC']
        self.epoch = {n: 0  for n in self.names}
        self.log   = {n: [] for n in self.names}
        self.count = {n: 0  for n in self.names}

    def update(self, count, losses):
        for k, v in losses.items():
            self.count[k] += count
            self.epoch[k] += v * count

    def save(self):
        for k, v in self.epoch.items(): self.log[k].append(v / self.count[k])

    def reset(self):
        for k in self.epoch.keys():
            self.count[k] = 0
            self.epoch[k] = 0

class Runner():

    def __init__(self, latent_dim, image_size, first_layer):

        self.latent_dim  = latent_dim  # hypersphere dimension
        self.image_size  = image_size  # resolution to use for experiment
        self.first_layer = first_layer # decoder first layer channels number

    def ini_dataloader(self, batch_size, dataset_type=BasicDataset):
        data = dataset_type(self.bdd)
        return DataLoader(data, batch_size, pin_memory=True, shuffle=True)

    def ini_decoder(self):
        '''
            The current init returns a DCGAN-like decoder:

            - a linear layer map latent_dim to first_layer channels number
            - number of channels is cut by half from layer to layer
            - input is upscale by a factor 2 from layer to layer
            - incidently the output image_size fixes the required layers number
        '''
        # layers channels number list
        layers = [self.latent_dim] + [
            self.first_layer//2**i for i in range(ceil(log(self.image_size, 2)))
        ]
        self.decoder = DataParallel(Decoder(layers, self.image_size).to(DEVICE))

    def ini_latent(self):

        n = self.bdd.shape[0]

        # Random init
        latent = export(pnsamp([n, self.latent_dim]), DEVICE)

        if 'latent' in self.save:
            torch.save(
                self.latent,
                f'{self.save_path}/{self.run_name}_lat_ini.pt'
            )

    def ini_optim(self, optim_params, epoch=0):

        optim_params.setdefault('decoder', {})
        optim_params.setdefault('latent' , {})
        dec_prm = {'params': self.decoder.module.parameters()}
        lat_prm = {'params': self.latent}
        optim_params['decoder'].update(dec_prm)
        optim_params['latent' ].update(lat_prm)

        self.optimizer = self.f_optimizer(list(optim_params.values()), epoch)
        self.scheduler = self.f_scheduler(self.optimizer)

    @timer
    def epoch(self):

        for indices, truth in self.dataloader:

            # Minimum batch size for models with batch_norm layers is 2
            # With N-GPUs parallel computation that makes a 2*N minimum
            if len(indices) < 2 * torch.cuda.device_count(): continue

            self.optimizer.zero_grad()

            self.truth = truth.to(DEVICE)
            latent = proj(self.latent[indices])
            pred = self.decoder(latent)
            self.pred = pred.detach()
            dec_loss = self.f_loss(self.truth, pred)
            loss = dec_loss
            loss.backward()
            self.optimizer.step()

            self.loss.update(
                len(indices),
                {'DEC': dec_loss.item()}
            )

        self.loss.save()
        self.loss.reset()
        self.scheduler.step()

    def save_state(self):
        '''
            save items of runner given in self.save
        '''

        save = {}

        if 'loss' in self.save:
            save.update({'loss': self.loss.log})

        if 'pred' in self.save:
            save.update({'pred': self.pred[:self.max_save]})

        if 'examples' in self.save:
            # examples (i.e. images) are saved by sequence of length self.nrow:
            # nrow truths, nrow preds, nrow truths, ...
            # with a maximum of self.max_save examples

            ex = torch.cat([torch.zeros_like(self.truth)]*2)
            n_ex = ex.shape[0] if ex.shape[0]<=self.max_save else self.max_save
            n_save = n_ex // (2*self.nrow) * (2*self.nrow)

            # mask to alternate ground truth and pred
            mk = torch.arange(n_save)\
                .div(self.nrow, rounding_mode='floor') % 2 == 0

            ex = ex[:n_save]
            ex[mk], ex[~mk] = self.truth[:n_save//2], self.pred[:n_save//2]
            save.update({'exp': ex})

        if 'latent' in self.save:
            save.update({'lat': proj(self.latent.detach())})

        if 'decoder' in self.save:
            save.update({'dec': self.decoder.module.state_dict()})

        return save

    def save_to_path(self, suffix):

        for k, v in self.save_state().items():
            p = f'{self.save_path}/{self.run_name}_{k}_{suffix}'
            torch.save(v, f'{p}.pt')
            if k in ['exp', 'pred']: save_image(v, f'{p}.png', nrow=self.nrow)

    def loop(self, n_epochs, checkpoints):
        '''
            loop on epoch method
            dataloader may be updated between epochs depending on given f_batch
        '''

        # save batch size of previous epoch
        batch_mem = self.f_batch(0)
        for e in range(n_epochs):

            # if batch size changes, instantiate new dataloader/optimizer
            if self.f_batch(e) != batch_mem:
                self.dataloader = self.ini_dataloader(self.f_batch(e))
                #self.ini_optim(self.optim_params, epoch=e)
                batch_mem = self.f_batch(e)

            self.epoch()

            # print losses
            to_print = [f'{k}: {v[-1]:.3e}' for k, v in self.loss.log.items()]
            logging.info('\t'.join([f'{e+1}:  '] + to_print))

            if (e+1) % checkpoints==0:
                logging.info(f'Saving checkpoint at epoch {e+1}')
                self.save_to_path(e+1)
            if self.loss.log['DEC'][-1] == np.min(self.loss.log['DEC']):
                logging.info('Saving best')
                self.save_to_path('best')

        logging.info('Saving last')
        self.save_to_path('last')

    @timer
    def run(self,
        run_name = 'exp',
        indices = None,
        # PATHS
        save_path = None,
        db_path = None,
        dec_path = None,
        lat_path = None,
        # INITS
        bdd_init = True,
        dec_init = True,
        lat_init = True,
        # LOADS
        dec_load = False,
        lat_load = False,
        # LOSSES
        f_loss = F.mse_loss,
        # LOOPS, BATCH, OPTIMIZER & SCHEDULER
        f_optimizer = lambda params, epoch: Adam(params),
        f_scheduler = lambda epoch: 1,
        f_batch = lambda epoch: 64,
        n_loop_epochs = 300,
        # CHECKPOINTS
        save = [],
        checkpoints = 100,
        max_save = 64,
        nrow = 4,
        # OPTIM
        **optim_params
    ):
        '''
            run experiment

            run_name:
            - experiment name that will be add to saved files

            indices:
            - indices to sample database, if None: full database

            xxx_path:
            - directory (save) or file (bdd, dec, lat) path
            - conditions on paths:
                * if xxx_load is set to True xxx_path must be given
                * if bdd_init is set to True db_path must be given
                * if save is not an empty list save_path must be given

            xxx_init:
            - True : init and replace if existing self.xxx
            - False: kept as is

            xxx_load:
            - True : load and replace if existing self.xxx
            - False: kept as is
            - note : if dec_load is set to True self.decoder must exists
                     or dec_init must be set to True

            loss:
            - f_loss: loss function used during training

            loops:
            - f_optimizer: optimizer to be used (torch.optim.optimizer)
            - f_scheduler: function to be used as LambdaLR
            - f_batch: function to control batch_size
            - n_loop_epochs: number of training epochs

            checkpoints:
            - save: lists items saved at each checkpoints, items available are:
                * loss: losses values
                * pred: predictions
                * examples: predictions and corresponding ground truths
                * latent: points coordinates in latent space
                * decoder: decoder weigths
            - max_save: max number of examples per save
            - nrow: examples series length, see self.save_state()

            optim:
            - optim_params: options that can be passed to the optimizer with key
                            indicating concerned parameters (latent or decoder):
                            ex: latent = {'lr': 1e-2}
        '''
        self.run_name = run_name
        self.save = save
        self.f_loss = f_loss
        self.f_scheduler = lambda x: LambdaLR(x, f_scheduler)
        self.f_optimizer = f_optimizer
        self.f_batch = f_batch
        self.save_path = save_path
        self.loss = LossLog()
        self.max_save = max_save
        self.nrow = nrow
        self.optim_params = optim_params

        if bdd_init:
            bdd = torch.load(db_path, map_location=torch.device('cpu'))
            if indices:
                bdd = bdd[indices]
            if self.image_size != bdd.shape[-1]:
                bdd = resize(bdd, [self.image_size]*2)
            self.bdd = bdd

        self.dataloader = self.ini_dataloader(self.f_batch(0))

        if dec_init: self.ini_decoder()

        if dec_load: self.decoder.module.load_state_dict(torch.load(dec_path))

        if lat_init: self.ini_latent()

        if lat_load: self.latent = torch.load(lat_path)

        self.ini_optim(optim_params)

        self.loop(n_loop_epochs, checkpoints)

