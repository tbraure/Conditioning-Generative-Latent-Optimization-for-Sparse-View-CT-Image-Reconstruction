import torch.nn.functional as F
from cglo.generic import set_seed, Runner
import logging

L = 320   # Latent dimension
R = 320   # Resolution
C = 8192  # First layer number of channels
I = 17898 # Number of slices for training

if __name__ == '__main__':
    '''
        Train example on 10% of the LIDC dataset
        db_path is expected to refer to a serialized torch tensor file
        with shape: [N, 1, R, R]; N > I
        The dataset is expected to be normalized (see preprocess)
    '''

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    set_seed()

    # Increasing batch sizes multiplied by duration (in epochs)
    # For instance, during the first 800 epochs the batch size will be 32
    batch_sizes = [32]*800 + [64]*200 + [128]*100 + [256]*50

    runner = Runner(L, R, C)
    runner.run(
        run_name = 'train',
        indices = list(range(I)),
        # PATHS
        save_path = '',
        db_path = '',
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
        # LOOPS, BATCH & SCHEDULER
        f_scheduler = lambda epoch: 1,
        f_batch = lambda epoch: batch_sizes[epoch],
        n_loop_epochs = len(batch_sizes),
        # CHECKPOINTS
        save = ['examples', 'loss', 'latent', 'decoder'],
        checkpoints = 100,
        max_save = 128,
        nrow = 4,
        # OPTIM
    )
