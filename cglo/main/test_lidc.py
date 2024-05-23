import torch
import numpy as np
from scipy.interpolate import interp1d

from cglo.generic import set_seed, Runner
from cglo.utils.losses import l1_dose_half_loss
from cglo.utils.physics import center_radon_transform

import logging

L = 320  # Latent dimension
R = 320  # Resolution
C = 8192 # First layer number of channels
DB = ''  # Database path
A = 100  # Number of viewing angles

if __name__ == '__main__':
    '''
        Reconstruction example on 5 patients from the LIDC test dataset
        db_path is expected to refer to a serialized torch tensor file
        with shape: [N, 1, R, R]; N > I
        The dataset is expected to be normalized (see preprocess)
    '''

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    set_seed()

    # Half dose loss expect y to be measurements from ground-truths
    # and x to be reconstructions
    # Pre-computing ground-truths measurements saves time
    f_loss = lambda x, y: l1_dose_half_loss(x, y, A)

    # Test slices database with shape [N, 1, R, R]
    # For LIDC test dataset R = 320 and N = 791
    x = torch.load(f'{DB}/test.pt').to(torch.float)

    # test_sizes should contain the sizes of each patient in the test set
    # For LIDC test_sizes = torch.Tensor([166, 133, 252, 125, 115])
    # so that, test_sizes.sum() == 791
    s = torch.load(f'{DB}/test_sizes.pt').cumsum(-1).to(torch.int).tolist()
    s.insert(0, 0)

    # Increasing batch sizes multiplied by duration (in epochs)
    # For instance, during the first 400 epochs the batch size will be 20
    batch_sizes = [20]*400 + [40]*200 + [80]*100 + [160]*50 + [320]*50

    c = 0
    # Looping on 5 patients
    for k, l in zip(s[:5], s[1:6]):

        # Pre-computing simulated experimental measurements
        p = center_radon_transform(x[list(range(k, l))], A).numpy()

        # Upsampling measurements (higher number of slices)
        i = np.linspace(0, 1, p.shape[0])
        j = np.concatenate(
            [np.linspace(n, m, 8, endpoint=False) for n, m in zip(i[:-1], i[1:])] \
          + [np.ones(1)]
        )
        f = interp1d(i, p, axis=0, assume_sorted=True)

        runner = Runner(L, R, C)
        runner.bdd = torch.from_numpy(f(j))
        runner.run(
            run_name = f'reco_{c}',
            indices = None,
            # PATHS
            save_path = '',
            db_path = None,
            dec_path = '',  # decoder from unsupervised training step
            lat_path = None,
            # INITS
            bdd_init = False,
            dec_init = True,
            lat_init = True,
            # LOADS
            dec_load = True,
            lat_load = False,
            # LOSSES
            f_loss = f_loss,
            # LOOPS, BATCH & SCHEDULER
            f_scheduler = lambda epoch: 1,
            f_batch = lambda epoch: batch_sizes[epoch],
            n_loop_epochs = len(batch_sizes),
            # CHECKPOINTS
            save = ['latent', 'pred', 'loss', 'decoder'],
            checkpoints = 100,
            max_save = 128,
            nrow = 4,
            # OPTIM
            latent = {'lr': 1e-2},
            decoder = {'lr': 1e-4}
        )
        c += 1

