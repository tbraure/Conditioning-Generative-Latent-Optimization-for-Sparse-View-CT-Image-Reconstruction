import logging
import numpy as np
import torch
import glob
import pydicom as pd

from tqdm import tqdm

DB_PATH = ''

def hounsfield_to_density(vol):

    # densities [g/cm³]
    water_density = 1
    air_density = 1e-3

    # mass attenuation coefficient at 150KeV [cm²/g]
    water_mass_attenuation = 0.1505
    air_mass_attenuation = 0.1356

    # attenuation coefficients [1/cm]
    water_attenuation = water_mass_attenuation * water_density
    air_attenuation = air_mass_attenuation * air_density

    attenuation = vol \
                * (water_attenuation - air_attenuation) \
                / 1000 \
                + water_attenuation

    return attenuation / water_attenuation * water_density

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    dir_list = glob.glob(
        f'{DB_PATH}/LDCT-and-Projection-data/**/*Full*',
        recursive=True
    )

    vols, sizes = [], []
    for d in tqdm(dir_list):

        file_list = glob.glob(f'{d}/*.dcm'); file_list.sort()
        slices = [pd.dcmread(f) for f in file_list]

        size = len(slices)
        intercept = float(slices[0].RescaleIntercept)
        slope = float(slices[0].RescaleSlope)

        vol = np.zeros((size, 512, 512), dtype=np.float16)
        for j, s in enumerate(slices):
            img = s.pixel_array.astype(np.float16)
            img *= slope
            img += intercept
            img = np.clip(img, -970, 3000)
            img[img==-970] = -1000
            vol[j, ...] = img

        vol = hounsfield_to_density(torch.from_numpy(vol).unsqueeze(1))
        sizes.append(vol.shape[0])
        vols.append(vol)

    vols = torch.cat(vols)
    vmax = vols.max()
    vmin = vols.min()
    torch.save((vols - vmin) / (vmax - vmin), f'{DB_PATH}/LDCT.pt')
    torch.save(torch.Tensor(sizes), f'{DB_PATH}/LDCT_sizes.pt')

