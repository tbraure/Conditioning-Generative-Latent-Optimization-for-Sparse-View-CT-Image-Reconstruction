import logging
import numpy as np
import torch
import pylidc as pl
from torch.nn.functional import interpolate, normalize
from torchvision.transforms.functional import pad, resize, InterpolationMode
from torchvision.utils import save_image

DB_PATH = ''
OUT_RES = [320, 320]
PATIENT = range(1, 1003)
RESIZE  = True

def my_resample(vol, slices, new_spacing=[1, 1, 1], out_res=OUT_RES):

    spacing = np.array(
        [float(slices[0].SliceThickness)] \
      + [float(v) for v in slices[0].PixelSpacing],
        dtype=np.float16
    )

    resize_factor = spacing / new_spacing
    resize_factor = (
        resize_factor[0],
        resize_factor[1] * out_res[0] / vol.shape[1],
        resize_factor[2] * out_res[1] / vol.shape[2]
    )

    vol = interpolate(
        torch.from_numpy(vol).to(torch.device('cuda')).view(1, 1, *vol.shape),
        scale_factor = resize_factor
    ).squeeze()

    logging.info(f'{vol.shape}')

    return vol.unsqueeze(1).cpu()

def hounsfield_to_density(vol):
    # densities [g/cm³]
    water_density = 1
    air_density = 1e-3

    # mass attenuation coefficient at 150KeV [cm²/g]
    water_mass_attenuation = 0.1505
    air_mass_attenuation = 0.1356

    # mass attenuation coefficient at 100KeV [cm²/g]
    # water_mass_attenuation = 0.1707
    # air_mass_attenuation = 0.1541

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
    import warnings; warnings.filterwarnings('ignore')

    patient_ids = ['LIDC-IDRI-' + f'0000{i}'[-4:] for i in PATIENT]
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id.in_(patient_ids))

    vols = []
    sizes = []
    for i, scan in enumerate(scans):

        logging.info(f'{i}')
        slices = scan.load_all_dicom_images()

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

        if RESIZE:
            vol = resize(
                torch.from_numpy(vol).to(torch.device('cuda')).unsqueeze(1),
                OUT_RES,
                interpolation = InterpolationMode.NEAREST
            ).cpu()
        else:
            vol = my_resample(vol, slices)

        vol = hounsfield_to_density(vol)

        sizes.append(vol.shape[0])
        vols.append(vol)

    if not RESIZE:
        padding = [0, 0, 0, 0]
        for i, vol in enumerate(vols):
            padding[0] = (OUT_RES[0] - vol.shape[2]) // 2 + vol.shape[2] % 2
            padding[1] = (OUT_RES[1] - vol.shape[3]) // 2 + vol.shape[3] % 2
            padding[2] = (OUT_RES[0] - vol.shape[2]) // 2
            padding[3] = (OUT_RES[1] - vol.shape[3]) // 2
            vols[i] = pad(vol, padding=padding)

    vols = torch.cat(vols)
    vmax = vols.max()
    vmin = vols.min()
    torch.save((vols - vmin) / (vmax - vmin), f'{DB_PATH}/train_{"resize" if RESIZE else "resample"}.pt')
    torch.save(torch.Tensor(sizes), f'{DB_PATH}/train_{"resize" if RESIZE else "resample"}_sizes.pt')

