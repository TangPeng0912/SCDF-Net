import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.io import loadmat
import tifffile
from scipy.ndimage import gaussian_filter
import xlrd
import pandas as pd


class HSI_DatasetAnyScale(Dataset):
    def __init__(self, opt, mode='train'):
        super(HSI_DatasetAnyScale, self).__init__()

        self.use_random_crop = False
        self.opt = opt
        self.mode = mode
        self.data_root = "./data"

        self.scale_factor = float(opt.scale_factor)

        # paths
        self.data_path = os.path.join(self.data_root, opt.dataset_name)
        self.wavelength_path = os.path.join(self.data_root, "wavelength", f"{opt.dataset_name}.txt")
        self.srf_path = os.path.join(self.data_root, "SRF", opt.srf_name)

        # wavelengths / SRF
        self.wavelengths = self.read_wavelengths()
        self.srf, self.srf_wl = self.load_srf()

        # simulate SRF
        self.target_srf, self.hsi2msi_mask = self.cal_simul_srf(self.wavelengths, self.srf)

        # avoid multi-write excel conflicts
        save_dir = os.path.join(opt.results_dir, f"{opt.dataset_name}_{opt.suffix}")
        xlsx_path = os.path.join(save_dir, f"{opt.dataset_name}_target_srf.xlsx")
        if not os.path.exists(xlsx_path):
            self.save_target_srf_to_excel(xlsx_path)

        # file list
        self.data_files = self.get_data_files()

        # preload
        self.data_list = []
        self.hrmsi_list = []
        self.lrhsi_list = []
        self.lrmsi_list = []
        self.lrlrhsi_list = []

        for data_file in self.data_files:
            data = self.load_data(data_file)                 # ✅ includes normalization to [0,1]
            hrmsi = self.generate_MSI(data, self.target_srf)  # still in [0,1] if data is normalized

            sigma = self.scale_factor / 2.35482

            # ✅ mixed degradation: integer -> blur+stride, fractional -> blur+resize
            lrhsi, h = self.generate_low_HSI_mixed(data, self.scale_factor, sigma)
            lrmsi, h = self.generate_low_HSI_mixed(hrmsi, self.scale_factor, sigma)
            lrlrhsi, h = self.generate_low_HSI_mixed(lrhsi, self.scale_factor, sigma)

            self.psf = h

            self.data_list.append(data)
            self.hrmsi_list.append(hrmsi)
            self.lrhsi_list.append(lrhsi)
            self.lrmsi_list.append(lrmsi)
            self.lrlrhsi_list.append(lrlrhsi)

    # =========================
    # File / metadata loading
    # =========================
    def get_data_files(self):
        if self.opt.dataset_name.lower() == 'chikusei':
            return glob.glob(os.path.join(self.data_path, '*.tif'))
        return glob.glob(os.path.join(self.data_path, '*.mat'))

    def read_wavelengths(self):
        try:
            with open(self.wavelength_path, 'r') as file:
                content = file.read().strip()
            wavelength = [float(num.strip()) for num in content.split(',') if num.strip()]
            return np.array(wavelength)
        except FileNotFoundError:
            print(f"文件 '{self.wavelength_path}' 未找到。")
            return np.array([])
        except Exception as e:
            print(f"处理文件 '{self.wavelength_path}' 时发生错误: {e}")
            return np.array([])

    def load_srf(self):
        xls_path = os.path.join(self.data_root, "SRF", self.opt.srf_name + '.xls')
        if not os.path.exists(xls_path):
            raise Exception(f"Spectral response path for '{self.opt.srf_name}' does not exist")

        data = xlrd.open_workbook(xls_path)
        table = data.sheets()[0]
        num_cols = table.ncols

        cols_list = [np.array(table.col_values(i)).reshape(-1, 1) for i in range(num_cols)]
        srf = np.concatenate(cols_list, axis=1)
        wl = srf[0, 1:].astype('float64')
        srf = srf[1:, :].astype('float64')
        return srf, wl

    def cal_simul_srf(self, tar_wl, sat_srf):
        if np.max(tar_wl[:]) < 100:
            tar_wl = tar_wl * 1000

        sat_wl = sat_srf[1:, 0].reshape(1, -1).astype("float64")
        sat_srf_value = sat_srf[1:, 1:].T.astype("float64")

        if tar_wl.ndim == 1:
            tar_wl = np.expand_dims(tar_wl, 1)

        min_value = tar_wl - sat_wl
        min_index = np.argmin(np.absolute(min_value), axis=1)

        tar_srf = sat_srf_value[:, min_index].T
        tar_srf_sum = np.sum(tar_srf, axis=0)

        mask = np.ones(tar_srf.shape[1]) > 0
        if np.any(tar_srf_sum == 0):
            mask = (tar_srf_sum != 0)
            tar_srf = tar_srf[:, mask]
            print("\033[1;31;40mThe mask for sat_srf to tar_srf:\033[0m", mask)

        return tar_srf / (tar_srf.sum(axis=0) + 1e-12), mask

    def load_data(self, data_file):
        """
        ✅ 关键：统一把数据归一化到 [0,1]，避免 Chikusei PSNR 负数/ RMSE 爆炸
        """
        if self.opt.dataset_name.lower() == 'chikusei':
            data = tifffile.imread(data_file).astype(np.float32)
        else:
            mat_data = loadmat(data_file)
            possible_keys = ['REF', 'img', 'data', 'hsi', 'HSI', 'hyperspectral', 'Hyperspectral']

            if hasattr(self.opt, 'mat_key') and self.opt.mat_key in mat_data:
                data = mat_data[self.opt.mat_key]
            else:
                data = None
                for key in possible_keys:
                    if key in mat_data and isinstance(mat_data[key], np.ndarray):
                        data = mat_data[key]
                        # print(f"使用键名 '{key}' 加载数据")
                        break
                if data is None:
                    special_keys = ['__header__', '__version__', '__globals__']
                    for key in mat_data.keys():
                        if key not in special_keys and isinstance(mat_data[key], np.ndarray):
                            data = mat_data[key]
                            # print(f"使用键名 '{key}' 加载数据")
                            break
                if data is None:
                    raise ValueError(f"无法在mat文件中找到有效的数据键名: {data_file}")

            data = data.astype(np.float32)

        # -------- normalization to [0,1] --------
        # robust: handle negative or constant arrays
        dmin = float(np.min(data))
        dmax = float(np.max(data))
        if not np.isfinite(dmin) or not np.isfinite(dmax):
            raise ValueError(f"数据包含非有限值: {data_file}")

        if dmax == dmin:
            # all constant
            data = np.zeros_like(data, dtype=np.float32)
        else:
            # if already [0,1], keep; otherwise normalize
            # (some mats are already normalized)
            if dmax > 1.0 or dmin < 0.0:
                data = (data - dmin) / (dmax - dmin)

        data = np.clip(data, 0.0, 1.0).astype(np.float32)
        return data

    # =========================
    # MSI generation
    # =========================
    def generate_MSI(self, img, sp_matrix):
        h, w, c = img.shape
        img_reshaped = img.reshape(-1, c)
        msi_reshaped = np.dot(img_reshaped, sp_matrix)
        msi = msi_reshaped.reshape(h, w, sp_matrix.shape[1]).astype(np.float32)
        msi = np.clip(msi, 0.0, 1.0)
        return msi

    # =========================
    # Mixed degradation
    # =========================
    def _is_near_integer(self, scale: float, eps: float = 1e-6) -> bool:
        s = float(scale)
        return s > 0 and abs(s - round(s)) <= eps

    def _gaussian_kernel_2d(self, sigma: float):
        kernel_size = max(3, int(round(float(sigma) * 6)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        m, n = [(kernel_size - 1.0) / 2.0] * 2
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h.astype(np.float32), kernel_size

    def _blur_and_stride_downsample(self, img: np.ndarray, stride: int, sigma: float):
        """
        Integer scale: blur + stride downsample
        img: numpy(H,W,C)
        """
        if img.ndim == 2:
            img = img[:, :, None]

        h_kernel, ksize = self._gaussian_kernel_2d(sigma)
        pad = ksize // 2

        x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        c = x.shape[1]

        k = torch.from_numpy(h_kernel).view(1, 1, ksize, ksize).repeat(c, 1, 1, 1)
        y = F.conv2d(x, k, bias=None, stride=(stride, stride), padding=pad, groups=c)

        out = y.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        return out, h_kernel

    def _resize_anyscale(self, img, scale):
        if scale <= 0:
            raise ValueError(f"scale必须为正数，当前为: {scale}")

        h, w = img.shape[:2]
        out_h = max(1, int(round(h / scale)))
        out_w = max(1, int(round(w / scale)))

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        resized = F.interpolate(img_t, size=(out_h, out_w), mode='bicubic', align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

    def generate_low_HSI_anyscale(self, img, scale, sigma):
        """
        Fractional scale: blur + bicubic resize (kept as your original idea)
        """
        if img.ndim == 2:
            img = img[:, :, None]

        # blur first (anti-alias)
        blurred = gaussian_filter(img, sigma=(sigma, sigma, 0), mode='nearest').astype(np.float32)

        # record a gaussian kernel as psf (for logging / psf return)
        kernel_size = max(3, int(round(sigma * 6)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        def matlab_style_gauss2D(shape=(3, 3), sigma_val=0.5):
            m, n = [(ss - 1.0) / 2.0 for ss in shape]
            y, x = np.ogrid[-m:m+1, -n:n+1]
            h = np.exp(-(x * x + y * y) / (2.0 * sigma_val * sigma_val))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h

        h = matlab_style_gauss2D((kernel_size, kernel_size), sigma).astype(np.float32)

        out_img = self._resize_anyscale(blurred, scale)
        out_img = np.clip(out_img, 0.0, 1.0)
        return out_img, h

    def generate_low_HSI_mixed(self, img, scale, sigma):
        """
        ✅ Mixed degradation:
        - near-integer: blur + stride
        - fractional:  blur + resize
        """
        if self._is_near_integer(scale):
            stride = int(round(float(scale)))
            out, h = self._blur_and_stride_downsample(img, stride=stride, sigma=sigma)
            out = np.clip(out, 0.0, 1.0)
            return out, h
        else:
            return self.generate_low_HSI_anyscale(img, scale, sigma)

    # =========================
    # (Optional) old stride-valid method (kept)
    # =========================
    def generate_low_HSI(self, img, stride, sigma):
        def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
            m, n = [(ss - 1.) / 2. for ss in shape]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h

        h = matlab_style_gauss2D((stride, stride), sigma)
        if img.ndim == 3:
            img_w, img_h, img_c = img.shape
        elif img.ndim == 2:
            img_c = 1
            img_w, img_h = img.shape
            img = img.reshape((img_w, img_h, 1))
        from scipy import signal
        out_img = np.zeros((img_w // stride, img_h // stride, img_c), dtype=np.float32)
        for i in range(img_c):
            out = signal.convolve2d(img[:, :, i], h, 'valid')
            out_img[:, :, i] = out[::stride, ::stride]
        out_img = np.clip(out_img, 0.0, 1.0)
        return out_img, h.astype(np.float32)

    # =========================
    # Random crop
    # =========================
    def random_crop(self, idx):
        lrlrhsi = self.lrlrhsi_list[idx]
        lrhsi = self.lrhsi_list[idx]
        lrmsi = self.lrmsi_list[idx]
        hsi = self.data_list[idx]
        hrmsi = self.hrmsi_list[idx]

        hsi_h, hsi_w = hsi.shape[:2]
        lrhsi_h, lrhsi_w = lrhsi.shape[:2]
        lrlrhsi_h, lrlrhsi_w = lrlrhsi.shape[:2]

        hsi_to_lrhsi_ratio = hsi_h / lrhsi_h
        lrhsi_to_lrlrhsi_ratio = lrhsi_h / lrlrhsi_h

        patch_size = 10
        x = random.randint(0, lrlrhsi_w - patch_size)
        y = random.randint(0, lrlrhsi_h - patch_size)

        lrhsi_x = int(x * lrhsi_to_lrlrhsi_ratio)
        lrhsi_y = int(y * lrhsi_to_lrlrhsi_ratio)
        lrhsi_patch_size = int(patch_size * lrhsi_to_lrlrhsi_ratio)

        hsi_x = int(lrhsi_x * hsi_to_lrhsi_ratio)
        hsi_y = int(lrhsi_y * hsi_to_lrhsi_ratio)
        hsi_patch_size = int(lrhsi_patch_size * hsi_to_lrhsi_ratio)

        lrlrhsi_patch = lrlrhsi[y:y + patch_size, x:x + patch_size]
        lrhsi_patch = lrhsi[lrhsi_y:lrhsi_y + lrhsi_patch_size, lrhsi_x:lrhsi_x + lrhsi_patch_size]
        lrmsi_patch = lrmsi[lrhsi_y:lrhsi_y + lrhsi_patch_size, lrhsi_x:lrhsi_x + lrhsi_patch_size]
        hsi_patch = hsi[hsi_y:hsi_y + hsi_patch_size, hsi_x:hsi_x + hsi_patch_size]
        hrmsi_patch = hrmsi[hsi_y:hsi_y + hsi_patch_size, hsi_x:hsi_x + hsi_patch_size]

        return hsi_patch, hrmsi_patch, lrhsi_patch, lrmsi_patch, lrlrhsi_patch

    def __getitem__(self, idx):
        if self.use_random_crop:
            hsi, hrmsi, lrhsi, lrmsi, lrlrhsi = self.random_crop(idx)
        else:
            hsi = self.data_list[idx]
            hrmsi = self.hrmsi_list[idx]
            lrhsi = self.lrhsi_list[idx]
            lrmsi = self.lrmsi_list[idx]
            lrlrhsi = self.lrlrhsi_list[idx]

        hsi = torch.from_numpy(hsi.transpose(2, 0, 1)).float()
        hrmsi = torch.from_numpy(hrmsi.transpose(2, 0, 1)).float()
        lrhsi = torch.from_numpy(lrhsi.transpose(2, 0, 1)).float()
        lrmsi = torch.from_numpy(lrmsi.transpose(2, 0, 1)).float()
        lrlrhsi = torch.from_numpy(lrlrhsi.transpose(2, 0, 1)).float()

        hsi_wl = torch.from_numpy(self.wavelengths).float()
        msi_wl = torch.from_numpy(self.srf_wl).float()
        srf = torch.from_numpy(self.srf).float()
        psf = torch.from_numpy(self.psf).float()

        return {
            'hsi': hsi,
            'hrmsi': hrmsi,
            'lrhsi': lrhsi,
            'lrmsi': lrmsi,
            'lrlrhsi': lrlrhsi,
            'hsi_wl': hsi_wl,
            "msi_wl": msi_wl,
            "srf": srf,
            "psf": psf
        }

    def __len__(self):
        return len(self.data_files)

    # =========================
    # export SRF
    # =========================
    def save_target_srf_to_excel(self, save_path):
        try:
            df = pd.DataFrame(self.target_srf)

            if hasattr(self, 'wavelengths') and len(self.wavelengths) == self.target_srf.shape[1]:
                df.columns = [f'波长_{wl:.2f}nm' for wl in self.wavelengths]
            else:
                df.columns = [f'列_{i+1}' for i in range(self.target_srf.shape[1])]
                print("警告：波长信息与数据维度不匹配，使用默认列名")

            if hasattr(self, 'srf_wl') and len(self.srf_wl) == self.target_srf.shape[0]:
                df.index = [f'波段_{wl:.2f}nm' for wl in self.srf_wl]
            else:
                df.index = [f'行_{i+1}' for i in range(self.target_srf.shape[0])]
                print("警告：波段信息与数据维度不匹配，使用默认行名")

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_excel(save_path, index=True)
            print(f"成功将target_srf保存到: {save_path}")
        except Exception as e:
            print(f"保存target_srf到Excel时发生错误: {e}")
