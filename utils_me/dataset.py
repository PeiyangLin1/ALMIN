import torch.utils.data as Data
import h5py
import numpy as np
import torch


class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path

        h5f = h5py.File(h5file_path, 'r')

        # 检查必须存在的 key（适配四个输入）
        required_keys = ['ir_patchs', 'vis_patchs', 'enhance_patchs', 'extra_patchs']
        for k in required_keys:
            if k not in h5f:
                raise KeyError(f"H5 文件中缺少必须的 key: '{k}'")

        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]

        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        ENHANCE = np.array(h5f['enhance_patchs'][key])
        EXTRA = np.array(h5f['extra_patchs'][key])

        h5f.close()
        return torch.Tensor(VIS), torch.Tensor(IR), torch.Tensor(ENHANCE), torch.Tensor(EXTRA)
