import cv2

from net import Encoder, Decoder,LGFB
import os
import numpy as np
from utils_me.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils_me.img_read_save import img_save,image_read_cv2
import warnings
import logging


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path = r""

TARGET_WIDTH = 640
TARGET_HEIGHT = 512

model_name=""
test_folder=r''
test_out_folder=os.path.join('')

os.makedirs(test_out_folder, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.empty_cache()

Encoder = nn.DataParallel(Encoder()).to(device)
Decoder = nn.DataParallel(Decoder()).to(device)
Base_LGFB = nn.DataParallel(LGFB(64)).to(device)
Detail_LGFB = nn.DataParallel(LGFB(64)).to(device)

Encoder.load_state_dict(torch.load(ckpt_path)['Encoder'])
Decoder.load_state_dict(torch.load(ckpt_path)['Decoder'])
Base_LGFB.load_state_dict(torch.load(ckpt_path)['Base_LGFB'])
Detail_LGFB.load_state_dict(torch.load(ckpt_path)['Detail_LGFB'])

Encoder.eval()
Decoder.eval()
Base_LGFB.eval()
Detail_LGFB.eval()


def resize_image(image, target_width, target_height):
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

with torch.no_grad():
    for img_name in os.listdir(os.path.join(test_folder,"ir")):
        print(f"Processing {img_name}...")
        a=img_name
        ir_img = image_read_cv2(os.path.join(test_folder,"ir",img_name), mode='GRAY')
        vi_img = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')
        img_name = os.path.splitext(img_name)[0] + ".png"

        illum_img = image_read_cv2(os.path.join(test_folder,"illumination_features",img_name), mode='GRAY')
        img_name = os.path.splitext(img_name)[0] + ".png"
        extra_img = image_read_cv2(os.path.join(test_folder,"reflectance",img_name), mode='GRAY')
        img_name=a


        data_IR = ir_img[np.newaxis, np.newaxis, ...] / 255.0
        data_VIS = vi_img[np.newaxis, np.newaxis, ...] / 255.0
        data_ILLUM = illum_img[np.newaxis, np.newaxis, ...] / 255.0
        data_EXTRA = extra_img[np.newaxis, np.newaxis, ...] / 255.0
        data_IR, data_VIS, data_ILLUM, data_EXTRA = (
            torch.FloatTensor(data_IR),
            torch.FloatTensor(data_VIS),
            torch.FloatTensor(data_ILLUM),
            torch.FloatTensor(data_EXTRA),
        )
        data_VIS, data_IR, data_ILLUM, data_EXTRA = data_VIS.cuda(), data_IR.cuda(), data_ILLUM.cuda(), data_EXTRA.cuda()
        eps=1e-6
        feature_I_B, feature_I_D,_= Encoder(data_IR)
        feature_V_B, feature_V_D,_= Encoder(data_EXTRA)


        feature_F_B = Base_LGFB( feature_I_B,feature_V_B,data_ILLUM)
        feature_F_D = Detail_LGFB(feature_I_D,feature_V_D,data_ILLUM)
        data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
        data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))

        fi = np.squeeze((data_Fuse * 255).clamp(0, 255).byte().cpu().numpy())
        img_save(fi, img_name.split(sep='.')[0], test_out_folder)

        if device == 'cuda':
            torch.cuda.empty_cache()

eval_folder = test_out_folder
ori_img_folder = test_folder

metric_result = np.zeros((8))
for img_name in os.listdir(os.path.join(ori_img_folder,"ir")):
        ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
        vi = image_read_cv2(os.path.join(ori_img_folder,"vi", img_name), 'GRAY')


        fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
        metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                    , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                    , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                    , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

metric_result /= len(os.listdir(eval_folder))
print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
        +str(np.round(metric_result[1], 2))+'\t'
        +str(np.round(metric_result[2], 2))+'\t'
        +str(np.round(metric_result[3], 2))+'\t'
        +str(np.round(metric_result[4], 2))+'\t'
        +str(np.round(metric_result[5], 2))+'\t'
        +str(np.round(metric_result[6], 2))+'\t'
        +str(np.round(metric_result[7], 2))
        )
print("="*80)
