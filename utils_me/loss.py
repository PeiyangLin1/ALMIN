import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F


def region_min_difference(img, block_size=16, target=0.7):
    """
    输入：
        img: [B, C, H, W]，单通道图像
        block_size: 小块大小（默认 16）
        target: 目标值（默认 0.7）

    输出：
        所有小块最小值与 target 的平均差值（绝对值）
    """
    B, C, H, W = img.shape
    assert H % block_size == 0 and W % block_size == 0, "图像尺寸必须能被 block_size 整除"

    # 划分图像为小块，输出形状为 [B, C * block_size * block_size, num_blocks]
    patches = F.unfold(img, kernel_size=block_size, stride=block_size)  # [B, C*K*K, N]

    # 对每个小块取最小值（dim=1 表示沿每个 patch 内部像素）
    patch_mins = patches.min(dim=1).values  # shape: [B, N]

    # 与 target 的差值
    diffs = patch_mins - target  # shape: [B, N]

    # 平均绝对差
    abs_mean_diff = diffs.abs().mean()

    return abs_mean_diff


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        squared_deviation = (image_vis - 0.5).pow(2)
        batch_size, channels, height, width = image_vis.size()
        spatial_variance = squared_deviation.sum(dim=[2, 3], keepdim=True) / (height * width - 1)
        over_value =  0.333*torch.exp(-(squared_deviation / (2 * spatial_variance)))
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(over_value*image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)+region_min_difference(image_vis)*0.75
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()