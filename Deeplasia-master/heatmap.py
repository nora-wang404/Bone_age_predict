import os
import logging.config
import pandas as pd
import torch
import yaml
import re
import warnings
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# 假设这些是你已有的导入
from data_process.data_config import rootdata
from lib.utils.log import LOG_CONFIG
from lib.datasets import InferenceDataset
from lib.models import BoneAgeModel
from lib.legacy import from_checkpoint as load_legacy_model
from lib import testing

logging.config.dictConfig(LOG_CONFIG)
warnings.filterwarnings("ignore")


import logging.config
import pandas as pd
import torch
import yaml
import re
import warnings
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# 假设这些是你已有的导入
from data_process.data_config import rootdata
from lib.utils.log import LOG_CONFIG
from lib.datasets import InferenceDataset
from lib.models import BoneAgeModel
from lib.legacy import from_checkpoint as load_legacy_model
from lib import testing

logging.config.dictConfig(LOG_CONFIG)
warnings.filterwarnings("ignore")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None  # 存储目标层特征图
        self.gradient = None       # 存储目标层梯度
        self.hook_handles = []     # 存储钩子句柄
        self._register_hooks()     # 注册钩子

    def _register_hooks(self):
        """注册钩子并强制检查目标层是否存在"""
        # 前向钩子：捕获特征图
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()  # 存储当前图像的特征图

        # 反向钩子：捕获梯度
        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0].detach()  # 存储当前图像的梯度

        # 遍历模型所有模块，查找目标层
        target_found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                # 注册钩子并保存句柄
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))
                target_found = True
                break

        # 关键检查：若目标层未找到，直接报错
        if not target_found:
            raise ValueError(f"目标层 '{self.target_layer}' 未在模型中找到！请检查层名是否正确。\n"
                             f"可用层名可通过打印 model.named_modules() 查看。")

    def __call__(self, x, male):
        """生成热力图（针对回归任务）"""
        self.model.eval()
        x = x.requires_grad_()  # 启用输入梯度计算

        # 前向传播：获取预测值（需传入male特征）
        output = self.model(x, male)
        pred = output if not isinstance(output, tuple) else output[0]  # 确保pred是标量张量

        # 反向传播：计算梯度（核心步骤，依赖钩子捕获梯度）
        self.model.zero_grad()  # 清空历史梯度
        pred.backward(retain_graph=True)  # 对预测值求导

        # 检查是否成功捕获特征图和梯度
        if self.feature_maps is None or self.gradient is None:
            raise RuntimeError("未捕获到特征图或梯度！可能目标层不是卷积层，或钩子注册失败。")

        # 计算Grad-CAM热力图
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)  # 全局平均池化梯度（权重）
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze()  # 加权组合特征图
        cam = F.relu(cam)  # 只保留正向影响

        # 归一化（避免数值溢出）
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)

        # 清理钩子（避免重复注册）
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []  # 重置钩子句柄

        return cam.cpu().detach().numpy(), pred.item()

def visualize_heatmap(original_image, heatmap, prediction, normalized=True, save_path=None):
    """可视化热力图（适配单通道/三通道图像）"""
    # 确保原始图像是RGB格式（若为单通道，转为三通道）
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    # 调整热力图尺寸至原图大小
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # 生成彩色热力图
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # 转为RGB格式

    # 叠加热力图与原图
    superimposed = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)  # 加权融合

    # 显示结果
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(original_image)
    plt.title("Origin")
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(heatmap_colored)
    plt.title("Grad-CAM")
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(superimposed)
    title = f"{prediction:.1f} months"
    plt.title(title)
    plt.axis('off')

    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"已保存热力图至: {save_path}")
    plt.close()  # 关闭图像以释放内存

    return superimposed

def main():
    logger = logging.getLogger()
    parser = create_parser()
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.gpus = 1 if device == "cuda" else 0

    # 加载数据
    loader = create_loader(args, logger)

    # 加载模型
    model = BoneAgeModel.load_from_checkpoint(args.ckp_path, weights_only=False)
    model.to(device)
    model.eval()

    #  设置 的目标层
    target_layer = "backbone.base._conv_head"

    # 遍历数据生成热力图
    num_samples = 100
    count = 0
    logger.info("开始生成热力图...")

    for batch in loader:
        if count >= num_samples:
            break

        # 解析批次数据
        images, male, imgpaths = batch['x'], batch['male'], batch['image_name']
        images = images.to(device)
        male = male.to(device)

        # 处理单张图像
        for i in range(images.size(0)):
            if count >= num_samples:
                break

            # 提取单样本数据
            img_tensor = images[i:i+1]  # 保持(batch=1, C, H, W)格式
            male_tensor = male[i:i+1]
            img_path = imgpaths[i]  # 当前图像路径

            # 生成热力图
            try:
                grad_cam = GradCAM(model, target_layer)  # 每次处理新图像时重新实例化，确保钩子正确注册
                heatmap, pred_normalized = grad_cam(img_tensor, male_tensor)
            except Exception as e:
                logger.error(f"处理图像 {img_path} 时出错: {str(e)}")
                continue

            # 转换为实际骨龄（月）
            pred_actual = rescale_prediction(torch.tensor(pred_normalized), args.ckp_path).item()

            # 加载原始图像（用于可视化）
            original_img = cv2.imread(img_path)
            if original_img is None:
                logger.warning(f"图像 {img_path} 无法读取，跳过...")
                continue
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 转为RGB格式

            # 生成并保存可视化结果
            save_name = f"bone_age_heatmap_{count}_{os.path.basename(img_path).split('.')[0]}.png"
            save_path = os.path.join("./reli", save_name)
            visualize_heatmap(original_img, heatmap, pred_actual, normalized=False, save_path=save_path)

            logger.info(f"已处理 {count+1}/{num_samples}，图像: {img_path}，预测骨龄: {pred_actual:.1f} 月")
            count += 1

    logger.info("热力图生成完成！")

# 以下函数保持不变（create_parser, create_loader, rescale_prediction）
def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--ckp_path", type=str, default="output/debug/version_1/ckp/best_model.ckpt")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--legacy_ckp", action="store_true")
    parser.add_argument("--annotation_csv", type=str, default="./val_data.csv")
    parser.add_argument("--split_column", type=str, default="")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--img_dir", type=str, default=os.path.join(rootdata, "boneage-training-dataset"))
    parser.add_argument("--input_size", nargs="+", default=[1, 512, 512], type=int)
    parser.add_argument("--image_norm_method", type=str, default="zscore")
    parser.add_argument("--mask_crop_size", type=float, default=-1)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--rotation_angle", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--output_path", type=str, default="output/predictions_results.csv")
    parser.add_argument("--store_activations", action="store_true")
    parser = pl.Trainer.add_argparse_args(parser)
    return parser

def create_loader(args, logger):
    args.mask_dirs = []
    loader = DataLoader(
        InferenceDataset(
            annotation_df=args.annotation_csv,
            split_column=args.split_column,
            split_name=args.split_name,
            img_dir=args.img_dir,
            norm_method=args.image_norm_method,
            input_size=args.input_size,
            mask_crop_size=args.mask_crop_size,
            flip=args.flip,
            rotation_angle=args.rotation_angle,
        ),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
    )
    return loader

def rescale_prediction(y_hat, ckp_path, params_path="data/parameters.yml"):
    with open(params_path, "r") as stream:
        cor_params = yaml.safe_load(stream)
    age_mean, age_sd = cor_params["age_mean"], cor_params["age_sd"]
    y_hat = y_hat * age_sd + age_mean  # 反归一化
    return y_hat

if __name__ == "__main__":
    main()

