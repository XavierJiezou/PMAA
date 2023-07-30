import warnings
from utils import *
import tqdm
import torch
import argparse
from torch.utils.data import DataLoader
from model.pmaa import PMAA
from dataset_new import Sen2_MTC
from torch.utils.data import DataLoader
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")


def test_and_visualization(opt, model_GEN, test_loader, criterion):
    model_GEN.eval()

    psnr_list = []
    ssim_list = []
    total_loss = 0

    pbar = tqdm.tqdm(total=len(test_loader), ncols=0,
                     desc=f"{opt.test_mode}", unit=" step")
    save_path = os.path.join(opt.predict_image_path, opt.test_mode)
    with torch.no_grad():
        for (real_A, real_B, image_names) in test_loader:
            real_A[0], real_A[1], real_A[2], real_B = real_A[0].cuda(
            ), real_A[1].cuda(), real_A[2].cuda(), real_B.cuda()
            real_A_input = torch.stack(
                (real_A[0], real_A[1], real_A[2]), 1).cuda()
            fake_B, cloud_mask, _ = model_GEN(real_A_input)

            loss = criterion(fake_B, real_B)

            for batch in range(opt.batch_size):
                image_name = image_names[batch]
                output, label = fake_B[batch], real_B[batch]
                input_1, input_2, input_3 = real_A[0][batch], real_A[1][batch], real_A[2][batch]
                input_1, input_2, input_3 = get_rgb(
                    input_1), get_rgb(input_2), get_rgb(input_3)

                output_rgb, label_rgb = get_rgb(output), get_rgb(label)

                psnr, ssim = psnr_ssim_cal(label_rgb, output_rgb)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                save_dir = os.path.join(
                    save_path, f"psnr_{psnr:.3f}_ssim_{ssim:.3f}")
                os.makedirs(save_dir, exist_ok=True)
                for idx, real_img in enumerate([input_1, input_2, input_3]):
                    save_image(real_img, save_dir, image_name +
                               f'_real_A{idx + 1}.png')
                save_heatmap([cloud_mask[0][batch], cloud_mask[1][batch],
                             cloud_mask[2][batch]], save_dir, image_name)
                save_image(output_rgb, save_dir, image_name + '_fake_B.png')
                save_image(label_rgb, save_dir, image_name + '_real_B.png')

            total_loss += loss.item()
            pbar.update()
            pbar.set_postfix(
                loss_val=f"{total_loss:.4f}"
            )

    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    psnr = np.mean(psnr_list)
    ssim = np.mean(ssim_list)

    pbar.set_postfix(loss_val=f"{total_loss:.4f}",
                     psnr=f"{psnr:.3f}", ssim=f"{ssim:.3f}")
    pbar.close()
    return psnr, ssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """Path"""
    parser.add_argument("--load_gen", type=str, default='checkpoints29/CTGAN_Sen2_MTC/G_best_PSNR_19.590_SSIM_0.598.pth',
                        help="which checkpoint you want to use for generator")
    parser.add_argument("--predict_image_path", type=str,
                        default='./image_out29', help="name of the dataset_list")
    parser.add_argument("--root", type=str, default='data',
                        help="Path to dataset")

    """Parameters"""
    parser.add_argument("--image_size", type=int,
                        default=256, help="image size")
    parser.add_argument("--in_channel", type=int, default=4,
                        help="the number of input channels")
    parser.add_argument("--out_channel", type=int, default=4,
                        help="the number of output channels")

    """base_options"""
    parser.add_argument("--test_mode", type=str, default='test',
                        help="which data_mode you want to use?(val/test)")
    parser.add_argument("--n_cpu", type=int, default=0,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--batch_size", type=int,
                        default=1, help="size of the batches")
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
    opt = parser.parse_args()

    random_seed_general = 2022
    fixed_seed(random_seed_general)
    os.makedirs(os.path.join(opt.predict_image_path,
                opt.test_mode), exist_ok=True)

    test_data = Sen2_MTC(opt, opt.test_mode)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size,
                             shuffle=False, num_workers=opt.n_cpu, drop_last=False)

    """define model & optimizer"""
    model_GEN = PMAA(32, 4)

    def replace_batchnorm(model):
        for name, child in model.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                child: torch.nn.BatchNorm2d = child
                setattr(model, name, torch.nn.InstanceNorm2d(child.num_features))
            else:
                replace_batchnorm(child)
    replace_batchnorm(model_GEN)
    model_GEN.load_state_dict(torch.load(opt.load_gen))
    print('load transformer model successfully!')
    model_GEN = model_GEN.cuda()

    criterion = torch.nn.L1Loss().cuda()

    test_and_visualization(opt=opt, model_GEN=model_GEN,
                           test_loader=test_loader, criterion=criterion)
