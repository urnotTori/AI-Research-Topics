# Import required libraries
from torch.utils.data import dataset  # Unused in this script, can be removed
from tqdm import tqdm  # For showing progress bars during processing
import network  # Custom module for defining DeepLab model architecture
import utils  # Custom module with helper functions
import os  # For file path operations
import argparse  # For command-line argument parsing
import numpy as np  # For numerical operations
from torch.utils import data  # Unused, can be removed
from torchvision import transforms as T  # For image preprocessing
import torch  # Main PyTorch library
import torch.nn as nn  # For neural network modules
from PIL import Image  # For image file operations
from glob import glob  # For file pattern matching

# CRF related imports
import pydensecrf.densecrf as dcrf  # Core DenseCRF functions
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian  # Helper functions

import cv2  # For image processing
import matplotlib.pyplot as plt  # For colormap visualization


def get_argparser():
    parser = argparse.ArgumentParser()  # Create argument parser

    parser.add_argument("--input", type=str, required=True)  # Input image or directory
    parser.add_argument("--dataset", type=str, default='cityscapes', choices=['voc', 'cityscapes'])  # Dataset type
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower()
                              and not name.startswith("__") and callable(network.modeling.__dict__[name]))  # Filter available models
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101', choices=available_models)  # Model choice
    parser.add_argument("--separable_conv", action='store_true', default=False)  # Use separable conv if True
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])  # Output stride setting
    parser.add_argument("--save_val_results_to", default=None)  # Output directory to save results
    parser.add_argument("--crop_val", action='store_true', default=False)  # Unused here, useful in val loader
    parser.add_argument("--val_batch_size", type=int, default=4)  # Unused here
    parser.add_argument("--crop_size", type=int, default=513)  # Unused here
    parser.add_argument("--ckpt", default=None, type=str)  # Checkpoint path
    parser.add_argument("--gpu_id", type=str, default='0')  # GPU device id
    return parser

# Define Cityscapes colormap for visualization
CITYSCAPES_COLORMAP = np.array([
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156),
    (190,153,153), (153,153,153), (250,170, 30), (220,220,  0),
    (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60),
    (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100),
    (  0, 80,100), (  0,  0,230), (119, 11, 32)
], dtype=np.uint8)

# Convert predicted mask (H x W) to RGB using colormap
def decode_cityscapes_target(mask):
    h, w = mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for label in np.unique(mask):
        if label < len(CITYSCAPES_COLORMAP):
            color_image[mask == label] = CITYSCAPES_COLORMAP[label]
    return color_image

# Blend original image and predicted mask
def overlay_mask_on_image(image, mask, alpha=0.6):
    image = np.array(image).astype(np.float32)
    mask = mask.astype(np.float32)
    blended = image * (1 - alpha) + mask * alpha
    return blended.astype(np.uint8)

# Smooth colormap overlay using matplotlib (for visualization only)
def smooth_colormap(pred):
    norm_pred = pred.astype(np.float32) / 18.0  # Normalize to [0, 1]
    color = plt.cm.plasma(norm_pred)[..., :3] * 255  # Apply colormap and convert to RGB
    return color.astype(np.uint8)

# Apply DenseCRF post-processing to refine predictions
def apply_dense_crf(image, prediction):
    h, w = prediction.shape
    n_labels = prediction.max() + 1
    d = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_labels(prediction, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(unary)
    feats_gaussian = create_pairwise_gaussian(sdims=(3, 3), shape=(h, w))
    d.addPairwiseEnergy(feats_gaussian, compat=3)
    feats_bilateral = create_pairwise_bilateral(sdims=(50, 50), schan=(13, 13, 13), img=image, chdim=2)
    d.addPairwiseEnergy(feats_bilateral, compat=10)
    Q = d.inference(5)
    refined_pred = np.argmax(Q, axis=0).reshape((h, w))
    return refined_pred


def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 19
    decode_fn = decode_cityscapes_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s' % ext), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)

    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location='cpu')
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Loaded checkpoint from", opts.ckpt)
        del checkpoint
    else:
        print("[!] No checkpoint found")
        model = nn.DataParallel(model)
        model.to(device)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if opts.save_val_results_to:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    with torch.no_grad():
        model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext) - 1]

            orig_img = Image.open(img_path).convert('RGB')
            img = transform(orig_img).unsqueeze(0).to(device)

            pred = model(img).max(1)[1].cpu().numpy()[0]

            orig_w, orig_h = orig_img.size
            pred = Image.fromarray(pred.astype(np.uint8)).resize((orig_w, orig_h), resample=Image.NEAREST)
            pred = np.array(pred)

            refined_pred = apply_dense_crf(np.array(orig_img), pred)

            color_mask = decode_fn(refined_pred)
            overlay = overlay_mask_on_image(orig_img, color_mask, alpha=0.6)
            pretty_color = smooth_colormap(refined_pred)
            pretty_color = cv2.resize(pretty_color, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            pretty_overlay = cv2.addWeighted(np.array(orig_img), 0.4, pretty_color, 0.6, 0)

            Image.fromarray(color_mask).save(os.path.join(opts.save_val_results_to, img_name + '_crf_pred.png'))
            Image.fromarray(overlay).save(os.path.join(opts.save_val_results_to, img_name + '_crf_overlay.png'))
            cv2.imwrite(os.path.join(opts.save_val_results_to, img_name + '_pretty_overlay.png'), pretty_overlay[..., ::-1])

            # Create 2x2 comparison grid
            original_np = np.array(orig_img.resize((orig_w, orig_h)))
            vis1 = color_mask  # crf_pred
            vis2 = overlay  # crf_overlay
            vis3 = pretty_overlay[..., ::-1]  # BGR to RGB

            top_row = np.concatenate([original_np, vis1], axis=1)
            bottom_row = np.concatenate([vis2, vis3], axis=1)
            grid = np.concatenate([top_row, bottom_row], axis=0)

            cv2.imwrite(os.path.join(opts.save_val_results_to, img_name + '_comparison.png'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
