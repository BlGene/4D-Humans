from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def save2image_fn(save_fn):
    video_name, frame_and_person = str(save_fn).split('.')[0:2]
    frame = frame_and_person.split("_")[0]
    return f"{'.'.join((video_name, frame))}.jpg"    


def main():
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    
    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    all_saved_paths = sorted(list(Path(args.out_folder).glob('*.pt')))
    for save_path in tqdm(all_saved_paths):
        img_fn = save2image_fn(save_path.stem)
        img_path = Path(args.img_folder) / img_fn
        if not img_path.is_file():
            print("Image file not found", img_path)
            continue
        render_path = Path(args.out_folder) / f'{img_fn}_all.png'
        if render_path.is_file():
            continue

        img_cv2 = cv2.imread(str(img_path))
        img_size = np.array(img_cv2.shape[:2])
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

        tmp = torch.load(save_path)
        all_verts = tmp['all_verts']
        all_cam_t = tmp['all_cam_t']

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[::-1], **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(str(render_path), 255*input_img_overlay[:, :, ::-1])
    print("done.")

if __name__ == '__main__':
    main()
