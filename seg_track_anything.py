import os
import cv2
from PIL import Image
import shutil
from mask_utils import draw_mask, save_mask
import numpy as np
import torch
import zipfile
from contextlib import nullcontext

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"

def tracking_objects_in_video(SegTracker, input_video, input_img_seq, fps, frame_num=0):
    if input_video is not None:
        video_name = os.path.basename(input_video).split('.')[0]
    elif input_img_seq is not None:
        file_name = input_img_seq.name.split('/')[-1].split('.')[0]
        file_path = f'./assets/{file_name}'
        imgs_path = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path)])
        video_name = file_name
    else:
        return None, None

    # Create dir to save result 
    tracking_result_dir = f'{os.path.join(os.path.dirname(__file__), "tracking_results", f"{video_name}")}'
    os.makedirs(tracking_result_dir, exist_ok=True)
    
    io_args = {
        'tracking_result_dir': tracking_result_dir,
        'output_mask_dir': f'{tracking_result_dir}/{video_name}_masks',
        'output_masked_frame_dir': f'{tracking_result_dir}/{video_name}_masked_frames',
        'output_video': f'{tracking_result_dir}/{video_name}_seg.mp4',
    }

    # Video Stream Processing
    if input_video:
        pred_list = []
        masked_pred_list = []

        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Pre-fill prior frames if tracking from an offset
        if frame_num > 0:
            if os.path.exists(io_args['output_mask_dir']) and os.path.exists(io_args['output_masked_frame_dir']):
                output_mask_name = sorted([img_name for img_name in os.listdir(io_args['output_mask_dir']) if not img_name.endswith('_new.png')])
                output_masked_frame_name = sorted([img_name for img_name in os.listdir(io_args['output_masked_frame_dir'])])

                available_masks = len(output_mask_name)
                available_frames = len(output_masked_frame_name)

                for i in range(frame_num):
                    cap.read()  # Always advance the video stream to frame_num
                    
                    # Safely append cached data only while within available file bounds
                    if i < available_masks:
                        pred_list.append(np.array(Image.open(os.path.join(io_args['output_mask_dir'], output_mask_name[i])).convert('P')))
                    if i < available_frames:
                        masked_pred_list.append(cv2.imread(os.path.join(io_args['output_masked_frame_dir'], output_masked_frame_name[i])))
                        
        # Setup output folders
        if frame_num == 0:
            if os.path.isdir(io_args['output_mask_dir']):
                shutil.rmtree(io_args['output_mask_dir'])
            if os.path.isdir(io_args['output_masked_frame_dir']):
                shutil.rmtree(io_args['output_masked_frame_dir'])

        output_mask_dir = io_args['output_mask_dir']
        os.makedirs(io_args['output_mask_dir'], exist_ok=True)
        os.makedirs(io_args['output_masked_frame_dir'], exist_ok=True)

        if USE_CUDA: torch.cuda.empty_cache()

        frame_idx = 0
        amp_ctx = torch.amp.autocast(device_type='cuda') if USE_CUDA else nullcontext

        with amp_ctx():
            while cap.isOpened():
                ret, frame = cap.read()  
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # First frame relative to current tracking pass uses the explicit mask provided
                if frame_idx == 0:
                    pred_mask = SegTracker.first_frame_mask
                else:
                    # Pure memory-based propagation without auto SAM re-segmentation
                    pred_mask = SegTracker.track(frame, update_memory=True)

                if USE_CUDA: torch.cuda.empty_cache()
                
                
                save_mask(pred_mask, output_mask_dir, str(frame_idx + frame_num).zfill(5) + '.png')
                pred_list.append(pred_mask)

                print("processed frame {}, obj_num {}".format(frame_idx + frame_num, SegTracker.get_obj_num()), end='\r')
                frame_idx += 1

            cap.release()
            print('\nfinished tracking loop')

        # Visualization Assembly
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= len(pred_list):
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_mask = pred_list[frame_idx]
            masked_frame = draw_mask(frame, pred_mask)
            cv2.imwrite(f"{io_args['output_masked_frame_dir']}/{str(frame_idx).zfill(5)}.png", masked_frame[:, :, ::-1])

            masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
            out.write(masked_frame)
            frame_idx += 1

        out.release()
        cap.release()

        # Archive mask output
        zip_path = f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(io_args['output_mask_dir']):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, io_args['output_mask_dir'])
                    zipf.write(full_path, arcname)

        if USE_CUDA: torch.cuda.empty_cache()
        

        return io_args['output_video'], zip_path

    # Image-Sequence Stream Processing
    if input_img_seq:
        pred_list = []
        masked_pred_list = []

        if frame_num > 0 and os.path.exists(io_args['output_mask_dir']) and os.path.exists(io_args['output_masked_frame_dir']):
            output_mask_name = sorted([img_name for img_name in os.listdir(io_args['output_mask_dir']) if not img_name.endswith('_new.png')])
            output_masked_frame_name = sorted([img_name for img_name in os.listdir(io_args['output_masked_frame_dir'])])
            for i in range(min(frame_num, len(output_mask_name))):
                pred_list.append(np.array(Image.open(os.path.join(io_args['output_mask_dir'], output_mask_name[i])).convert('P')))
                masked_pred_list.append(cv2.imread(os.path.join(io_args['output_masked_frame_dir'], output_masked_frame_name[i])))

        if frame_num == 0:
            if os.path.isdir(io_args['output_mask_dir']):
                shutil.rmtree(io_args['output_mask_dir'])
            if os.path.isdir(io_args['output_masked_frame_dir']):
                shutil.rmtree(io_args['output_masked_frame_dir'])

        output_mask_dir = io_args['output_mask_dir']
        os.makedirs(io_args['output_mask_dir'], exist_ok=True)
        os.makedirs(io_args['output_masked_frame_dir'], exist_ok=True)

        i_frame_num = frame_num

        if USE_CUDA: torch.cuda.empty_cache()
        

        frame_idx = 0
        amp_ctx = torch.amp.autocast(device_type='cuda') if USE_CUDA else nullcontext

        with amp_ctx():
            for img_path in imgs_path:
                if i_frame_num > 0:
                    i_frame_num -= 1
                    continue

                frame_name = os.path.basename(img_path).split('.')[0]
                frame = cv2.imread(img_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if frame_idx == 0:
                    pred_mask = SegTracker.first_frame_mask
                else:
                    pred_mask = SegTracker.track(frame, update_memory=True)

                if USE_CUDA: torch.cuda.empty_cache()
                
                
                save_mask(pred_mask, output_mask_dir, f'{frame_name}.png')
                pred_list.append(pred_mask)

                print("processed frame {}, obj_num {}".format(frame_idx + frame_num, SegTracker.get_obj_num()), end='\r')
                frame_idx += 1

        # Visualization Assembly
        height, width = pred_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

        frame_idx = 0
        for img_path in imgs_path:
            if frame_idx >= len(pred_list):
                break

            frame_name = os.path.basename(img_path).split('.')[0]
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pred_mask = pred_list[frame_idx]
            masked_frame = draw_mask(frame, pred_mask)
            cv2.imwrite(f"{io_args['output_masked_frame_dir']}/{frame_name}.png", masked_frame[:, :, ::-1])

            masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
            out.write(masked_frame)
            frame_idx += 1

        out.release()

        zip_path = f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(io_args['output_mask_dir']):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, io_args['output_mask_dir'])
                    zipf.write(full_path, arcname)

        if USE_CUDA: torch.cuda.empty_cache()
        
        return io_args['output_video'], zip_path