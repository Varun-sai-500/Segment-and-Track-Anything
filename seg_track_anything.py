import os
import cv2
from PIL import Image
import shutil
from mask_utils import draw_mask, save_mask
import numpy as np
import torch
import gc
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

    # create dir to save result 
    tracking_result_dir = f'{os.path.join(os.path.dirname(__file__), "tracking_results", f"{video_name}")}'
    os.makedirs(tracking_result_dir,exist_ok=True)
    
    io_args = {
        'tracking_result_dir': tracking_result_dir,
        'output_mask_dir': f'{tracking_result_dir}/{video_name}_masks',
        'output_masked_frame_dir': f'{tracking_result_dir}/{video_name}_masked_frames',
        'output_video': f'{tracking_result_dir}/{video_name}_seg.mp4', # keep same format as input video
    }
    if input_video:
        pred_list = []
        masked_pred_list = []

        # source video to segment
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if frame_num > 0:
            output_mask_name = sorted([img_name for img_name in os.listdir(io_args['output_mask_dir'])])
            output_masked_frame_name = sorted([img_name for img_name in os.listdir(io_args['output_masked_frame_dir'])])

            for i in range(0, frame_num):
                cap.read()
                pred_list.append(np.array(Image.open(os.path.join(io_args['output_mask_dir'], output_mask_name[i])).convert('P')))
                masked_pred_list.append(cv2.imread(os.path.join(io_args['output_masked_frame_dir'], output_masked_frame_name[i])))

        
        # create dir to save predicted mask and masked frame
        if frame_num == 0:
            if os.path.isdir(io_args['output_mask_dir']):
                shutil.rmtree(io_args['output_mask_dir'])
            if os.path.isdir(io_args['output_masked_frame_dir']):
                shutil.rmtree(io_args['output_masked_frame_dir'])
        output_mask_dir = io_args['output_mask_dir']
        os.makedirs(io_args['output_mask_dir'],exist_ok=True)
        os.makedirs(io_args['output_masked_frame_dir'],exist_ok=True)

        if USE_CUDA: torch.cuda.empty_cache()
        gc.collect()
        sam_gap = SegTracker.sam_gap
        frame_idx = 0

        amp_ctx = torch.amp.autocast(device_type='cuda') if USE_CUDA else nullcontext
        with amp_ctx():
            while cap.isOpened():
                ret, frame  = cap.read()  
                if not ret:
                    break
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                
                if frame_idx == 0:
                    pred_mask = SegTracker.first_frame_mask
                    if USE_CUDA: torch.cuda.empty_cache()
                    gc.collect()
                elif (frame_idx % sam_gap) == 0:
                    seg_mask = SegTracker.seg(frame)
                    if USE_CUDA: torch.cuda.empty_cache()
                    gc.collect()
                    track_mask = SegTracker.track(frame)
                    # find new objects, and update tracker with new objects
                    new_obj_mask = SegTracker.find_new_objs(track_mask,seg_mask)
                    save_mask(new_obj_mask, output_mask_dir, str(frame_idx+frame_num).zfill(5) + '_new.png')
                    pred_mask = track_mask + new_obj_mask
                    # segtracker.restart_tracker()
                    SegTracker.add_reference(frame, pred_mask)
                else:
                    pred_mask = SegTracker.track(frame,update_memory=True)
                if USE_CUDA: torch.cuda.empty_cache()
                gc.collect()
                
                save_mask(pred_mask, output_mask_dir, str(frame_idx + frame_num).zfill(5) + '.png')
                pred_list.append(pred_mask)

                print("processed frame {}, obj_num {}".format(frame_idx + frame_num, SegTracker.get_obj_num()),end='\r')
                frame_idx += 1
            cap.release()
            print('\nfinished')
        
        ##################
        # Visualization
        ##################

        # draw pred mask on frame and save as a video
        cap = cv2.VideoCapture(input_video)
        # if frame_num > 0:
        #     for i in range(0, frame_num):
        #         cap.read()  
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pred_mask = pred_list[frame_idx]
            masked_frame = draw_mask(frame, pred_mask)
            cv2.imwrite(f"{io_args['output_masked_frame_dir']}/{str(frame_idx).zfill(5)}.png", masked_frame[:, :, ::-1])

            masked_pred_list.append(masked_frame)
            masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
            out.write(masked_frame)
            print('frame {} writed'.format(frame_idx),end='\r')
            frame_idx += 1
        out.release()
        cap.release()
        print("\n{} saved".format(io_args['output_video']))
        print('\nfinished')


        # zip predicted mask
        zip_path = f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(io_args['output_mask_dir']):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, io_args['output_mask_dir'])
                    zipf.write(full_path, arcname)
        # manually release memory (after cuda out of memory)
        del SegTracker
        if USE_CUDA: torch.cuda.empty_cache()
        gc.collect()

        return io_args['output_video'], f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"

    if input_img_seq:
        pred_list = []
        masked_pred_list = []

        if frame_num > 0:
            output_mask_name = sorted([img_name for img_name in os.listdir(io_args['output_mask_dir'])])
            output_masked_frame_name = sorted([img_name for img_name in os.listdir(io_args['output_masked_frame_dir'])])
            for i in range(0, frame_num):
                pred_list.append(np.array(Image.open(os.path.join(io_args['output_mask_dir'], output_mask_name[i])).convert('P')))
                masked_pred_list.append(cv2.imread(os.path.join(io_args['output_masked_frame_dir'], output_masked_frame_name[i])))

        # create dir to save predicted mask and masked frame
        if frame_num == 0:
            if os.path.isdir(io_args['output_mask_dir']):
                shutil.rmtree(io_args['output_mask_dir'])
            if os.path.isdir(io_args['output_masked_frame_dir']):
                shutil.rmtree(io_args['output_masked_frame_dir'])

        output_mask_dir = io_args['output_mask_dir']
        os.makedirs(io_args['output_mask_dir'],exist_ok=True)
        os.makedirs(io_args['output_masked_frame_dir'],exist_ok=True)


        i_frame_num = frame_num

        if USE_CUDA: torch.cuda.empty_cache()
        gc.collect()
        sam_gap = SegTracker.sam_gap
        frame_idx = 0

        amp_ctx = torch.amp.autocast(device_type='cuda') if USE_CUDA else nullcontext
        with amp_ctx():
            for img_path in imgs_path:
                if i_frame_num > 0:
                    i_frame_num = i_frame_num - 1
                    continue

                frame_name = os.path.basename(img_path).split('.')[0]
                frame = cv2.imread(img_path)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                
                if frame_idx == 0:
                    pred_mask = SegTracker.first_frame_mask
                    if USE_CUDA:   torch.cuda.empty_cache()
                    gc.collect()
                elif (frame_idx % sam_gap) == 0:
                    seg_mask = SegTracker.seg(frame)
                    if USE_CUDA: torch.cuda.empty_cache()
                    gc.collect()
                    track_mask = SegTracker.track(frame)
                    # find new objects, and update tracker with new objects
                    new_obj_mask = SegTracker.find_new_objs(track_mask,seg_mask)
                    save_mask(new_obj_mask, output_mask_dir, f'{frame_name}_new.png')
                    pred_mask = track_mask + new_obj_mask
                    # segtracker.restart_tracker()
                    SegTracker.add_reference(frame, pred_mask)
                else:
                    pred_mask = SegTracker.track(frame,update_memory=True)
                if USE_CUDA: torch.cuda.empty_cache()
                gc.collect()
                
                save_mask(pred_mask, output_mask_dir, f'{frame_name}.png')
                pred_list.append(pred_mask)

                print("processed frame {}, obj_num {}".format(frame_idx+frame_num, SegTracker.get_obj_num()),end='\r')
                frame_idx += 1
            print('\nfinished')
        
        ##################
        # Visualization
        ##################

        # draw pred mask on frame and save as a video
        height, width = pred_list[0].shape
        fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
        i_frame_num =frame_num 

        out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

        frame_idx = 0
        for img_path in imgs_path:
            # if i_frame_num > 0:
            #     i_frame_num = i_frame_num - 1
            #     continue
            frame_name = os.path.basename(img_path).split('.')[0]
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            pred_mask = pred_list[frame_idx]
            masked_frame = draw_mask(frame, pred_mask)
            masked_pred_list.append(masked_frame)
            cv2.imwrite(f"{io_args['output_masked_frame_dir']}/{frame_name}.png", masked_frame[:, :, ::-1])

            masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
            out.write(masked_frame)
            print('frame {} writed'.format(frame_name),end='\r')
            frame_idx += 1
        out.release()
        print("\n{} saved".format(io_args['output_video']))
        print('\nfinished')

        # zip predicted mask
        zip_path = f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(io_args['output_mask_dir']):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, io_args['output_mask_dir'])
                    zipf.write(full_path, arcname)

        # manually release memory (after cuda out of memory)
        del SegTracker
        if USE_CUDA: torch.cuda.empty_cache()
        gc.collect()

        return io_args['output_video'], f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"
