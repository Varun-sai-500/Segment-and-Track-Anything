import os
import tempfile
import zipfile

import cv2
import gradio as gr
import numpy as np

from model_args import deaot_args, dino_args, sam_args
from pipeline import Pipeline

# ------------------------------------------------------------------
# Pipeline Helpers & State Reset
# ------------------------------------------------------------------


def ensure_pipeline(tracker):
    return tracker if tracker is not None else Pipeline(sam_args, dino_args, deaot_args)


def draw_points(points, modes, frame):
    for (x, y), mode in zip(points, modes):
        color = (0, 153, 255) if mode == 1 else (255, 80, 80)
        cv2.circle(frame, (int(x), int(y)), 8, color, -1)
    return frame


def reset_state(tracker, output_path=None):
    if tracker is not None:
        tracker.restart_tracker()
    _discard_tracking_output(output_path)
    return tracker, [], [], [], [], 0, None, None


# ------------------------------------------------------------------
# Input Processing
# ------------------------------------------------------------------


def get_meta_from_video(input_video, tracker, output_path):
    if input_video is None:
        return _reset_and_unpack(tracker, output_path)

    cap = cv2.VideoCapture(input_video)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return _reset_and_unpack(tracker, output_path)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _reset_and_unpack(tracker, output_path, frame)


def get_meta_from_img_seq(input_img_seq, tracker, output_path):
    if input_img_seq is None:
        return _reset_and_unpack(tracker, output_path)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(input_img_seq.name, "r") as zf:
            zf.extractall(temp_dir)

        paths = sorted([
            os.path.join(r, f)
            for r, _, files in os.walk(temp_dir)
            for f in files if os.path.splitext(f)[1].lower() in valid_exts
        ])
        if not paths:
            raise ValueError("Image sequence ZIP contains no valid images.")

        frame = cv2.imread(paths[0])
        if frame is None:
            raise ValueError(f"Failed to decode image: {paths[0]}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _reset_and_unpack(tracker, output_path, frame)


def _reset_and_unpack(tracker, output_path, frame=None):
    state = reset_state(tracker, output_path)
    # Replaces state tuple items with `frame` where applicable
    return (frame, frame, "") + state[:6] + (frame, frame)


# ------------------------------------------------------------------
# Segmentation & Object Commit
# ------------------------------------------------------------------


def _get_existing_mask(tracker):
    return tracker.get_current_mask() if tracker and tracker.current_mask is not None else None


def _commit_segmentation(tracker, reference_frame, predicted_mask, frame_idx, previous_mask):
    if predicted_mask is None or not np.any(predicted_mask):
        return

    predicted_mask = np.asarray(predicted_mask)
    if previous_mask is None:
        tracker.initialize_reference(reference_frame, predicted_mask, frame_step=frame_idx)
        return

    prev_ids = set(np.unique(previous_mask)) - {0}
    curr_ids = set(np.unique(predicted_mask)) - {0}
    new_ids = curr_ids - prev_ids

    if new_ids:
        new_objects_mask = np.where(np.isin(predicted_mask, list(new_ids)), predicted_mask, 0).astype(np.uint8)
        if np.any(new_objects_mask):
            tracker.add_objects(mask=new_objects_mask, frame_step=frame_idx)


def execute_segmentation(tracker, origin_frame, ref_frame, coords, modes, c_groups, m_groups, frame_idx):
    if origin_frame is None:
        return tracker, None, None, coords, modes, c_groups, m_groups

    if coords:
        c_groups.append(coords.copy())
        m_groups.append(modes.copy())

    if not c_groups:
        return tracker, origin_frame, origin_frame, [], [], [], []

    tracker = ensure_pipeline(tracker)
    prev_mask = _get_existing_mask(tracker)
    pred_mask, masked_frame = tracker.seg_acc_click(
        origin_frame=origin_frame, coords_groups=c_groups, modes_groups=m_groups
    )

    _commit_segmentation(tracker, ref_frame, pred_mask, frame_idx or 0, prev_mask)
    return tracker, masked_frame, masked_frame, [], [], [], []


def gd_detect(tracker, origin_frame, ref_frame, caption, box_thresh, text_thresh, frame_idx):
    if origin_frame is None or not caption:
        return tracker, origin_frame, origin_frame

    tracker = ensure_pipeline(tracker)
    prev_mask = _get_existing_mask(tracker)
    pred_mask, masked_frame = tracker.detect_and_seg(origin_frame, caption, box_thresh, text_thresh)

    _commit_segmentation(tracker, ref_frame, pred_mask, frame_idx or 0, prev_mask)
    return tracker, masked_frame, masked_frame


# ------------------------------------------------------------------
# Interactive Points
# ------------------------------------------------------------------


def record_click(origin_frame, point_mode, coords, modes, evt: gr.SelectData):
    if origin_frame is None:
        return None, coords, modes

    coords.append([evt.index[0], evt.index[1]])
    modes.append(1 if point_mode == "Positive" else 0)
    frame_display = draw_points(coords, modes, origin_frame.copy())
    return frame_display, coords, modes


def undo_last_click(origin_frame, coords, modes):
    if origin_frame is None:
        return None, coords, modes

    if coords:
        coords.pop()
        modes.pop()

    frame_display = draw_points(coords, modes, origin_frame.copy()) if coords else origin_frame.copy()
    return frame_display, coords, modes


def add_new_object(coords, modes, c_groups, m_groups):
    if coords:
        c_groups.append(coords.copy())
        m_groups.append(modes.copy())
    return [], [], c_groups, m_groups


# ------------------------------------------------------------------
# Output & Video Writing Session Management
# ------------------------------------------------------------------

_TRACKING_OUTPUTS = {}


def _create_tracking_output():
    fd, path = tempfile.mkstemp(prefix="tracked_", suffix=".mp4")
    os.close(fd)
    _TRACKING_OUTPUTS[path] = {"writer": None, "width": None, "height": None, "fps": None, "completed": False}
    return path


def _discard_tracking_output(path):
    session = _TRACKING_OUTPUTS.pop(path, None)
    if session and session.get("writer"):
        session["writer"].release()
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _output_fps(input_video, fps):
    if input_video:
        cap = cv2.VideoCapture(input_video)
        try:
            if cap.isOpened():
                src_fps = cap.get(cv2.CAP_PROP_FPS)
                if src_fps and np.isfinite(src_fps) and src_fps > 0:
                    return float(src_fps)
        finally:
            cap.release()
    return float(fps)


def _write_tracking_frame(path, frame, input_video, fps):
    session = _TRACKING_OUTPUTS.get(path)
    if not session:
        raise RuntimeError("Tracking session unavailable.")

    h, w = frame.shape[:2]
    if session["writer"] is None:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), _output_fps(input_video, fps), (w, h))
        if not writer.isOpened():
            writer.release()
            _discard_tracking_output(path)
            raise RuntimeError("Could not create tracked video output.")
        session.update({"writer": writer, "width": w, "height": h})

    if session["width"] != w or session["height"] != h:
        raise RuntimeError("Frame dimensions changed during tracking.")

    session["writer"].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _finish_tracking_output(path):
    session = _TRACKING_OUTPUTS.get(path)
    if not session:
        return None

    if session.get("writer"):
        session["writer"].release()
        session["writer"] = None

    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        _discard_tracking_output(path)
        return None

    session["completed"] = True
    return path


def _get_completed_output(path):
    session = _TRACKING_OUTPUTS.get(path)
    return path if session and session.get("completed") and os.path.isfile(path) and os.path.getsize(path) > 0 else None


def _black_frame(frame):
    return np.zeros_like(frame) if frame is not None and frame.ndim == 3 and frame.size > 0 else None


# ------------------------------------------------------------------
# Tracking Pipeline Engine
# ------------------------------------------------------------------


def pause_and_load_frame_for_segmentation(tracker, frame, frame_idx):
    if tracker:
        tracker.stop_tracking()
    if frame is None:
        raise gr.Error("No tracked frame is available to edit.")
    return gr.update(value=frame, interactive=True), frame, [], [], [], []


def tracking_objects(tracker, current_frame, input_video, input_img_seq, fps, frame_idx, output_path):
    if tracker is None or tracker.current_mask is None:
        raise gr.Error("Please perform initial segmentation or detection before tracking!")
    if (input_video is None) == (input_img_seq is None):
        raise gr.Error("Select either a video OR an image sequence.")

    start_idx = int(frame_idx or 0)
    session = _TRACKING_OUTPUTS.get(output_path)
    if not output_path or not session or session.get("completed"):
        output_path = _create_tracking_output()

    last_frame, last_idx, saw_frame = current_frame, start_idx, False

    try:
        for masked_frame, idx in tracker.track_video_sequence(
            input_video=input_video, input_img_seq=input_img_seq, fps=fps, frame_num=start_idx
        ):
            if masked_frame is None:
                continue

            masked_frame = np.asarray(masked_frame)
            _write_tracking_frame(output_path, masked_frame, input_video, fps)
            last_frame, last_idx, saw_frame = masked_frame, int(idx), True

            yield masked_frame, last_idx, masked_frame, output_path

        if not tracker._stop_event.is_set() and saw_frame:
            completed_path = _finish_tracking_output(output_path)
            if completed_path is None:
                raise RuntimeError("Tracked video could not be finalized.")

            black = _black_frame(last_frame)
            if black is not None:
                yield black, last_idx, black, completed_path

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Tracking failed: {e}")


def reset_SegTracker(tracker, origin_frame, output_path):
    if tracker:
        tracker.restart_tracker()
    _discard_tracking_output(output_path)
    frame = origin_frame.copy() if origin_frame is not None else None
    return tracker, frame, [], [], [], [], "", 0, frame, None


# ------------------------------------------------------------------
# Gradio UI Layout
# ------------------------------------------------------------------

CUSTOM_CSS = """
.yellow-btn {
    background-color: #f59e0b !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
}
.yellow-btn:hover { background-color: #d97706 !important; }
"""


def app():
    with gr.Blocks(title="Segment and Track Anything") as demo:
        gr.Markdown("<h1 style='text-align: center;'>Segment and Track Anything</h1>")

        # States
        coords, modes = gr.State([]), gr.State([])
        c_groups, m_groups = gr.State([]), gr.State([])
        origin_frame, current_frame, ref_frame = gr.State(None), gr.State(None), gr.State(None)
        tracker, frame_idx, output_path = gr.State(None), gr.State(0), gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Video Input"):
                        input_video = gr.Video(label="Input video", height=230)
                    with gr.Tab("Image-Seq Input"):
                        with gr.Row():
                            input_img_seq = gr.File(label="Input Image-Seq (ZIP)", height=150, scale=3)
                            with gr.Column(scale=2, min_width=120):
                                extract_btn = gr.Button("Extract", size="sm")
                                fps = gr.Slider(label="FPS", minimum=5, maximum=50, value=8, step=1)

                with gr.Tabs():
                    with gr.Tab("Click Prompt"):
                        with gr.Row(equal_height=True):
                            point_mode = gr.Radio(["Positive", "Negative"], value="Positive", label="Prompt Type", scale=3)
                            undo_btn = gr.Button("Undo", scale=1, min_width=70, size="sm")
                        segment_btn = gr.Button("Segment / Add Object Reference", elem_classes=["yellow-btn"])
                        new_obj_btn = gr.Button("Add Point Group (New Object)")

                    with gr.Tab("Text Prompt"):
                        caption = gr.Textbox(label="Detection Prompt", placeholder="Objects separated by periods")
                        with gr.Accordion("Advanced Options", open=False):
                            with gr.Row():
                                box_thresh = gr.Slider(label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25)
                                text_thresh = gr.Slider(label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25)
                        detect_btn = gr.Button("Segment / Add Object Reference", elem_classes=["yellow-btn"])

                reset_btn = gr.Button("Reset All", variant="stop")

            with gr.Column(scale=1):
                canvas = gr.Image(label="Interactive Canvas", interactive=True, height=480)

        gr.Markdown("---\n### TRACK")
        with gr.Row():
            track_btn = gr.Button("Start / Resume Tracking", variant="primary")
            add_obj_track_btn = gr.Button("Add Objects During Tracking", variant="secondary")

        with gr.Row():
            output_frame = gr.Image(label="Live Tracking Progress", height=480, scale=8)
            with gr.Column(scale=2, min_width=170):
                output_file = gr.File(label="Download", interactive=False)
                status = gr.Textbox(label="Status", value="Ready", interactive=False, lines=2)

        # Video/Image-Seq Inputs Setup
        inputs_meta = [tracker, output_path]
        outputs_meta = [canvas, origin_frame, caption, tracker, coords, modes, c_groups, m_groups, frame_idx, current_frame, ref_frame]

        input_video.upload(get_meta_from_video, [input_video] + inputs_meta, outputs_meta)
        input_img_seq.upload(get_meta_from_img_seq, [input_img_seq] + inputs_meta, outputs_meta)
        extract_btn.click(get_meta_from_img_seq, [input_img_seq] + inputs_meta, outputs_meta)

        for inp in (input_video, input_img_seq):
            inp.upload(lambda: None, None, [output_file], queue=False)

        # Interactions
        canvas.select(record_click, [origin_frame, point_mode, coords, modes], [canvas, coords, modes])
        undo_btn.click(undo_last_click, [origin_frame, coords, modes], [canvas, coords, modes])
        new_obj_btn.click(add_new_object, [coords, modes, c_groups, m_groups], [coords, modes, c_groups, m_groups])

        segment_btn.click(
            execute_segmentation,
            [tracker, origin_frame, ref_frame, coords, modes, c_groups, m_groups, frame_idx],
            [tracker, canvas, origin_frame, coords, modes, c_groups, m_groups],
        )
        detect_btn.click(
            gd_detect,
            [tracker, origin_frame, ref_frame, caption, box_thresh, text_thresh, frame_idx],
            [tracker, canvas, origin_frame],
        )

        # Tracking events
        track_evt = track_btn.click(
            lambda: ("Tracking...", gr.update(interactive=False)), None, [status, canvas], queue=False
        ).then(
            tracking_objects,
            [tracker, current_frame, input_video, input_img_seq, fps, frame_idx, output_path],
            [output_frame, frame_idx, current_frame, output_path],
            concurrency_limit=1,
        )

        track_evt.then(_get_completed_output, [output_path], [output_file])
        track_evt.then(
            lambda p: "Tracking complete." if _get_completed_output(p) else "Tracking paused.",
            [output_path],
            [status],
        )
        track_evt.then(
            lambda p, frame: gr.update(value=_black_frame(frame), interactive=False) if _get_completed_output(p) else gr.update(),
            [output_path, current_frame],
            [canvas],
        )

        add_obj_track_btn.click(
            pause_and_load_frame_for_segmentation,
            [tracker, current_frame, frame_idx],
            [canvas, origin_frame, coords, modes, c_groups, m_groups],
            cancels=[track_evt],
        ).then(lambda: "Tracking paused — ready for new objects.", None, [status], queue=False)

        # Reset button events
        reset_btn.click(
            reset_SegTracker,
            [tracker, origin_frame, output_path],
            [tracker, canvas, coords, modes, c_groups, m_groups, caption, frame_idx, current_frame, ref_frame],
            queue=False,
        )
        reset_btn.click(lambda: (None, "Ready", gr.update(interactive=True)), None, [output_file, status, canvas], queue=False)

    return demo


if __name__ == "__main__":
    demo = app()
    demo.queue().launch(css=CUSTOM_CSS, debug=True, share=False)