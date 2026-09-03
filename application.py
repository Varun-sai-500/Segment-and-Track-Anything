import os
import tempfile
import zipfile

import cv2
import gradio as gr
import numpy as np

from model_args import deaot_args, dino_args, sam_args
from pipeline import Pipeline

# ------------------------------------------------------------------
# Pipeline Helpers
# ------------------------------------------------------------------


def ensure_pipeline(Seg_Tracker):
    if Seg_Tracker is None:
        Seg_Tracker = Pipeline(sam_args, dino_args, deaot_args)
    return Seg_Tracker


def draw_points_on_frame(points, modes, frame):
    for x, y in points[modes == 0]:
        cv2.circle(frame, (int(x), int(y)), 8, (255, 80, 80), -1)
    for x, y in points[modes == 1]:
        cv2.circle(frame, (int(x), int(y)), 8, (0, 153, 255), -1)
    return frame


def reset_pipeline_state(Seg_Tracker, output_path=None):
    if Seg_Tracker is not None:
        Seg_Tracker.restart_tracker()
    _discard_tracking_output(output_path)
    return Seg_Tracker, [], [], [], [], 0, None, None


# ------------------------------------------------------------------
# Input Handling
# ------------------------------------------------------------------


def get_meta_from_video(input_video, Seg_Tracker, output_path):
    if input_video is None:
        _discard_tracking_output(output_path)
        return None, None, "", Seg_Tracker, [], [], [], [], 0, None, None

    cap = cv2.VideoCapture(input_video)
    ret, first_frame = cap.read()
    cap.release()

    if not ret or first_frame is None:
        _discard_tracking_output(output_path)
        return None, None, "", Seg_Tracker, [], [], [], [], 0, None, None

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    (
        Seg_Tracker,
        current_coords,
        current_modes,
        coords_groups,
        modes_groups,
        current_frame_idx,
        current_frame,
        reference_frame,
    ) = reset_pipeline_state(Seg_Tracker, output_path)

    return (
        first_frame,
        first_frame,
        "",
        Seg_Tracker,
        current_coords,
        current_modes,
        coords_groups,
        modes_groups,
        current_frame_idx,
        first_frame,
        first_frame,
    )


def get_meta_from_img_seq(input_img_seq, Seg_Tracker, output_path):
    if input_img_seq is None:
        _discard_tracking_output(output_path)
        return None, None, "", Seg_Tracker, [], [], [], [], 0, None, None

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(input_img_seq.name, "r") as zf:
            zf.extractall(temp_dir)

        image_paths = [
            os.path.join(root, filename)
            for root, _, files in os.walk(temp_dir)
            for filename in files
            if os.path.splitext(filename)[1].lower() in valid_extensions
        ]
        image_paths = sorted(image_paths)

        if not image_paths:
            raise ValueError("Image sequence ZIP contains no images.")

        first_frame = cv2.imread(image_paths[0])
        if first_frame is None:
            raise ValueError(f"Failed to decode image: {image_paths[0]}")

        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    (
        Seg_Tracker,
        current_coords,
        current_modes,
        coords_groups,
        modes_groups,
        current_frame_idx,
        current_frame,
        reference_frame,
    ) = reset_pipeline_state(Seg_Tracker, output_path)

    return (
        first_frame,
        first_frame,
        "",
        Seg_Tracker,
        current_coords,
        current_modes,
        coords_groups,
        modes_groups,
        current_frame_idx,
        first_frame,
        first_frame,
    )


# ------------------------------------------------------------------
# Tracker Reference Handling
# ------------------------------------------------------------------


def _get_existing_mask(Seg_Tracker):
    if Seg_Tracker is None or Seg_Tracker.current_mask is None:
        return None
    return Seg_Tracker.get_current_mask()


def _extract_new_objects(previous_mask, current_mask):
    """Extract only object IDs that were not present in the previous reference mask."""
    if previous_mask is None:
        return current_mask.copy()

    previous_ids = set(np.unique(previous_mask).tolist())
    current_ids = set(np.unique(current_mask).tolist())

    previous_ids.discard(0)
    current_ids.discard(0)

    new_ids = current_ids - previous_ids
    if not new_ids:
        return np.zeros_like(current_mask, dtype=np.uint8)

    return np.where(np.isin(current_mask, list(new_ids)), current_mask, 0).astype(
        np.uint8
    )


def _commit_segmentation(
    Seg_Tracker, reference_frame, predicted_mask, frame_idx, previous_mask
):
    if predicted_mask is None:
        return

    predicted_mask = np.asarray(predicted_mask)

    if not np.any(predicted_mask):
        return

    if previous_mask is None:
        Seg_Tracker.initialize_reference(
            reference_frame, predicted_mask, frame_step=frame_idx
        )
        return

    new_objects_mask = _extract_new_objects(previous_mask, predicted_mask)

    if not np.any(new_objects_mask):
        return

    Seg_Tracker.add_objects(mask=new_objects_mask, frame_step=frame_idx)


# ------------------------------------------------------------------
# Interactive Point Handling
# ------------------------------------------------------------------


def record_click(
    origin_frame, point_mode, current_coords, current_modes, evt: gr.SelectData
):
    if origin_frame is None:
        return None, current_coords, current_modes

    mode = 1 if point_mode == "Positive" else 0
    coord = [evt.index[0], evt.index[1]]

    current_coords.append(coord)
    current_modes.append(mode)

    frame_display = draw_points_on_frame(
        points=np.array(current_coords),
        modes=np.array(current_modes),
        frame=origin_frame.copy(),
    )

    return frame_display, current_coords, current_modes


def undo_last_click(origin_frame, current_coords, current_modes):
    if origin_frame is None:
        return None, current_coords, current_modes

    if len(current_coords) > 0:
        current_coords.pop()
        current_modes.pop()

    if len(current_coords) > 0:
        frame_display = draw_points_on_frame(
            points=np.array(current_coords),
            modes=np.array(current_modes),
            frame=origin_frame.copy(),
        )
    else:
        frame_display = origin_frame.copy()

    return frame_display, current_coords, current_modes


def add_new_object(current_coords, current_modes, coords_groups, modes_groups):
    if len(current_coords) > 0:
        coords_groups.append(current_coords.copy())
        modes_groups.append(current_modes.copy())

    return [], [], coords_groups, modes_groups


# ------------------------------------------------------------------
# Segmentation
# ------------------------------------------------------------------


def execute_segmentation(
    Seg_Tracker,
    origin_frame,
    reference_frame,
    current_coords,
    current_modes,
    coords_groups,
    modes_groups,
    current_frame_idx,
):
    if origin_frame is None:
        return (
            Seg_Tracker,
            None,
            None,
            current_coords,
            current_modes,
            coords_groups,
            modes_groups,
        )

    if len(current_coords) > 0:
        coords_groups.append(current_coords.copy())
        modes_groups.append(current_modes.copy())

    if not coords_groups:
        return Seg_Tracker, origin_frame, origin_frame, [], [], [], []

    Seg_Tracker = ensure_pipeline(Seg_Tracker)
    previous_mask = _get_existing_mask(Seg_Tracker)

    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click(
        origin_frame=origin_frame,
        coords_groups=coords_groups,
        modes_groups=modes_groups,
    )

    target_idx = current_frame_idx if current_frame_idx is not None else 0
    _commit_segmentation(
        Seg_Tracker, reference_frame, predicted_mask, target_idx, previous_mask
    )

    return Seg_Tracker, masked_frame, masked_frame, [], [], [], []


# ------------------------------------------------------------------
# Grounding DINO + SAM
# ------------------------------------------------------------------


def gd_detect(
    Seg_Tracker,
    origin_frame,
    reference_frame,
    grounding_caption,
    box_threshold,
    text_threshold,
    current_frame_idx,
):
    if origin_frame is None or not grounding_caption:
        return Seg_Tracker, origin_frame, origin_frame

    Seg_Tracker = ensure_pipeline(Seg_Tracker)
    previous_mask = _get_existing_mask(Seg_Tracker)

    predicted_mask, masked_frame = Seg_Tracker.detect_and_seg(
        origin_frame, grounding_caption, box_threshold, text_threshold
    )

    target_idx = current_frame_idx if current_frame_idx is not None else 0
    _commit_segmentation(
        Seg_Tracker, reference_frame, predicted_mask, target_idx, previous_mask
    )

    return Seg_Tracker, masked_frame, masked_frame


# ------------------------------------------------------------------
# Application-Owned Temporary Tracked Video
# ------------------------------------------------------------------

_TRACKING_OUTPUTS = {}


def _create_tracking_output():
    fd, path = tempfile.mkstemp(prefix="tracked_", suffix=".mp4")
    os.close(fd)
    _TRACKING_OUTPUTS[path] = {
        "writer": None,
        "width": None,
        "height": None,
        "fps": None,
        "completed": False,
    }
    return path


def _discard_tracking_output(path):
    if path is None:
        return
    session = _TRACKING_OUTPUTS.pop(path, None)
    if session is not None:
        writer = session.get("writer")
        if writer is not None:
            writer.release()
    try:
        os.remove(path)
    except OSError:
        pass


def _output_fps(input_video, fps):
    if input_video is not None:
        cap = cv2.VideoCapture(input_video)
        try:
            if cap.isOpened():
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                if source_fps and np.isfinite(source_fps) and source_fps > 0:
                    return float(source_fps)
        finally:
            cap.release()
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        raise RuntimeError("Output FPS is invalid.")
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError("Output FPS must be greater than zero.")
    return fps


def _write_tracking_frame(path, frame, input_video, fps):
    session = _TRACKING_OUTPUTS.get(path)
    if session is None:
        raise RuntimeError("Tracking output session is unavailable.")

    height, width = frame.shape[:2]
    if session["writer"] is None:
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            _output_fps(input_video, fps),
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            _discard_tracking_output(path)
            raise RuntimeError("Could not create the temporary tracked video.")
        session["writer"] = writer
        session["width"] = width
        session["height"] = height

    if session["width"] != width or session["height"] != height:
        raise RuntimeError("Frame dimensions changed during tracking.")

    session["writer"].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _finish_tracking_output(path):
    session = _TRACKING_OUTPUTS.get(path)
    if session is None:
        return None

    writer = session.get("writer")
    if writer is not None:
        writer.release()
        session["writer"] = None

    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        _discard_tracking_output(path)
        return None

    session["completed"] = True
    return path


def _black_frame(frame):
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        return None
    return np.zeros_like(frame)


def _get_completed_output(path):
    session = _TRACKING_OUTPUTS.get(path)
    if (
        session is not None
        and session.get("completed")
        and os.path.isfile(path)
        and os.path.getsize(path) > 0
    ):
        return path
    return None


def _completed_status(path):
    return (
        "Tracking complete." if _get_completed_output(path) else "Tracking paused."
    )


def _completion_canvas(path, current_frame):
    if _get_completed_output(path) is None:
        return gr.update()
    return gr.update(value=_black_frame(current_frame), interactive=False)


# ------------------------------------------------------------------
# Tracking Controls
# ------------------------------------------------------------------


def pause_and_load_frame_for_segmentation(
    Seg_Tracker, current_frame, current_frame_idx
):
    if Seg_Tracker is not None:
        Seg_Tracker.stop_tracking()
    if current_frame is None:
        raise gr.Error("No tracked frame is available to edit.")
    return (
        gr.update(value=current_frame, interactive=True),
        current_frame,
        [],
        [],
        [],
        [],
    )


def tracking_objects(
    Seg_Tracker,
    current_frame,
    input_video,
    input_img_seq,
    fps,
    current_frame_idx,
    output_path,
):
    if Seg_Tracker is None or Seg_Tracker.current_mask is None:
        raise gr.Error(
            "Please perform initial segmentation or detection before tracking!"
        )

    if input_video is not None and input_img_seq is not None:
        raise gr.Error("Select either a video or an image sequence, not both.")
    if input_video is None and input_img_seq is None:
        raise gr.Error("Please provide a video or an image sequence.")

    try:
        start_idx = int(
            current_frame_idx if current_frame_idx is not None else 0
        )
    except (TypeError, ValueError):
        raise gr.Error("Current frame index is invalid.")
    if start_idx < 0:
        raise gr.Error("Current frame index cannot be negative.")

    session = _TRACKING_OUTPUTS.get(output_path) if output_path else None
    if output_path is None or session is None or session.get("completed"):
        output_path = _create_tracking_output()

    last_frame = current_frame
    last_frame_idx = start_idx
    saw_frame = False

    try:
        for masked_frame, frame_idx in Seg_Tracker.track_video_sequence(
            input_video=input_video,
            input_img_seq=input_img_seq,
            fps=fps,
            frame_num=start_idx,
        ):
            if masked_frame is None:
                continue

            masked_frame = np.asarray(masked_frame)
            if (
                masked_frame.ndim != 3
                or masked_frame.shape[2] != 3
                or masked_frame.size == 0
            ):
                raise RuntimeError("Pipeline returned an invalid frame.")

            _write_tracking_frame(
                output_path,
                masked_frame,
                input_video,
                fps,
            )

            last_frame = masked_frame
            last_frame_idx = int(frame_idx)
            saw_frame = True

            yield (
                masked_frame,
                last_frame_idx,
                masked_frame,
                output_path,
            )

        paused = Seg_Tracker._stop_event.is_set()

        if not paused and saw_frame:
            completed_path = _finish_tracking_output(output_path)
            if completed_path is None:
                raise RuntimeError("Tracked video could not be finalized.")

            black = _black_frame(last_frame)
            if black is not None:
                yield (
                    black,
                    last_frame_idx,
                    black,
                    completed_path,
                )

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Tracking failed: {str(e)}")


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------


def reset_SegTracker(Seg_Tracker, origin_frame, output_path):
    if Seg_Tracker is not None:
        Seg_Tracker.restart_tracker()

    _discard_tracking_output(output_path)

    frame = origin_frame.copy() if origin_frame is not None else None

    return (
        Seg_Tracker,
        frame,
        [],
        [],
        [],
        [],
        "",
        0,
        frame,
        None,
    )


# ------------------------------------------------------------------
# Front-End Application
# ------------------------------------------------------------------

custom_css = """
.yellow-btn {
    background-color: #f59e0b !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
}
.yellow-btn:hover {
    background-color: #d97706 !important;
}
"""


def app():
    with gr.Blocks(title="Segment and Track Anything") as app:
        gr.Markdown(
            """
            <div style="text-align:center;">
                <span style="font-size:2.5em; font-weight:bold;">Segment and Track Anything</span>
            </div>
            """
        )

        # App State
        current_coords = gr.State([])
        current_modes = gr.State([])
        coords_groups = gr.State([])
        modes_groups = gr.State([])
        origin_frame = gr.State(None)
        current_frame = gr.State(None)
        reference_frame = gr.State(None)
        Seg_Tracker = gr.State(None)
        current_frame_idx = gr.State(0)
        output_path = gr.State(None)

        # Top Layout
        with gr.Row():
            # Left Column: Inputs & Controls
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab(label="Video type input"):
                        input_video = gr.Video(
                            label="Input video", height=230
                        )

                    with gr.Tab(label="Image-Seq type input"):
                        with gr.Row():
                            input_img_seq = gr.File(
                                label="Input Image-Seq (Zip)",
                                height=150,
                                scale=3,
                            )
                            with gr.Column(scale=2, min_width=120):
                                extract_button = gr.Button(
                                    value="Extract", size="sm"
                                )
                                fps = gr.Slider(
                                    label="FPS",
                                    minimum=5,
                                    maximum=50,
                                    value=8,
                                    step=1,
                                )

                with gr.Tabs():
                    with gr.Tab(label="Segmentation-click"):
                        with gr.Row(equal_height=True):
                            point_mode = gr.Radio(
                                choices=["Positive", "Negative"],
                                value="Positive",
                                label="Point Prompt",
                                interactive=True,
                                scale=3,
                            )
                            click_undo_but = gr.Button(
                                value="Undo",
                                interactive=True,
                                scale=1,
                                min_width=70,
                                size="sm",
                            )
                        segment_button = gr.Button(
                            value="Segment / Add Object Reference",
                            elem_classes=["yellow-btn"],
                            interactive=True,
                        )
                        new_object_button = gr.Button(
                            value="Add Point Group (New Object)",
                            interactive=True,
                            scale=1,
                        )

                    with gr.Tab(label="Text Prompt Detection"):
                        grounding_caption = gr.Textbox(
                            label="Detection Prompt",
                            placeholder="Enter objects separated by fullstops",
                        )
                        with gr.Accordion("Advanced options", open=False):
                            with gr.Row():
                                box_threshold = gr.Slider(
                                    label="Box Threshold",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.25,
                                    step=0.001,
                                )
                                text_threshold = gr.Slider(
                                    label="Text Threshold",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.25,
                                    step=0.001,
                                )
                        detect_button = gr.Button(
                            value="Segment / Add Object Reference",
                            elem_classes=["yellow-btn"],
                            interactive=True,
                        )

                with gr.Row():
                    reset_button = gr.Button(
                        value="Reset All",
                        interactive=True,
                        variant="stop",
                        scale=1,
                    )

            # Right Column: Interactive Canvas
            with gr.Column(scale=1):
                input_first_frame = gr.Image(
                    label="Interactive Canvas",
                    interactive=True,
                    height=480,
                )

        # Tracking Controls
        gr.Markdown("---")
        gr.Markdown("### TRACK")

        with gr.Row():
            track_for_video = gr.Button(
                value="Start / Resume Tracking",
                interactive=True,
                variant="primary",
            )
            add_objects_button = gr.Button(
                value="Add Objects During Tracking",
                interactive=True,
                variant="secondary",
            )

        with gr.Row():
            output_frame = gr.Image(
                label="Live Tracking Progress",
                height=480,
                scale=8,
            )

            with gr.Column(scale=2, min_width=170):
                output_file = gr.File(
                    label="Download",
                    interactive=False,
                )
                tracking_status = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    lines=2,
                )

        # Upload Resets
        input_video.upload(
            fn=lambda: None,
            inputs=[],
            outputs=[output_file],
            queue=False,
        )

        input_img_seq.upload(
            fn=lambda: None,
            inputs=[],
            outputs=[output_file],
            queue=False,
        )

        # Event Listeners
        input_video.upload(
            fn=get_meta_from_video,
            inputs=[input_video, Seg_Tracker, output_path],
            outputs=[
                input_first_frame,
                origin_frame,
                grounding_caption,
                Seg_Tracker,
                current_coords,
                current_modes,
                coords_groups,
                modes_groups,
                current_frame_idx,
                current_frame,
                reference_frame,
            ],
        )

        input_img_seq.upload(
            fn=get_meta_from_img_seq,
            inputs=[input_img_seq, Seg_Tracker, output_path],
            outputs=[
                input_first_frame,
                origin_frame,
                grounding_caption,
                Seg_Tracker,
                current_coords,
                current_modes,
                coords_groups,
                modes_groups,
                current_frame_idx,
                current_frame,
                reference_frame,
            ],
        )

        extract_button.click(
            fn=get_meta_from_img_seq,
            inputs=[input_img_seq, Seg_Tracker, output_path],
            outputs=[
                input_first_frame,
                origin_frame,
                grounding_caption,
                Seg_Tracker,
                current_coords,
                current_modes,
                coords_groups,
                modes_groups,
                current_frame_idx,
                current_frame,
                reference_frame,
            ],
        )

        input_first_frame.select(
            fn=record_click,
            inputs=[origin_frame, point_mode, current_coords, current_modes],
            outputs=[input_first_frame, current_coords, current_modes],
        )

        click_undo_but.click(
            fn=undo_last_click,
            inputs=[origin_frame, current_coords, current_modes],
            outputs=[input_first_frame, current_coords, current_modes],
        )

        new_object_button.click(
            fn=add_new_object,
            inputs=[
                current_coords,
                current_modes,
                coords_groups,
                modes_groups,
            ],
            outputs=[
                current_coords,
                current_modes,
                coords_groups,
                modes_groups,
            ],
        )

        segment_button.click(
            fn=execute_segmentation,
            inputs=[
                Seg_Tracker,
                origin_frame,
                reference_frame,
                current_coords,
                current_modes,
                coords_groups,
                modes_groups,
                current_frame_idx,
            ],
            outputs=[
                Seg_Tracker,
                input_first_frame,
                origin_frame,
                current_coords,
                current_modes,
                coords_groups,
                modes_groups,
            ],
        )

        detect_button.click(
            fn=gd_detect,
            inputs=[
                Seg_Tracker,
                origin_frame,
                reference_frame,
                grounding_caption,
                box_threshold,
                text_threshold,
                current_frame_idx,
            ],
            outputs=[Seg_Tracker, input_first_frame, origin_frame],
        )

        reset_button.click(
            fn=reset_SegTracker,
            inputs=[Seg_Tracker, origin_frame, output_path],
            outputs=[
                Seg_Tracker,
                input_first_frame,
                current_coords,
                current_modes,
                coords_groups,
                modes_groups,
                grounding_caption,
                current_frame_idx,
                current_frame,
                reference_frame,
            ],
            queue=False,
        )

        reset_button.click(
            fn=lambda: (None, "Ready"),
            inputs=[],
            outputs=[output_file, tracking_status],
            queue=False,
        )

        reset_button.click(
            fn=lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[input_first_frame],
            queue=False,
        )

        tracking_status_event = track_for_video.click(
            fn=lambda: "Tracking...",
            inputs=[],
            outputs=[tracking_status],
            queue=False,
        )

        lock_canvas_event = tracking_status_event.then(
            fn=lambda: gr.update(interactive=False),
            inputs=[],
            outputs=[input_first_frame],
            queue=False,
        )

        tracking_event = lock_canvas_event.then(
            fn=tracking_objects,
            inputs=[
                Seg_Tracker,
                current_frame,
                input_video,
                input_img_seq,
                fps,
                current_frame_idx,
                output_path,
            ],
            outputs=[
                output_frame,
                current_frame_idx,
                current_frame,
                output_path,
            ],
            concurrency_limit=1,
        )

        tracking_event.then(
            fn=_get_completed_output,
            inputs=[output_path],
            outputs=[output_file],
        )

        tracking_event.then(
            fn=_completed_status,
            inputs=[output_path],
            outputs=[tracking_status],
        )

        tracking_event.then(
            fn=_completion_canvas,
            inputs=[output_path, current_frame],
            outputs=[input_first_frame],
        )

        add_objects_button.click(
            fn=pause_and_load_frame_for_segmentation,
            inputs=[Seg_Tracker, current_frame, current_frame_idx],
            outputs=[
                input_first_frame,
                origin_frame,
                current_coords,
                current_modes,
                coords_groups,
                modes_groups,
            ],
            cancels=[tracking_event],
        )

        add_objects_button.click(
            fn=lambda: "Tracking paused — ready for new objects.",
            inputs=[],
            outputs=[tracking_status],
            queue=False,
        )

    return app


if __name__ == "__main__":
    demo = app()
    demo.queue()
    demo.launch(css=custom_css, debug=True, share=False)