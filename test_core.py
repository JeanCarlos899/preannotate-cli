import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import cv2
import numpy as np

from preannotate_core import (
    export_proposals,
    inspect_dataset,
    merge_datasets,
    render_video,
    run_smooth,
    run_thresholds,
    smooth_contour,
)
from preannotate_worker import run_baselines


def create_synthetic_cvat_dataset(
    root: Path, annotated_indices: tuple[int, ...] = (0,)
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    ann = ET.Element("annotations")
    ET.SubElement(ann, "version").text = "1.1"
    meta = ET.SubElement(ann, "meta")
    task = ET.SubElement(meta, "task")
    labels = ET.SubElement(task, "labels")
    lbl = ET.SubElement(labels, "label")
    ET.SubElement(lbl, "name").text = "smoke"

    for i in range(4):
        name = f"frame_{i * 10:06d}.png"
        img = np.full((120, 160, 3), 30, dtype=np.uint8)
        cv2.circle(img, (50 + i * 6, 60), 18, (200, 200, 200), -1)
        cv2.imwrite(str(root / name), img)
        im_node = ET.SubElement(
            ann,
            "image",
            {"id": str(i), "name": name, "width": "160", "height": "120"},
        )
        if i in annotated_indices:
            cx = 50 + i * 6
            pts = f"{cx - 15},45;{cx + 15},45;{cx + 15},75;{cx - 15},75"
            ET.SubElement(
                im_node,
                "polygon",
                {
                    "label": "smoke",
                    "source": "manual",
                    "occluded": "0",
                    "z_order": "0",
                    "points": pts,
                },
            )
    ET.indent(ann)
    ET.ElementTree(ann).write(
        root / "annotations.xml", encoding="utf-8", xml_declaration=True
    )
    return root


class PreannotateCoreTests(unittest.TestCase):
    def test_inspect_and_merge_datasets(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            ds1 = create_synthetic_cvat_dataset(tmp / "ds1", annotated_indices=(0, 1))
            ds2 = create_synthetic_cvat_dataset(tmp / "ds2", annotated_indices=(2,))

            info = inspect_dataset(ds1, label="smoke")
            self.assertEqual(info["total_frames"], 4)
            self.assertEqual(info["annotated_frames"], [0, 1])
            self.assertEqual(info["empty_frames"], [2, 3])

            merged_out = tmp / "merged"
            merged = merge_datasets(
                [ds1, ds2],
                merged_out,
                label="smoke",
                export_yolo_seg=True,
                val_ratio=0.33,
            )
            self.assertEqual(merged["total_frames"], 8)
            self.assertEqual(merged["annotated_frames"], 3)
            self.assertTrue((merged_out / "annotations.xml").is_file())
            self.assertTrue((merged_out / "yolo_seg" / "dataset.yaml").is_file())

    def test_smooth_and_export_proposals_preserve_manual_frames(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            ds = create_synthetic_cvat_dataset(tmp / "ds", annotated_indices=(0,))
            masks_dir = tmp / "raw_masks"
            masks_dir.mkdir()

            for i in range(4):
                name = f"frame_{i * 10:06d}.png"
                m = np.zeros((120, 160), dtype=np.uint8)
                if i < 3:
                    cv2.rectangle(m, (35 + i * 5, 45), (65 + i * 5, 75), 255, -1)
                cv2.imwrite(str(masks_dir / name), m)

            smooth_out = tmp / "smoothed"
            sm_res = run_smooth(
                dataset_path=ds,
                masks_dir=masks_dir,
                out_dir=smooth_out,
                sigma=0.8,
                compare_levels=[0.0, 0.8, 1.2],
                compare_frames=[1],
            )
            self.assertEqual(sm_res["frames_processed"], 4)
            self.assertEqual(len(sm_res["comparison_sheets"]), 1)

            exp_out = tmp / "proposals"
            exp_res = export_proposals(
                dataset_path=ds,
                masks_dir=masks_dir,
                out_dir=exp_out,
                label="smoke",
                sigma=0.8,
            )
            self.assertEqual(exp_res["preserved_frames"], [0])
            self.assertEqual(exp_res["filled_frames"], [1, 2])
            self.assertEqual(exp_res["empty_frames"], [3])

            payload = json.loads(
                (exp_out / "proposals.json").read_text(encoding="utf-8")
            )
            frames_in_json = {s["frame"] for s in payload["shapes"]}
            self.assertNotIn(0, frames_in_json)
            self.assertEqual(frames_in_json, {1, 2})

            with ZipFile(exp_out / "predictions_cvat_1.1.zip") as zf:
                self.assertIsNone(zf.testzip())

    def test_thresholds_baselines_and_video(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            ds = create_synthetic_cvat_dataset(tmp / "ds", annotated_indices=(0,))

            # Test classical baselines locally
            base_out = tmp / "baselines"
            base_res = run_baselines(ds, base_out, label="smoke")
            self.assertEqual(base_res["frames"], 4)
            self.assertTrue((base_out / "comparacao_baselines.jpg").is_file())

            # Test thresholds
            logits = np.full((4, 120, 160), -5.0, dtype=np.float16)
            logits[:, 40:70, 40:70] = 3.0
            logits_file = tmp / "logits.npz"
            np.savez_compressed(logits_file, scores=logits)
            th_out = tmp / "thresholds"
            th_res = run_thresholds(
                ds, logits_file, th_out, levels=[0.0, 2.0, 4.0], sample_frames=[0, 2]
            )
            self.assertTrue(Path(th_res["sheet"]).is_file())

            # Test video rendering
            vid_out = tmp / "video"
            vid_res = render_video(
                ds,
                [base_out / "fluxo" / "mascaras"],
                vid_out,
                titles=["Fluxo"],
                fps=4,
                gif=True,
            )
            self.assertTrue(Path(vid_res["video_mp4"]).is_file())
            self.assertTrue(Path(vid_res["video_gif"]).is_file())

    def test_contour_smoothing_geometry(self):
        square = np.array(
            [[[20, 20]], [[60, 20]], [[60, 60]], [[20, 60]]], dtype=np.int32
        )
        smoothed = smooth_contour(square, sigma=0.8)
        self.assertGreaterEqual(len(smoothed), 8)
        self.assertTrue(np.isfinite(smoothed).all())


if __name__ == "__main__":
    unittest.main()
