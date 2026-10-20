"""Compute worker for classical baselines, SAM 2.1 propagation, leave-one-out evaluation, and YOLO-seg training/prediction."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw

from preannotate_core import (
    draw_mask_overlay,
    extract_dataset_if_zip,
    merge_datasets,
    rasterize_image_annotations,
    resolve_image_path,
    safe_parse_xml,
)


def load_sequence(dataset_path: Path, label: str, scratch_dir: Path):
    base_dir, xml_path = extract_dataset_if_zip(dataset_path, scratch_dir)
    root = safe_parse_xml(xml_path)
    nodes = sorted(root.findall("image"), key=lambda n: int(n.attrib["id"]))
    if not nodes:
        raise ValueError(f"No <image> entries found in {xml_path}")
    frames = []
    gt = {}
    for idx, node in enumerate(nodes):
        img_path = resolve_image_path(base_dir, node.attrib["name"])
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise ValueError(f"Unable to read image: {img_path}")
        frames.append(frame)
        h, w = frame.shape[:2]
        mask, count = rasterize_image_annotations(node, h, w, label=label)
        if count > 0 and np.any(mask):
            gt[idx] = mask
    return root, nodes, frames, gt


def compute_crop_band(
    frames: list[np.ndarray],
    gt: dict[int, np.ndarray],
    crop_y_arg: str | None = None,
    margin_px: int = 90,
) -> tuple[int, int]:
    h = frames[0].shape[0]
    if crop_y_arg:
        parts = crop_y_arg.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid --crop-y '{crop_y_arg}' (expected Y0:Y1)")
        y0, y1 = int(parts[0]), int(parts[1])
        if not (0 <= y0 < y1 <= h):
            raise ValueError(f"Crop band {(y0, y1)} out of bounds for height {h}")
        return y0, y1
    ys_all = [np.where(m > 0)[0] for m in gt.values() if np.any(m > 0)]
    if not ys_all:
        return 0, h
    y_min = min(int(ys.min()) for ys in ys_all)
    y_max = max(int(ys.max()) for ys in ys_all)
    y0 = max(0, y_min - margin_px)
    y1 = min(h, y_max + margin_px)
    if y1 - y0 < 64:
        center = (y0 + y1) // 2
        y0 = max(0, center - 32)
        y1 = min(h, y0 + 64)
    return y0, y1


def compute_camera_and_evidence(
    frames: list[np.ndarray],
    gt: dict[int, np.ndarray],
    crop_band: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    h, w = frames[0].shape[:2]
    y0, y1 = crop_band
    orb = cv2.ORB_create(nfeatures=3500)
    static_mask = np.full((h, w), 255, dtype=np.uint8)
    if (y1 - y0) < int(0.85 * h):
        static_mask[y0:y1, :] = 0
    ref_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    kp0, d0 = orb.detectAndCompute(ref_gray, static_mask)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    transforms: list[np.ndarray] = []
    aligned: list[np.ndarray] = []
    registration: list[dict] = []

    for i, f in enumerate(frames):
        if i == 0 or d0 is None or len(kp0) < 12:
            H = np.eye(3, dtype=np.float64)
            inliers = 0
            error = 0.0
        else:
            kp, d = orb.detectAndCompute(
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), static_mask
            )
            if d is None or len(kp) < 12:
                H = transforms[-1].copy()
                inliers = 0
                error = 0.0
            else:
                matches = bf.knnMatch(d, d0, k=2)
                good = [
                    m[0]
                    for m in matches
                    if len(m) == 2 and m[0].distance < 0.75 * m[1].distance
                ]
                if len(good) < 12:
                    H = transforms[-1].copy()
                    inliers = len(good)
                    error = 0.0
                else:
                    a = np.float32([kp[m.queryIdx].pt for m in good])
                    b = np.float32([kp0[m.trainIdx].pt for m in good])
                    A, status = cv2.estimateAffinePartial2D(
                        a, b, method=cv2.RANSAC, ransacReprojThreshold=2.5
                    )
                    if A is None or int(status.sum()) < 8:
                        H = transforms[-1].copy()
                        inliers = 0 if status is None else int(status.sum())
                        error = 0.0
                    else:
                        H = np.vstack([A, [0.0, 0.0, 1.0]])
                        inliers = int(status.sum())
                        residuals = np.linalg.norm(
                            cv2.transform(a[:, None], A)[:, 0] - b, axis=1
                        )[status[:, 0] > 0]
                        error = float(np.median(residuals)) if len(residuals) else 0.0
        transforms.append(H)
        warp = cv2.warpPerspective(f, H, (w, h)).astype(np.float32)
        valid = (
            cv2.warpPerspective(
                np.ones((h, w), dtype=np.uint8), H, (w, h), flags=cv2.INTER_NEAREST
            )
            > 0
        )
        warp[~valid] = np.nan
        aligned.append(warp)
        registration.append(
            {
                "frame": i,
                "inliers": inliers,
                "median_error_px": round(error, 4),
                "to_frame0": H.tolist(),
            }
        )

    stack = np.asarray(aligned, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        background = np.nanpercentile(stack, 25, axis=0)
    background = np.nan_to_num(background).astype(np.uint8)

    first_prompt = min(gt)
    known_bg_ref = (
        cv2.dilate(
            gt[first_prompt].astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        == 0
    )
    ref_frame_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)

    evidences = []
    evidences_zero = []
    deltas = []
    for i, f in enumerate(frames):
        Hinv = np.linalg.inv(transforms[i])
        bg = cv2.warpPerspective(background, Hinv, (w, h))
        valid = (
            cv2.warpPerspective(
                np.ones((h, w), dtype=np.uint8), Hinv, (w, h), flags=cv2.INTER_NEAREST
            )
            > 0
        )
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
        delta = cv2.GaussianBlur(gray - bg_gray, (7, 7), 0)
        evidence = np.clip((delta - 5.0) / 25.0, 0.0, 1.0) * valid

        # Frame-0 known background variant
        f0_warped = cv2.warpPerspective(ref_frame_gray, Hinv, (w, h))
        known_warped = (
            cv2.warpPerspective(
                known_bg_ref.astype(np.uint8), Hinv, (w, h), flags=cv2.INTER_NEAREST
            )
            > 0
        ) & valid
        diff0 = gray - f0_warped
        offset = float(np.median(diff0[known_warped])) if np.any(known_warped) else 0.0
        delta0 = cv2.GaussianBlur(diff0 - offset, (7, 7), 0)
        ev0 = np.clip((delta0 - 5.0) / 25.0, 0.0, 1.0) * valid
        ev0[~known_warped] = 0.5

        if i > 0:
            prev = cv2.warpPerspective(frames[i - 1], Hinv @ transforms[i - 1], (w, h))
            diff = cv2.absdiff(
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
            )
            temporal = np.clip(
                cv2.GaussianBlur(diff.astype(np.float32), (7, 7), 0) / 25.0, 0.0, 1.0
            )
        else:
            temporal = np.zeros((h, w), dtype=np.float32)

        evidences.append(evidence)
        evidences_zero.append(ev0)
        deltas.append(temporal)

    return (
        np.asarray(evidences, dtype=np.float32),
        np.asarray(evidences_zero, dtype=np.float32),
        np.asarray(deltas, dtype=np.float32),
        registration,
    )


def run_baselines(dataset_path: Path, out_dir: Path, label: str = "smoke") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.setRNGSeed(42)
    cv2.setNumThreads(4)

    with TemporaryDirectory() as tmp:
        _, nodes, frames, gt = load_sequence(dataset_path, label, Path(tmp))
        if not gt:
            raise ValueError(f"Dataset has no manual annotations for label '{label}'")
        first_idx = min(gt)
        seed = gt[first_idx].copy()
        h, w = frames[0].shape[:2]
        methods = ["fluxo", "fluxo_grabcut", "aparencia_local"]
        masks: dict[str, list[np.ndarray]] = {
            m: [np.zeros((h, w), dtype=np.uint8) for _ in range(len(frames))]
            for m in methods
        }
        for m in methods:
            for idx, ref_mask in gt.items():
                masks[m][idx] = ref_mask.copy()

        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
        lab0 = cv2.cvtColor(frames[first_idx], cv2.COLOR_BGR2LAB).astype(np.float32)
        eroded = cv2.erode(seed, k(3))
        colors = lab0[eroded > 0] if np.any(eroded > 0) else lab0[seed > 0]
        mean = colors.mean(0)
        std = np.maximum(colors.std(0), [18.0, 5.0, 5.0])

        def clean(mask_arr: np.ndarray, support: np.ndarray) -> np.ndarray:
            closed = cv2.morphologyEx(mask_arr.astype(np.uint8), cv2.MORPH_CLOSE, k(3))
            count, labels_cc, stats, _ = cv2.connectedComponentsWithStats(closed)
            result = np.zeros_like(closed)
            for j in range(1, count):
                region = labels_cc == j
                if stats[j, cv2.CC_STAT_AREA] >= 12 and np.any(region & (support > 0)):
                    result[region] = 1
            return result

        for i in range(first_idx + 1, len(frames)):
            if i in gt:
                continue
            prev, cur = frames[i - 1], frames[i]
            a, b = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (prev, cur)]
            flow = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 4, 25, 5, 7, 1.5, 0)
            for method in methods:
                prior = cv2.remap(
                    masks[method][i - 1].astype(np.float32),
                    xx + flow[:, :, 0],
                    yy + flow[:, :, 1],
                    cv2.INTER_LINEAR,
                )
                support = (prior > 0.35).astype(np.uint8)
                if method == "fluxo":
                    res = support
                elif method == "fluxo_grabcut":
                    domain = cv2.dilate(support, k(25))
                    g = np.zeros((h, w), dtype=np.uint8)
                    g[domain > 0] = cv2.GC_PR_BGD
                    g[support > 0] = cv2.GC_PR_FGD
                    core = cv2.erode(support, k(5))
                    g[core > 0] = cv2.GC_FGD
                    if np.any(core):
                        cv2.grabCut(
                            cur,
                            g,
                            None,
                            np.zeros((1, 65)),
                            np.zeros((1, 65)),
                            3,
                            cv2.GC_INIT_WITH_MASK,
                        )
                        res = clean((g == 1) | (g == 3), support)
                    else:
                        res = support
                else:
                    lab = cv2.cvtColor(cur, cv2.COLOR_BGR2LAB).astype(np.float32)
                    distance = np.sqrt(np.mean(((lab - mean) / std) ** 2, axis=2))
                    domain = cv2.dilate(support, k(31)) > 0
                    res = clean(domain & (distance < 1.5), support)
                masks[method][i] = res

        metrics = []
        for method in methods:
            folder = out_dir / method
            (folder / "mascaras").mkdir(parents=True, exist_ok=True)
            (folder / "sobreposicoes").mkdir(parents=True, exist_ok=True)
            for i, (frame, mask) in enumerate(zip(frames, masks[method])):
                name = nodes[i].attrib["name"]
                cv2.imwrite(str(folder / "mascaras" / name), mask * 255)
                overlay = draw_mask_overlay(
                    frame, mask, (50, 235, 60) if i in gt else (40, 230, 255)
                )
                cv2.imwrite(str(folder / "sobreposicoes" / name), overlay)
                metrics.append(
                    {
                        "metodo": method,
                        "frame": i,
                        "arquivo": name,
                        "area_px": int(mask.sum()),
                    }
                )

        sample_ids = sorted(
            {
                0,
                min(1, len(frames) - 1),
                min(2, len(frames) - 1),
                len(frames) // 2,
                len(frames) - 1,
            }
        )
        cell_w = min(440, w)
        cell_h = min(190, h)
        sheet = Image.new(
            "RGB",
            (len(sample_ids) * cell_w, (len(methods) + 1) * (cell_h + 20)),
            (24, 24, 24),
        )
        draw = ImageDraw.Draw(sheet)
        for row, method in enumerate(["original"] + methods):
            for col, idx in enumerate(sample_ids):
                f = (
                    frames[idx]
                    if method == "original"
                    else cv2.imread(
                        str(
                            out_dir
                            / method
                            / "sobreposicoes"
                            / nodes[idx].attrib["name"]
                        )
                    )
                )
                resized = cv2.resize(f, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                sheet.paste(
                    Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)),
                    (col * cell_w, row * (cell_h + 20) + 20),
                )
                draw.text(
                    (col * cell_w + 6, row * (cell_h + 20) + 2),
                    f"{method} | frame {idx:02d}",
                    fill="white",
                )
        comp_path = out_dir / "comparacao_baselines.jpg"
        sheet.save(comp_path, quality=92)
        metrics_path = out_dir / "metricas.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return {
            "output_dir": str(out_dir),
            "methods": methods,
            "comparison_sheet": str(comp_path),
            "metrics_json": str(metrics_path),
            "frames": len(frames),
        }


def build_sam2_predictor(checkpoint: Path, config_name: str, device: str):
    from sam2.build_sam import build_sam2_video_predictor

    if not checkpoint.is_file():
        raise ValueError(f"SAM 2 checkpoint not found: {checkpoint}")
    return build_sam2_video_predictor(
        config_name,
        str(checkpoint.resolve()),
        device=device,
        apply_postprocessing=False,
    )


def infer_sam2_scores(
    predictor,
    frames: list[np.ndarray],
    prompts: dict[int, np.ndarray],
    crop_band: tuple[int, int],
    work_dir: Path,
    device: str,
    ref_Bank: list[tuple[np.ndarray, np.ndarray]] | None = None,
    limit: int | None = None,
) -> np.ndarray:
    import torch

    h, w = frames[0].shape[:2]
    y0, y1 = crop_band
    video_dir = work_dir / f"frames_jpeg_{y0}_{y1}"
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    bank = ref_Bank or []
    offset = len(bank)

    # Prepend reference bank frames (resized to target width x crop_height)
    crop_h = y1 - y0
    for r_idx, (r_frame, r_mask) in enumerate(bank):
        rf_resized = cv2.resize(r_frame, (w, crop_h), interpolation=cv2.INTER_AREA)
        Image.fromarray(cv2.cvtColor(rf_resized, cv2.COLOR_BGR2RGB)).save(
            video_dir / f"{r_idx:05d}.jpg", quality=100, subsampling=0
        )

    for i, f in enumerate(frames):
        Image.fromarray(cv2.cvtColor(f[y0:y1], cv2.COLOR_BGR2RGB)).save(
            video_dir / f"{offset + i:05d}.jpg", quality=100, subsampling=0
        )

    scores = np.full((len(frames), h, w), -32.0, dtype=np.float16)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_cuda
        else torch.cpu.amp.autocast(enabled=False)
    )
    with torch.inference_mode(), ctx:
        state = predictor.init_state(
            video_path=str(video_dir), offload_video_to_cpu=True
        )
        for r_idx, (_, r_mask) in enumerate(bank):
            rm_resized = (
                cv2.resize(
                    r_mask.astype(np.uint8),
                    (w, crop_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
            if np.any(rm_resized):
                predictor.add_new_mask(
                    state, frame_idx=r_idx, obj_id=1, mask=rm_resized
                )

        shifted_prompts = {}
        for p_idx, p_mask in prompts.items():
            cropped_mask = p_mask[y0:y1].astype(bool)
            shifted_prompts[offset + p_idx] = cropped_mask
            predictor.add_new_mask(
                state, frame_idx=offset + p_idx, obj_id=1, mask=cropped_mask
            )

        start_fwd = min(shifted_prompts) if shifted_prompts else 0
        for vid_idx, _, logits in predictor.propagate_in_video(
            state,
            start_frame_idx=start_fwd,
            max_frame_num_to_track=limit,
            reverse=False,
        ):
            target_idx = vid_idx - offset
            if 0 <= target_idx < len(frames):
                scores[target_idx, y0:y1] = (
                    logits[0, 0].float().cpu().numpy().astype(np.float16)
                )

        if shifted_prompts and max(shifted_prompts) > offset:
            start_rev = max(shifted_prompts)
            min_prompt = min(shifted_prompts)
            for vid_idx, _, logits in predictor.propagate_in_video(
                state,
                start_frame_idx=start_rev,
                max_frame_num_to_track=limit,
                reverse=True,
            ):
                if vid_idx >= min_prompt:
                    continue
                target_idx = vid_idx - offset
                if 0 <= target_idx < len(frames):
                    scores[target_idx, y0:y1] = (
                        logits[0, 0].float().cpu().numpy().astype(np.float16)
                    )
        del state
    if use_cuda:
        torch.cuda.empty_cache()
    gc.collect()
    return scores


def load_reference_bank(
    ref_datasets: list[Path], label: str, scratch_dir: Path, max_refs: int = 12
) -> list[tuple[np.ndarray, np.ndarray]]:
    bank: list[tuple[np.ndarray, np.ndarray]] = []
    for idx, ref_path in enumerate(ref_datasets):
        _, _, r_frames, r_gt = load_sequence(
            ref_path, label, scratch_dir / f"ref_{idx:02d}"
        )
        for f_idx in sorted(r_gt):
            bank.append((r_frames[f_idx], r_gt[f_idx]))
            if len(bank) >= max_refs:
                return bank
    return bank


def apply_fusion(
    scores: np.ndarray,
    fusion: str,
    threshold: float,
    evidences: np.ndarray | None = None,
    evidences_zero: np.ndarray | None = None,
    deltas: np.ndarray | None = None,
) -> np.ndarray:
    s = scores.astype(np.float32)
    if fusion == "none":
        return s > threshold
    if fusion == "background":
        if evidences is None:
            raise ValueError("Background evidence required for fusion='background'")
        return (s + 2.0 * (2.0 * evidences - 1.0)) > threshold
    if fusion == "background-zero":
        if evidences_zero is None:
            raise ValueError("Frame-0 evidence required for fusion='background-zero'")
        return (s + 2.0 * (2.0 * evidences_zero - 1.0)) > threshold
    if fusion == "temporal":
        if deltas is None:
            raise ValueError("Temporal evidence required for fusion='temporal'")
        return (s + 2.0 * (2.0 * deltas - 1.0)) > threshold
    raise ValueError(f"Unsupported fusion mode: {fusion}")


def run_propagate(
    dataset_path: Path,
    out_dir: Path,
    checkpoint: Path,
    config_name: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    label: str = "smoke",
    mode: str = "full",
    crop_y: str | None = None,
    fusion: str = "none",
    threshold: float = 0.0,
    prompts_Filter: list[int] | None = None,
    ref_datasets: list[Path] | None = None,
    device: str = "cuda",
) -> dict:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root, nodes, frames, gt = load_sequence(
            dataset_path, label, tmp_path / "target"
        )
        if prompts_Filter is not None:
            selected_gt = {i: gt[i] for i in prompts_Filter if i in gt}
        else:
            selected_gt = dict(gt)
        bank = (
            load_reference_bank(ref_datasets, label, tmp_path / "refs")
            if ref_datasets
            else []
        )
        if not selected_gt and not bank:
            raise ValueError(
                f"No manual annotations found for label '{label}' in target dataset or --ref-dataset"
            )

        h, w = frames[0].shape[:2]
        crop_band = (
            compute_crop_band(frames, selected_gt or gt, crop_y)
            if mode == "crop"
            else (0, h)
        )

        evidences = evidences_zero = deltas = None
        registration = None
        if fusion != "none":
            ev_band = compute_crop_band(frames, selected_gt or gt, crop_y)
            evidences, evidences_zero, deltas, registration = (
                compute_camera_and_evidence(
                    frames,
                    selected_gt or {0: np.zeros((h, w), dtype=np.uint8)},
                    ev_band,
                )
            )
            (out_dir / "registro_camera.json").write_text(
                json.dumps(registration, indent=2), encoding="utf-8"
            )

        started = time.time()
        predictor = build_sam2_predictor(checkpoint, config_name, device=device)
        scores = infer_sam2_scores(
            predictor=predictor,
            frames=frames,
            prompts=selected_gt,
            crop_band=crop_band,
            work_dir=tmp_path / "work",
            device=device,
            ref_Bank=bank,
        )
        elapsed = time.time() - started

        np.savez_compressed(out_dir / "logits.npz", scores=scores)
        pred_masks = apply_fusion(
            scores,
            fusion=fusion,
            threshold=threshold,
            evidences=evidences,
            evidences_zero=evidences_zero,
            deltas=deltas,
        )

        masks_dir = out_dir / "mascaras"
        overlays_dir = out_dir / "sobreposicoes"
        masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        export = ET.Element("annotations")
        ET.SubElement(export, "version").text = "1.1"
        if root.find("meta") is not None:
            export.append(ET.fromstring(ET.tostring(root.find("meta"))))

        frame_reports = []
        for i, node in enumerate(nodes):
            name = node.attrib["name"]
            mask = (
                selected_gt[i].copy()
                if i in selected_gt
                else pred_masks[i].astype(np.uint8)
            )
            cv2.imwrite(str(masks_dir / name), mask * 255)
            overlay = draw_mask_overlay(
                frames[i], mask, (50, 235, 60) if i in selected_gt else (0, 230, 255)
            )
            cv2.imwrite(str(overlays_dir / name), overlay)

            im_elem = ET.SubElement(export, "image", dict(node.attrib))
            if i in selected_gt:
                for child in list(node):
                    im_elem.append(ET.fromstring(ET.tostring(child)))
            else:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for c in contours:
                    if cv2.contourArea(c) < 4:
                        continue
                    approx = cv2.approxPolyDP(c, 0.7, True).reshape(-1, 2)
                    if len(approx) >= 3:
                        pts_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in approx)
                        ET.SubElement(
                            im_elem,
                            "polygon",
                            label=label,
                            source="auto",
                            occluded="0",
                            z_order="0",
                            points=pts_str,
                        )
            frame_reports.append(
                {
                    "frame": i,
                    "name": name,
                    "manual_prompt": i in selected_gt,
                    "area_px": int(mask.sum()),
                }
            )

        ET.indent(export)
        xml_out = out_dir / "predicoes_cvat_1.1.xml"
        ET.ElementTree(export).write(xml_out, encoding="utf-8", xml_declaration=True)

        report = {
            "output_dir": str(out_dir),
            "masks_dir": str(masks_dir),
            "overlays_dir": str(overlays_dir),
            "logits_npz": str(out_dir / "logits.npz"),
            "predictions_xml": str(xml_out),
            "label": label,
            "mode": mode,
            "crop_band_y": list(crop_band),
            "fusion": fusion,
            "threshold": threshold,
            "prompt_frames": sorted(selected_gt),
            "reference_bank_frames": len(bank),
            "elapsed_seconds": round(elapsed, 3),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu",
            "frames": frame_reports,
        }
        (out_dir / "relatorio.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report


def compute_metrics(mask: np.ndarray, truth: np.ndarray) -> dict:
    inter = int(np.logical_and(mask > 0, truth > 0).sum())
    union = int(np.logical_or(mask > 0, truth > 0).sum())
    m_sum = int((mask > 0).sum())
    t_sum = int((truth > 0).sum())
    return {
        "iou": round(inter / union, 6) if union else 1.0,
        "dice": round(2.0 * inter / (m_sum + t_sum), 6) if (m_sum + t_sum) else 1.0,
        "area_px": m_sum,
        "reference_px": t_sum,
    }


def run_evaluate(
    dataset_path: Path,
    out_dir: Path,
    checkpoint: Path,
    config_name: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    label: str = "smoke",
    crop_y: str | None = None,
    device: str = "cuda",
) -> dict:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    cv2.setRNGSeed(42)

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _, nodes, frames, gt = load_sequence(dataset_path, label, tmp_path / "target")
        if len(gt) < 2:
            raise ValueError(
                f"Leave-one-out evaluation requires at least 2 annotated frames (found {len(gt)})"
            )
        h, w = frames[0].shape[:2]
        crop_band = compute_crop_band(frames, gt, crop_y)
        evidences, evidences_zero, deltas, registration = compute_camera_and_evidence(
            frames, gt, crop_band
        )
        (out_dir / "registro_camera.json").write_text(
            json.dumps(registration, indent=2), encoding="utf-8"
        )

        predictor = build_sam2_predictor(checkpoint, config_name, device=device)
        started = time.time()
        records: list[dict] = []

        first_prompt = min(gt)
        single_scores = infer_sam2_scores(
            predictor,
            frames,
            {first_prompt: gt[first_prompt]},
            (0, h),
            tmp_path / "single",
            device=device,
        )
        np.savez_compressed(
            out_dir / "logits_baseline_1_prompt.npz", scores=single_scores
        )
        for idx in sorted(gt):
            if idx == first_prompt:
                continue
            records.append(
                {
                    "experimento": "baseline_1_prompt",
                    "metodo": "sam",
                    "frame_avaliado": idx,
                    "prompts": [first_prompt],
                    **compute_metrics(single_scores[idx] > 0, gt[idx]),
                }
            )

        max_gt_idx = max(gt)
        limit_loo = max_gt_idx + 1

        for use_crop in (False, True):
            mode_prefix = "recorte" if use_crop else "inteiro"
            band = crop_band if use_crop else (0, h)
            for held in sorted(gt):
                fold_prompts = {i: gt[i] for i in gt if i != held}
                scores = infer_sam2_scores(
                    predictor,
                    frames,
                    fold_prompts,
                    band,
                    tmp_path / f"loo_{mode_prefix}_{held:02d}",
                    device=device,
                    limit=limit_loo,
                )
                for variant_name, fusion_key in (
                    ("sam", "none"),
                    ("sam_fundo", "background"),
                    ("sam_fundo_zero", "background-zero"),
                    ("sam_temporal", "temporal"),
                ):
                    pred = apply_fusion(
                        scores,
                        fusion=fusion_key,
                        threshold=0.0,
                        evidences=evidences,
                        evidences_zero=evidences_zero,
                        deltas=deltas,
                    )
                    records.append(
                        {
                            "experimento": "leave_one_out",
                            "metodo": f"{mode_prefix}_{variant_name}",
                            "frame_avaliado": held,
                            "prompts": sorted(fold_prompts),
                            **compute_metrics(pred[held], gt[held]),
                        }
                    )

            final_scores = infer_sam2_scores(
                predictor,
                frames,
                gt,
                band,
                tmp_path / f"final_{mode_prefix}",
                device=device,
            )
            for variant_name, fusion_key in (
                ("sam", "none"),
                ("sam_fundo", "background"),
                ("sam_fundo_zero", "background-zero"),
            ):
                pred = apply_fusion(
                    final_scores,
                    fusion=fusion_key,
                    threshold=0.0,
                    evidences=evidences,
                    evidences_zero=evidences_zero,
                    deltas=deltas,
                )
                folder = out_dir / f"final_{mode_prefix}_{variant_name}"
                (folder / "mascaras").mkdir(parents=True, exist_ok=True)
                (folder / "sobreposicoes").mkdir(parents=True, exist_ok=True)
                for i, node in enumerate(nodes):
                    mask = gt[i].copy() if i in gt else pred[i].astype(np.uint8)
                    cv2.imwrite(
                        str(folder / "mascaras" / node.attrib["name"]), mask * 255
                    )
                    overlay = draw_mask_overlay(
                        frames[i], mask, (50, 235, 60) if i in gt else (0, 230, 255)
                    )
                    cv2.imwrite(
                        str(folder / "sobreposicoes" / node.attrib["name"]), overlay
                    )

        # Summarize metrics by method on frames != first_prompt
        by_method: dict[str, list[float]] = {}
        for r in records:
            if r["frame_avaliado"] == first_prompt:
                continue
            by_method.setdefault(r["metodo"], []).append(r["iou"])
        summary_table = [
            {
                "metodo": m,
                "iou_medio": round(float(np.mean(vals)), 6),
                "folds_avaliados": len(vals),
            }
            for m, vals in sorted(
                by_method.items(), key=lambda kv: float(np.mean(kv[1])), reverse=True
            )
        ]

        elapsed = time.time() - started
        report = {
            "output_dir": str(out_dir),
            "annotated_frames": sorted(gt),
            "crop_band_y": list(crop_band),
            "summary_by_method": summary_table,
            "records": records,
            "elapsed_seconds": round(elapsed, 3),
            "torch": torch.__version__,
        }
        (out_dir / "avaliacao.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (out_dir / "resumo_metricas.json").write_text(
            json.dumps(summary_table, indent=2), encoding="utf-8"
        )
        return report


def run_train_seg(
    dataset_path: Path,
    out_dir: Path,
    label: str = "smoke",
    base_model: str = "yolo11n-seg.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "0",
) -> dict:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = dataset_path / "yolo_seg" / "dataset.yaml"
    if not yaml_path.is_file():
        merged_dir = out_dir / "prepared_dataset"
        merge_info = merge_datasets(
            [dataset_path],
            merged_dir,
            label=label,
            export_yolo_seg=True,
            val_ratio=0.2,
        )
        yaml_path = Path(merge_info["yolo_seg"]["dataset_yaml"])

    model = YOLO(base_model)
    started = time.time()
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out_dir),
        name="yolo_seg_run",
        exist_ok=True,
        workers=0,
        seed=42,
    )
    elapsed = time.time() - started
    run_dir = out_dir / "yolo_seg_run"
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    chosen_pt = best_pt if best_pt.is_file() else last_pt
    target_pt = out_dir / "best.pt"
    if chosen_pt.is_file():
        shutil.copy2(chosen_pt, target_pt)

    summary = {
        "output_dir": str(out_dir),
        "weights_best": str(target_pt),
        "dataset_yaml": str(yaml_path),
        "base_model": base_model,
        "epochs": epochs,
        "imgsz": imgsz,
        "elapsed_seconds": round(elapsed, 3),
    }
    (out_dir / "treino_relatorio.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_predict_seg(
    dataset_path: Path,
    weights_path: Path,
    out_dir: Path,
    label: str = "smoke",
    conf: float = 0.25,
    imgsz: int = 640,
    refine_sam2: bool = False,
    sam2_checkpoint: Path | None = None,
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cuda",
) -> dict:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path.expanduser().resolve()))

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _, nodes, frames, gt = load_sequence(dataset_path, label, tmp_path / "target")
        h, w = frames[0].shape[:2]
        yolo_masks: dict[int, np.ndarray] = dict(gt)

        for idx, frame in enumerate(frames):
            if idx in gt:
                continue
            preds = model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                verbose=False,
            )
            mask = np.zeros((h, w), dtype=np.uint8)
            if preds and preds[0].masks is not None:
                for seg in preds[0].masks.xy:
                    if len(seg) >= 3:
                        cv2.fillPoly(mask, [np.rint(seg).astype(np.int32)], 1)
            if np.any(mask):
                yolo_masks[idx] = mask

        if refine_sam2:
            if sam2_checkpoint is None or not sam2_checkpoint.is_file():
                raise ValueError("--refine-sam2 requires a valid --checkpoint")
            if not yolo_masks:
                raise ValueError("No masks detected or provided to prompt SAM 2")
            predictor = build_sam2_predictor(
                sam2_checkpoint, sam2_config, device=device
            )
            scores = infer_sam2_scores(
                predictor,
                frames,
                yolo_masks,
                (0, h),
                tmp_path / "sam2_refine",
                device=device,
            )
            final_masks = [
                gt[i].copy() if i in gt else (scores[i] > 0).astype(np.uint8)
                for i in range(len(frames))
            ]
        else:
            final_masks = [
                yolo_masks.get(i, np.zeros((h, w), dtype=np.uint8))
                for i in range(len(frames))
            ]

        masks_dir = out_dir / "mascaras"
        overlays_dir = out_dir / "sobreposicoes"
        masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        frame_records = []
        for i, node in enumerate(nodes):
            name = node.attrib["name"]
            mask = final_masks[i]
            cv2.imwrite(str(masks_dir / name), mask * 255)
            overlay = draw_mask_overlay(
                frames[i], mask, (50, 235, 60) if i in gt else (0, 230, 255)
            )
            cv2.imwrite(str(overlays_dir / name), overlay)
            frame_records.append(
                {
                    "frame": i,
                    "name": name,
                    "manual_prompt": i in gt,
                    "area_px": int(mask.sum()),
                }
            )

        summary = {
            "output_dir": str(out_dir),
            "masks_dir": str(masks_dir),
            "overlays_dir": str(overlays_dir),
            "weights": str(weights_path),
            "refined_with_sam2": refine_sam2,
            "frames": frame_records,
        }
        (out_dir / "predicao_relatorio.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary


def worker_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Internal compute worker for preannotate")
    sub = p.add_subparsers(dest="action", required=True)

    b = sub.add_parser("baselines")
    b.add_argument("--dataset", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--label", default="smoke")

    pr = sub.add_parser("propagate")
    pr.add_argument("--dataset", type=Path, required=True)
    pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--checkpoint", type=Path, required=True)
    pr.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    pr.add_argument("--label", default="smoke")
    pr.add_argument("--mode", choices=["full", "crop"], default="full")
    pr.add_argument("--crop-y")
    pr.add_argument(
        "--fusion",
        choices=["none", "background", "background-zero", "temporal"],
        default="none",
    )
    pr.add_argument("--threshold", type=float, default=0.0)
    pr.add_argument("--prompts")
    pr.add_argument("--ref-dataset", action="append", default=[])
    pr.add_argument("--device", default="cuda")

    ev = sub.add_parser("evaluate")
    ev.add_argument("--dataset", type=Path, required=True)
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ev.add_argument("--label", default="smoke")
    ev.add_argument("--crop-y")
    ev.add_argument("--device", default="cuda")

    tr = sub.add_parser("train-seg")
    tr.add_argument("--dataset", type=Path, required=True)
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--label", default="smoke")
    tr.add_argument("--model", default="yolo11n-seg.pt")
    tr.add_argument("--epochs", type=int, default=25)
    tr.add_argument("--imgsz", type=int, default=640)
    tr.add_argument("--batch", type=int, default=8)
    tr.add_argument("--device", default="0")

    ps = sub.add_parser("predict-seg")
    ps.add_argument("--dataset", type=Path, required=True)
    ps.add_argument("--weights", type=Path, required=True)
    ps.add_argument("--out", type=Path, required=True)
    ps.add_argument("--label", default="smoke")
    ps.add_argument("--conf", type=float, default=0.25)
    ps.add_argument("--imgsz", type=int, default=640)
    ps.add_argument("--refine-sam2", action="store_true")
    ps.add_argument("--checkpoint", type=Path)
    ps.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ps.add_argument("--device", default="cuda")

    return p


def main(argv: list[str] | None = None) -> None:
    args = worker_parser().parse_args(argv)
    if args.action == "baselines":
        res = run_baselines(args.dataset, args.out, label=args.label)
    elif args.action == "propagate":
        prompts = (
            [int(x) for x in args.prompts.split(",") if x.strip()]
            if args.prompts
            else None
        )
        refs = [Path(r) for r in args.ref_dataset] if args.ref_dataset else None
        res = run_propagate(
            dataset_path=args.dataset,
            out_dir=args.out,
            checkpoint=args.checkpoint,
            config_name=args.config,
            label=args.label,
            mode=args.mode,
            crop_y=args.crop_y,
            fusion=args.fusion,
            threshold=args.threshold,
            prompts_Filter=prompts,
            ref_datasets=refs,
            device=args.device,
        )
    elif args.action == "evaluate":
        res = run_evaluate(
            dataset_path=args.dataset,
            out_dir=args.out,
            checkpoint=args.checkpoint,
            config_name=args.config,
            label=args.label,
            crop_y=args.crop_y,
            device=args.device,
        )
    elif args.action == "train-seg":
        res = run_train_seg(
            dataset_path=args.dataset,
            out_dir=args.out,
            label=args.label,
            base_model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    elif args.action == "predict-seg":
        res = run_predict_seg(
            dataset_path=args.dataset,
            weights_path=args.weights,
            out_dir=args.out,
            label=args.label,
            conf=args.conf,
            imgsz=args.imgsz,
            refine_sam2=args.refine_sam2,
            sam2_checkpoint=args.checkpoint,
            sam2_config=args.config,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown action: {args.action}")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
