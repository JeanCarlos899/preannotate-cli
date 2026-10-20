"""Core local operations for dataset inspection, merging, smoothing, visualization, and CVAT proposal export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def ensure_new_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise ValueError(
            f"Output path already exists (refusing to overwrite): {resolved}"
        )
    return resolved


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_parse_xml(xml_path: Path) -> ET.Element:
    text = xml_path.read_text(encoding="utf-8")
    lowered = text.casefold()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise ValueError(f"Refusing XML with DTD/ENTITY declarations: {xml_path}")
    return ET.fromstring(text)


def resolve_image_path(base_dir: Path, name: str) -> Path:
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe image path in XML: {name}")
    candidates = [
        base_dir / rel,
        base_dir / "images" / rel,
        base_dir / "images" / "default" / rel,
        base_dir / rel.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"Image file not found for XML entry '{name}' under {base_dir}")


def extract_dataset_if_zip(dataset_path: Path, scratch_dir: Path) -> tuple[Path, Path]:
    dataset_path = dataset_path.expanduser().resolve()
    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        extract_root = scratch_dir / dataset_path.stem
        extract_root.mkdir(parents=True, exist_ok=True)
        with ZipFile(dataset_path) as zf:
            for member in zf.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe ZIP entry: {member.filename}")
            zf.extractall(extract_root)
        xml_path = extract_root / "annotations.xml"
        if not xml_path.is_file():
            raise ValueError(f"annotations.xml not found inside {dataset_path}")
        return extract_root, xml_path
    if dataset_path.is_dir():
        xml_path = dataset_path / "annotations.xml"
        if not xml_path.is_file():
            raise ValueError(f"annotations.xml not found in directory {dataset_path}")
        return dataset_path, xml_path
    raise ValueError(
        f"Dataset must be a directory or CVAT 1.1 .zip file: {dataset_path}"
    )


def polygon_points_from_attr(points_attr: str) -> np.ndarray:
    pairs = []
    for pair in points_attr.strip().split(";"):
        if not pair.strip():
            continue
        x_str, y_str = pair.split(",")
        pairs.append([float(x_str), float(y_str)])
    pts = np.asarray(pairs, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError(f"Invalid polygon points: {points_attr}")
    return pts


def rasterize_image_annotations(
    image_node: ET.Element, height: int, width: int, label: str | None = None
) -> tuple[np.ndarray, int]:
    mask = np.zeros((height, width), dtype=np.uint8)
    count = 0
    for poly in image_node.findall("polygon"):
        if label is not None and poly.attrib.get("label") != label:
            continue
        pts = polygon_points_from_attr(poly.attrib["points"])
        cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 1)
        count += 1
    for box in image_node.findall("box"):
        if label is not None and box.attrib.get("label") != label:
            continue
        xtl = int(round(float(box.attrib["xtl"])))
        ytl = int(round(float(box.attrib["ytl"])))
        xbr = int(round(float(box.attrib["xbr"])))
        ybr = int(round(float(box.attrib["ybr"])))
        cv2.rectangle(mask, (xtl, ytl), (xbr, ybr), 1, thickness=-1)
        count += 1
    return mask, count


def has_any_annotation(image_node: ET.Element, label: str | None = None) -> bool:
    tags = ("polygon", "box", "polyline", "points", "ellipse", "cuboid", "tag", "mask")
    for tag in tags:
        for elem in image_node.findall(tag):
            if label is None or elem.attrib.get("label") == label:
                return True
    return False


def inspect_dataset(
    dataset_path: Path, xml_override: Path | None = None, label: str | None = "smoke"
) -> dict:
    with TemporaryDirectory() as tmp:
        base_dir, xml_path = extract_dataset_if_zip(dataset_path, Path(tmp))
        if xml_override is not None:
            xml_path = xml_override.expanduser().resolve()
        root = safe_parse_xml(xml_path)
        nodes = sorted(root.findall("image"), key=lambda n: int(n.attrib["id"]))
        if not nodes:
            raise ValueError(f"No <image> entries found in {xml_path}")
        width = int(nodes[0].attrib["width"])
        height = int(nodes[0].attrib["height"])
        labels_found: set[str] = set()
        annotated_frames: list[int] = []
        empty_frames: list[int] = []
        frames_info: list[dict] = []

        for idx, node in enumerate(nodes):
            cvat_id = int(node.attrib["id"])
            name = node.attrib["name"]
            img_path = resolve_image_path(base_dir, name)
            for child in list(node):
                if "label" in child.attrib:
                    labels_found.add(child.attrib["label"])
            mask, shape_count = rasterize_image_annotations(
                node, height, width, label=label
            )
            occupied_any = has_any_annotation(node, label=None)
            if shape_count > 0:
                annotated_frames.append(idx)
            elif not occupied_any:
                empty_frames.append(idx)
            frames_info.append(
                {
                    "sequence_index": idx,
                    "cvat_id": cvat_id,
                    "name": name,
                    "sha256": sha256_file(img_path),
                    "target_shapes": shape_count,
                    "target_area_px": int(mask.sum()),
                    "occupied_any_label": occupied_any,
                }
            )

        return {
            "dataset": str(dataset_path.expanduser().resolve()),
            "xml": str(xml_path),
            "target_label": label,
            "total_frames": len(nodes),
            "width": width,
            "height": height,
            "labels_present": sorted(labels_found),
            "annotated_frames": annotated_frames,
            "empty_frames": empty_frames,
            "frames": frames_info,
        }


def merge_datasets(
    inputs: list[Path],
    out_dir: Path,
    label: str = "smoke",
    export_yolo_seg: bool = False,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    if not inputs:
        raise ValueError("Provide at least one dataset to merge.")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0.0, 1.0).")
    out = ensure_new_path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    images_out = out
    merged_root = ET.Element("annotations")
    ET.SubElement(merged_root, "version").text = "1.1"
    meta = ET.SubElement(merged_root, "meta")
    task_meta = ET.SubElement(meta, "task")
    labels_meta = ET.SubElement(task_meta, "labels")
    label_elem = ET.SubElement(labels_meta, "label")
    ET.SubElement(label_elem, "name").text = label

    manifest_entries: list[dict] = []
    global_idx = 0
    seen_hashes: set[str] = set()

    with TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for source_idx, item in enumerate(inputs):
            base_dir, xml_path = extract_dataset_if_zip(
                item, tmp_root / f"src_{source_idx:02d}"
            )
            root = safe_parse_xml(xml_path)
            nodes = sorted(root.findall("image"), key=lambda n: int(n.attrib["id"]))
            for local_idx, node in enumerate(nodes):
                orig_name = node.attrib["name"]
                img_path = resolve_image_path(base_dir, orig_name)
                digest = sha256_file(img_path)
                ext = img_path.suffix.lower() or ".png"
                safe_stem = Path(orig_name).stem.replace(" ", "_")
                new_name = f"src{source_idx:02d}_f{local_idx:04d}_{safe_stem}{ext}"
                dest_img = images_out / new_name
                shutil.copy2(img_path, dest_img)

                h = int(node.attrib["height"])
                w = int(node.attrib["width"])
                mask, shape_count = rasterize_image_annotations(node, h, w, label=label)

                new_node = ET.SubElement(
                    merged_root,
                    "image",
                    {
                        "id": str(global_idx),
                        "name": new_name,
                        "width": str(w),
                        "height": str(h),
                    },
                )
                polygons_norm: list[list[float]] = []
                for poly in node.findall("polygon"):
                    if poly.attrib.get("label") != label:
                        continue
                    new_node.append(ET.fromstring(ET.tostring(poly)))
                    pts = polygon_points_from_attr(poly.attrib["points"])
                    norm = np.column_stack(
                        [
                            np.clip(pts[:, 0] / max(1, w), 0.0, 1.0),
                            np.clip(pts[:, 1] / max(1, h), 0.0, 1.0),
                        ]
                    )
                    polygons_norm.append(np.round(norm.reshape(-1), 6).tolist())

                manifest_entries.append(
                    {
                        "merged_id": global_idx,
                        "name": new_name,
                        "source_index": source_idx,
                        "source_dataset": str(item.expanduser().resolve()),
                        "source_frame_id": int(node.attrib["id"]),
                        "source_name": orig_name,
                        "width": w,
                        "height": h,
                        "sha256": digest,
                        "duplicate_hash": digest in seen_hashes,
                        "annotated": shape_count > 0,
                        "area_px": int(mask.sum()),
                        "polygons_norm": polygons_norm,
                    }
                )
                seen_hashes.add(digest)
                global_idx += 1

    ET.indent(merged_root)
    xml_out = out / "annotations.xml"
    ET.ElementTree(merged_root).write(xml_out, encoding="utf-8", xml_declaration=True)

    yolo_summary = None
    if export_yolo_seg:
        yolo_dir = out / "yolo_seg"
        for split in ("train", "val"):
            (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        annotated_items = [
            e for e in manifest_entries if e["annotated"] and not e["duplicate_hash"]
        ]
        rng = np.random.default_rng(seed)
        indices = np.arange(len(annotated_items))
        rng.shuffle(indices)
        val_count = int(round(len(annotated_items) * val_ratio))
        if len(annotated_items) > 1 and val_ratio > 0 and val_count == 0:
            val_count = 1
        val_set = set(indices[:val_count].tolist())

        train_count = 0
        val_actual = 0
        for pos, entry in enumerate(annotated_items):
            split = "val" if pos in val_set else "train"
            if split == "val":
                val_actual += 1
            else:
                train_count += 1
            src_img = images_out / entry["name"]
            dst_img = yolo_dir / "images" / split / entry["name"]
            shutil.copy2(src_img, dst_img)
            lbl_path = yolo_dir / "labels" / split / f"{Path(entry['name']).stem}.txt"
            lines = [
                "0 " + " ".join(f"{coord:.6f}" for coord in poly)
                for poly in entry["polygons_norm"]
                if len(poly) >= 6
            ]
            lbl_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

        yaml_path = yolo_dir / "dataset.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    f"path: {yolo_dir}",
                    "train: images/train",
                    "val: images/val",
                    "names:",
                    f"  0: {label}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        yolo_summary = {
            "yolo_dir": str(yolo_dir),
            "dataset_yaml": str(yaml_path),
            "train_images": train_count,
            "val_images": val_actual,
        }

    summary = {
        "output_dir": str(out),
        "annotations_xml": str(xml_out),
        "label": label,
        "sources_count": len(inputs),
        "total_frames": len(manifest_entries),
        "annotated_frames": sum(1 for e in manifest_entries if e["annotated"]),
        "unannotated_frames": sum(1 for e in manifest_entries if not e["annotated"]),
        "yolo_seg": yolo_summary,
    }
    (out / "merge_manifest.json").write_text(
        json.dumps(
            {
                **summary,
                "entries": [
                    {k: v for k, v in e.items() if k != "polygons_norm"}
                    for e in manifest_entries
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def smooth_contour(contour: np.ndarray, sigma: float, step: float = 0.4) -> np.ndarray:
    points = contour.reshape(-1, 2).astype(np.float64)
    if sigma <= 0 or len(points) < 3:
        return points
    closed = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    if cumulative[-1] <= 1e-6:
        return points
    num_samples = max(8, int(np.ceil(cumulative[-1] / step)))
    samples = np.linspace(0.0, cumulative[-1], num_samples, endpoint=False)
    xy = np.column_stack(
        [np.interp(samples, cumulative, closed[:, j]) for j in range(2)]
    )
    spacing = cumulative[-1] / len(samples)
    s = sigma / max(spacing, 1e-6)
    radius = max(1, int(np.ceil(3.0 * s)))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (axis / s) ** 2)
    kernel /= kernel.sum()
    return np.column_stack(
        [
            np.convolve(np.pad(xy[:, j], radius, mode="wrap"), kernel, mode="valid")
            for j in range(2)
        ]
    )


def load_font(size: int = 20):
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def run_smooth(
    dataset_path: Path,
    masks_dir: Path,
    out_dir: Path,
    sigma: float = 0.8,
    compare_levels: list[float] | None = None,
    compare_frames: list[int] | None = None,
    step: float = 0.4,
) -> dict:
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    masks_dir = masks_dir.expanduser().resolve()
    if not masks_dir.is_dir():
        raise ValueError(f"Masks directory not found: {masks_dir}")
    out = ensure_new_path(out_dir)
    out.mkdir(parents=True, exist_ok=False)

    with TemporaryDirectory() as tmp:
        base_dir, xml_path = extract_dataset_if_zip(dataset_path, Path(tmp))
        root = safe_parse_xml(xml_path)
        nodes = sorted(root.findall("image"), key=lambda n: int(n.attrib["id"]))

        comparison_sheets: list[str] = []
        comparison_metrics: list[dict] = []

        if compare_levels:
            levels = [float(x) for x in compare_levels]
            target_indices = compare_frames or [
                idx
                for idx in (3, len(nodes) // 2, max(0, len(nodes) - 10))
                if idx < len(nodes)
            ]
            font = load_font(20)
            for idx in target_indices:
                if idx < 0 or idx >= len(nodes):
                    raise ValueError(f"Frame index out of range: {idx}")
                name = nodes[idx].attrib["name"]
                mask_file = masks_dir / name
                if not mask_file.is_file():
                    raise ValueError(f"Mask file missing for frame {idx}: {mask_file}")
                original = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 0
                frame = cv2.imread(str(resolve_image_path(base_dir, name)))
                if not np.any(original):
                    continue
                ys, xs = np.where(original)
                contours, hierarchy = cv2.findContours(
                    original.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
                )
                x0 = max(0, int(xs.min()) - 18)
                x1 = min(frame.shape[1], int(xs.max()) + 19)
                y0 = max(0, int(ys.min()) - 22)
                y1 = min(frame.shape[0], int(ys.max()) + 23)
                crop = frame[y0:y1, x0:x1]
                scale = 600.0 / max(1, crop.shape[1])
                height_px = max(40, round(crop.shape[0] * scale))
                cell_h = height_px + 65
                cols = min(3, max(1, len(levels)))
                rows = int(np.ceil(len(levels) / cols))
                sheet = Image.new("RGB", (cols * 620, rows * cell_h + 45), (24, 24, 24))
                draw = ImageDraw.Draw(sheet)
                draw.text(
                    (10, 8),
                    f"Frame {idx:02d} | Suavizacao do contorno (subpixel)",
                    font=font,
                    fill="white",
                )
                for j, lvl in enumerate(levels):
                    paths = [smooth_contour(c, lvl, step=step) for c in contours]
                    mask = np.zeros(original.shape, np.uint8)
                    if paths:
                        cv2.fillPoly(
                            mask,
                            [np.rint(p * 256).astype(np.int32) for p in paths],
                            1,
                            shift=8,
                        )
                    if lvl == 0:
                        mask = original.astype(np.uint8)
                    area = int(mask.sum())
                    orig_area = max(1, int(original.sum()))
                    change = 100.0 * (area / orig_area - 1.0)
                    view = cv2.resize(
                        crop, (600, height_px), interpolation=cv2.INTER_CUBIC
                    )
                    tint = (
                        cv2.resize(
                            mask[y0:y1, x0:x1],
                            (600, height_px),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        > 0
                    )
                    view[tint] = (
                        view[tint] * 0.88 + np.array([0, 230, 255]) * 0.12
                    ).astype(np.uint8)
                    for path in paths:
                        shifted = (path - np.array([x0, y0])) * scale
                        cv2.polylines(
                            view,
                            [np.rint(shifted * 256).astype(np.int32)],
                            True,
                            (0, 230, 255),
                            1,
                            cv2.LINE_AA,
                            shift=8,
                        )
                    x = (j % cols) * 620 + 10
                    y = (j // cols) * cell_h + 45
                    title = "Original" if lvl == 0 else f"Suavizacao sigma {lvl:g} px"
                    draw.text((x, y), title, font=font, fill="white")
                    draw.text(
                        (x, y + 25),
                        f"Variacao de area: {change:+.2f}%",
                        font=font,
                        fill="#bbbbbb",
                    )
                    sheet.paste(
                        Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)),
                        (x, y + 55),
                    )
                    comparison_metrics.append(
                        {
                            "frame": idx,
                            "name": name,
                            "sigma_px": lvl,
                            "area_px": area,
                            "area_change_percent": round(change, 4),
                        }
                    )
                sheet_path = out / f"comparacao_frame_{idx:02d}.png"
                sheet.save(sheet_path)
                comparison_sheets.append(str(sheet_path))

        smoothed_masks_dir = out / "mascaras"
        overlays_dir = out / "sobreposicoes"
        smoothed_masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        frame_records: list[dict] = []
        for idx, node in enumerate(nodes):
            name = node.attrib["name"]
            mask_file = masks_dir / name
            if not mask_file.is_file():
                raise ValueError(
                    f"Mask file missing for frame {idx} ({name}): {mask_file}"
                )
            original = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 0
            frame = cv2.imread(str(resolve_image_path(base_dir, name)))
            if sigma > 0 and np.any(original):
                contours, _ = cv2.findContours(
                    original.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
                )
                paths = [smooth_contour(c, sigma, step=step) for c in contours]
                mask = np.zeros(original.shape, np.uint8)
                if paths:
                    cv2.fillPoly(
                        mask,
                        [np.rint(p * 256).astype(np.int32) for p in paths],
                        1,
                        shift=8,
                    )
            else:
                mask = original.astype(np.uint8)
                paths = []
            cv2.imwrite(str(smoothed_masks_dir / name), mask * 255)
            overlay = frame.copy()
            overlay[mask > 0] = (
                overlay[mask > 0] * 0.72 + np.array([40, 230, 255]) * 0.28
            ).astype(np.uint8)
            ext_contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, ext_contours, -1, (0, 230, 255), 1)
            cv2.imwrite(str(overlays_dir / name), overlay)
            orig_sum = int(original.sum())
            new_sum = int(mask.sum())
            frame_records.append(
                {
                    "frame": idx,
                    "name": name,
                    "original_area_px": orig_sum,
                    "smoothed_area_px": new_sum,
                    "area_change_percent": round(100.0 * (new_sum / orig_sum - 1.0), 4)
                    if orig_sum > 0
                    else 0.0,
                }
            )

        summary = {
            "output_dir": str(out),
            "masks_dir": str(smoothed_masks_dir),
            "overlays_dir": str(overlays_dir),
            "sigma_px": sigma,
            "step_px": step,
            "comparison_sheets": comparison_sheets,
            "comparison_metrics": comparison_metrics,
            "frames_processed": len(frame_records),
        }
        (out / "parametros.json").write_text(
            json.dumps({**summary, "frames": frame_records}, indent=2),
            encoding="utf-8",
        )
        return summary


def run_thresholds(
    dataset_path: Path,
    logits_path: Path,
    out_dir: Path,
    levels: list[float] | None = None,
    sample_frames: list[int] | None = None,
) -> dict:
    logits_path = logits_path.expanduser().resolve()
    out = ensure_new_path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    thresholds = [float(x) for x in (levels or [0.0, 2.0, 4.0, 6.0])]

    if logits_path.is_file() and logits_path.suffix == ".npz":
        scores_stack = np.load(logits_path)["scores"].astype(np.float32)
    elif logits_path.is_dir():
        npy_files = sorted(logits_path.glob("*.npy"))
        if not npy_files:
            raise ValueError(f"No .npy files found in {logits_path}")
        scores_stack = np.stack(
            [np.load(f).astype(np.float32) for f in npy_files], axis=0
        )
    else:
        raise ValueError(
            f"logits_path must be a .npz file or directory of .npy files: {logits_path}"
        )

    with TemporaryDirectory() as tmp:
        base_dir, xml_path = extract_dataset_if_zip(dataset_path, Path(tmp))
        root = safe_parse_xml(xml_path)
        nodes = sorted(root.findall("image"), key=lambda n: int(n.attrib["id"]))
        if len(nodes) != len(scores_stack):
            raise ValueError(
                f"Frame count mismatch: {len(nodes)} in XML vs {len(scores_stack)} in logits"
            )
        frames = [
            cv2.imread(str(resolve_image_path(base_dir, n.attrib["name"])))
            for n in nodes
        ]
        records: list[dict] = []

        for th in thresholds:
            th_dir = out / f"limiar_{th:g}" / "mascaras"
            th_dir.mkdir(parents=True, exist_ok=True)
            for idx, node in enumerate(nodes):
                mask = (scores_stack[idx] > th).astype(np.uint8)
                cv2.imwrite(str(th_dir / node.attrib["name"]), mask * 255)
                records.append(
                    {
                        "threshold": th,
                        "frame": idx,
                        "name": node.attrib["name"],
                        "area_px": int(mask.sum()),
                    }
                )

        chosen_frames = sample_frames or [
            i for i in (0, 5, len(nodes) // 2, max(0, len(nodes) - 1)) if i < len(nodes)
        ]
        chosen_frames = sorted(set(chosen_frames))
        h, w = frames[0].shape[:2]
        cell_w = min(440, w)
        cell_h = min(220, h)
        sheet = Image.new(
            "RGB",
            (len(chosen_frames) * cell_w, len(thresholds) * (cell_h + 24)),
            (24, 24, 24),
        )
        draw = ImageDraw.Draw(sheet)
        for row_idx, th in enumerate(thresholds):
            for col_idx, f_idx in enumerate(chosen_frames):
                mask = (scores_stack[f_idx] > th).astype(np.uint8)
                overlay = frames[f_idx].copy()
                overlay[mask > 0] = (
                    overlay[mask > 0] * 0.72 + np.array([40, 230, 255]) * 0.28
                ).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, (0, 230, 255), 1)
                resized = cv2.resize(
                    overlay, (cell_w, cell_h), interpolation=cv2.INTER_AREA
                )
                x = col_idx * cell_w
                y = row_idx * (cell_h + 24)
                sheet.paste(
                    Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)),
                    (x, y + 22),
                )
                draw.text(
                    (x + 6, y + 4),
                    f"limiar {th:g} | f{f_idx:02d} ({int(mask.sum())} px)",
                    fill="white",
                )
        sheet_path = out / "limiares.jpg"
        sheet.save(sheet_path, quality=92)

        summary = {
            "output_dir": str(out),
            "sheet": str(sheet_path),
            "thresholds": thresholds,
            "sample_frames": chosen_frames,
        }
        (out / "limiares.json").write_text(
            json.dumps({**summary, "records": records}, indent=2),
            encoding="utf-8",
        )
        return summary


def export_proposals(
    dataset_path: Path,
    masks_dir: Path,
    out_dir: Path,
    label: str = "smoke",
    sigma: float = 0.8,
    min_area: float = 1.0,
    step: float = 0.4,
) -> dict:
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    masks_dir = masks_dir.expanduser().resolve()
    if not masks_dir.is_dir():
        raise ValueError(f"Masks directory not found: {masks_dir}")
    out = ensure_new_path(out_dir)
    out.mkdir(parents=True, exist_ok=False)

    with TemporaryDirectory() as tmp:
        _, xml_path = extract_dataset_if_zip(dataset_path, Path(tmp))
        root = safe_parse_xml(xml_path)
        nodes = sorted(root.findall("image"), key=lambda n: int(n.attrib["id"]))
        if not nodes:
            raise ValueError(f"No <image> elements found in {xml_path}")

        export_xml = ET.Element("annotations")
        ET.SubElement(export_xml, "version").text = "1.1"
        meta_node = root.find("meta")
        if meta_node is not None:
            export_xml.append(ET.fromstring(ET.tostring(meta_node)))

        preserved_frames: list[int] = []
        filled_frames: list[int] = []
        empty_frames: list[int] = []
        holes_skipped: list[dict] = []
        json_shapes: list[dict] = []

        for idx, node in enumerate(nodes):
            h = int(node.attrib["height"])
            w = int(node.attrib["width"])
            name = node.attrib["name"]
            im_elem = ET.SubElement(export_xml, "image", dict(node.attrib))

            if has_any_annotation(node, label=None):
                preserved_frames.append(idx)
                for child in list(node):
                    im_elem.append(ET.fromstring(ET.tostring(child)))
                continue

            mask_path = masks_dir / name
            if not mask_path.is_file():
                raise ValueError(
                    f"Predicted mask missing for frame {idx} ({name}): {mask_path}"
                )
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (h, w):
                raise ValueError(
                    f"Invalid mask dimensions for {name}: expected {(h, w)}, got {None if mask is None else mask.shape}"
                )

            contours, hierarchy = cv2.findContours(
                (mask > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
            )
            before_count = len(json_shapes)
            if contours and hierarchy is not None:
                for j, contour in enumerate(contours):
                    if hierarchy[0, j, 3] >= 0:
                        holes_skipped.append(
                            {
                                "frame": idx,
                                "name": name,
                                "area_px": float(cv2.contourArea(contour)),
                            }
                        )
                        continue
                    area = float(cv2.contourArea(contour))
                    if len(contour) < 3 or area < min_area:
                        continue
                    pts = (
                        smooth_contour(contour, sigma=sigma, step=step)
                        if sigma > 0
                        else contour.reshape(-1, 2).astype(np.float64)
                    )
                    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
                    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
                    if len(pts) < 3 or not np.isfinite(pts).all():
                        continue
                    # Ensure polygon has at least 3 distinct points and non-zero area
                    poly_area = float(cv2.contourArea(pts.astype(np.float32)))
                    if (
                        poly_area <= 0.0
                        or len(
                            {(round(float(x), 3), round(float(y), 3)) for x, y in pts}
                        )
                        < 3
                    ):
                        continue
                    flat_points = np.round(pts, 3).reshape(-1).tolist()
                    json_shapes.append(
                        {
                            "label": label,
                            "frame": idx,
                            "type": "polygon",
                            "points": flat_points,
                            "occluded": False,
                            "outside": False,
                            "z_order": 0,
                            "rotation": 0.0,
                            "attributes": {},
                        }
                    )
                    points_text = ";".join(f"{x:.3f},{y:.3f}" for x, y in pts)
                    ET.SubElement(
                        im_elem,
                        "polygon",
                        label=label,
                        source="auto",
                        occluded="0",
                        z_order="0",
                        points=points_text,
                    )

            if len(json_shapes) > before_count:
                filled_frames.append(idx)
            else:
                empty_frames.append(idx)

        ET.indent(export_xml)
        xml_out_path = out / "annotations.xml"
        ET.ElementTree(export_xml).write(
            xml_out_path, encoding="utf-8", xml_declaration=True
        )

        zip_out_path = out / "predictions_cvat_1.1.zip"
        with ZipFile(zip_out_path, "w", ZIP_DEFLATED) as zf:
            zf.write(xml_out_path, "annotations.xml")
        with ZipFile(zip_out_path) as zf:
            if zf.testzip() is not None:
                raise RuntimeError(f"Corrupted ZIP generated at {zip_out_path}")

        proposals_json_path = out / "proposals.json"
        proposals_payload = {"shapes": json_shapes, "tags": []}
        proposals_json_path.write_text(
            json.dumps(proposals_payload, indent=2), encoding="utf-8"
        )

        summary = {
            "output_dir": str(out),
            "proposals_json": str(proposals_json_path),
            "annotations_xml": str(xml_out_path),
            "predictions_zip": str(zip_out_path),
            "label": label,
            "sigma_px": sigma,
            "shapes_created": len(json_shapes),
            "preserved_frames": preserved_frames,
            "filled_frames": filled_frames,
            "empty_frames": empty_frames,
            "holes_skipped_count": len(holes_skipped),
            "holes_area_total_px": round(sum(h["area_px"] for h in holes_skipped), 2),
        }
        (out / "export_summary.json").write_text(
            json.dumps({**summary, "holes_skipped": holes_skipped}, indent=2),
            encoding="utf-8",
        )
        return summary


def draw_mask_overlay(
    frame: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]
) -> np.ndarray:
    out = frame.copy()
    binary = mask > 0
    if np.any(binary):
        out[binary] = (
            out[binary] * 0.72 + np.asarray(color_bgr, dtype=np.float32) * 0.28
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color_bgr, 1)
    return out


def render_video(
    dataset_path: Path,
    masks_dirs: list[Path],
    out_dir: Path,
    titles: list[str] | None = None,
    fps: int = 4,
    crop_y: tuple[int, int] | None = None,
    gif: bool = False,
) -> dict:
    if not masks_dirs:
        raise ValueError("Provide at least one masks directory.")
    resolved_dirs = [d.expanduser().resolve() for d in masks_dirs]
    for d in resolved_dirs:
        if not d.is_dir():
            raise ValueError(f"Masks directory not found: {d}")
    out = ensure_new_path(out_dir)
    out.mkdir(parents=True, exist_ok=False)

    with TemporaryDirectory() as tmp:
        base_dir, xml_path = extract_dataset_if_zip(dataset_path, Path(tmp))
        root = safe_parse_xml(xml_path)
        nodes = sorted(root.findall("image"), key=lambda n: int(n.attrib["id"]))
        if not nodes:
            raise ValueError(f"No frames in {xml_path}")

        manual_frames = {
            idx for idx, n in enumerate(nodes) if has_any_annotation(n, label=None)
        }
        first_frame = cv2.imread(
            str(resolve_image_path(base_dir, nodes[0].attrib["name"]))
        )
        h, w = first_frame.shape[:2]
        y0, y1 = crop_y if crop_y is not None else (0, h)
        if not (0 <= y0 < y1 <= h):
            raise ValueError(f"Invalid crop_y {(y0, y1)} for frame height {h}")

        panel_titles = ["Original"] + (
            titles
            if titles and len(titles) == len(resolved_dirs)
            else [d.name for d in resolved_dirs]
        )
        total_panels = len(panel_titles)
        cols = 2 if total_panels > 1 else 1
        rows = int(np.ceil(total_panels / cols))
        crop_h = y1 - y0
        header_h = 24
        cell_h = crop_h + header_h
        canvas_w = cols * w
        canvas_h = rows * cell_h
        # Ensure even dimensions for H.264 compatibility
        canvas_w += canvas_w % 2
        canvas_h += canvas_h % 2

        raw_mp4 = out / "demonstracao_bruto.mp4"
        final_mp4 = out / "demonstracao_h264.mp4"
        writer = cv2.VideoWriter(
            str(raw_mp4),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (canvas_w, canvas_h),
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to initialize OpenCV VideoWriter")

        pil_frames: list[Image.Image] = []
        inspect_dir = out / "inspecao"
        inspect_dir.mkdir(exist_ok=True)
        key_indices = {0, 1, 2, 5, 10, 20, 30, len(nodes) - 1}

        for idx, node in enumerate(nodes):
            name = node.attrib["name"]
            frame = cv2.imread(str(resolve_image_path(base_dir, name)))
            is_manual = idx in manual_frames
            color = (50, 235, 60) if is_manual else (0, 230, 255)
            views = [frame]
            for d in resolved_dirs:
                m_path = d / name
                if not m_path.is_file():
                    raise ValueError(f"Mask missing at {m_path}")
                mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE) > 0
                views.append(draw_mask_overlay(frame, mask, color))

            sheet = Image.new("RGB", (canvas_w, canvas_h), (24, 24, 24))
            draw = ImageDraw.Draw(sheet)
            for j, (view, title) in enumerate(zip(views, panel_titles)):
                col = j % cols
                row = j // cols
                x = col * w
                y = row * cell_h
                cropped = view[y0:y1, 0:w]
                sheet.paste(
                    Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)),
                    (x, y + header_h),
                )
                suffix = (
                    " | manual"
                    if (j > 0 and is_manual)
                    else (" | auto" if j > 0 else "")
                )
                draw.text(
                    (x + 6, y + 4), f"{title} | frame {idx:02d}{suffix}", fill="white"
                )

            writer.write(cv2.cvtColor(np.asarray(sheet), cv2.COLOR_RGB2BGR))
            if gif:
                pil_frames.append(sheet.copy())
            if idx in key_indices:
                sheet.save(inspect_dir / f"frame_{idx:02d}.jpg", quality=92)

        writer.release()

        if shutil.which("ffmpeg"):
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(raw_mp4),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(final_mp4),
                ],
                check=True,
            )
            raw_mp4.unlink(missing_ok=True)
        else:
            raw_mp4.rename(final_mp4)

        gif_path = None
        if gif and pil_frames:
            gif_file = out / "demonstracao.gif"
            pil_frames[0].save(
                gif_file,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(round(1000.0 / max(1, fps))),
                loop=0,
            )
            gif_path = str(gif_file)

        summary = {
            "output_dir": str(out),
            "video_mp4": str(final_mp4),
            "video_gif": gif_path,
            "inspect_dir": str(inspect_dir),
            "frames": len(nodes),
            "fps": fps,
            "panels": panel_titles,
        }
        (out / "video_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
