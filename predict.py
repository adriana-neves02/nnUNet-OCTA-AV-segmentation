"""Segment arteries and veins in OCTA en-face images with the trained nnU-Net.

Usage:
    python predict.py -i <image or folder> -o <output folder> [options]
    python predict.py -i <folder> -o <output folder> --recursive --pattern "*_SVP_*.png"

With --recursive, subfolders are searched too, and <name> keeps the subfolder path, so the
output folders mirror the input tree. For each input image <name>, it writes:
    <output>/artery/<name>.png     artery mask (0 / 255)
    <output>/vein/<name>.png       vein mask (0 / 255)
    <output>/av/<name>.png         colour A/V map: red artery, blue vein, magenta both (crossings)
    <output>/overlay/<name>.png    the OCTA image with arteries (red) and veins (blue) on top
    <output>/panel/<name>.png      input | (enhanced input) | overlay | A/V map, side by side
    <output>/probabilities/<name>.npz   artery and vein probabilities (only with --save-probabilities)
    <output>/predictions.csv       one row per image: size, vessel fractions, uncertain fraction

The model is the pooled nnU-Net (ResEnc L, 2D) trained on OCTA-500 3 mm, OCTA-500 6 mm
and XJTU. By default it averages its 4 folds (an ensemble) and the mirrored images (test
time augmentation), as nnU-Net does.

Arteries and veins are two separate outputs (a pixel at a crossing can be both), each
thresholded at --threshold. nnU-Net normalizes every image itself (z-score), so the input
does not need any preprocessing. See the README for when --enhance or --scale may help.
"""
from pathlib import Path
import argparse
import csv
import os
import sys

# Before importing torch: let the Apple GPU fall back to the CPU for unsupported operations
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
from PIL import Image
from skimage import exposure, transform

REPO = Path(__file__).resolve().parent
DEFAULT_MODEL = REPO / 'weights' / 'Dataset501_OCTA_all' / 'nnUNetTrainer__nnUNetResEncUNetLPlans__2d'
EXTENSIONS = {'.png', '.bmp', '.tif', '.tiff', '.jpg', '.jpeg'}
ENHANCEMENTS = ('none', 'stretch', 'gamma', 'clahe')

ARTERY_COLOUR = np.array([255, 0, 0], dtype=np.float32)
VEIN_COLOUR = np.array([0, 80, 255], dtype=np.float32)
BOTH_COLOUR = np.array([255, 0, 255], dtype=np.float32)


# ── Images ────────────────────────────────────────────────────────────────────
def list_images(path, recursive=False, pattern='*'):
    """Images of a folder (or one image). With recursive, all subfolders are searched too.
    pattern selects file names, e.g. '*_SVP_*.png' (default: every image)."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        sys.exit(f'Not found: {path}')
    candidates = path.rglob(pattern) if recursive else path.glob(pattern)
    files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        sys.exit(f'No images matching "{pattern}" in {path}' + (' or its subfolders' if recursive else ''))
    return files


def output_name(fn, root):
    """Name of the outputs of an image: its path below the input folder, without extension,
    so that images with the same file name in different subfolders do not overwrite each other."""
    root = Path(root)
    if root.is_file():
        return fn.stem
    return fn.relative_to(root).with_suffix('').as_posix()


def load_gray(fn):
    """Grayscale image as float32, on its original intensity scale.
    RGB images are converted as in training (PIL 'L')."""
    img = np.array(Image.open(fn))
    if img.ndim == 3:
        img = np.array(Image.fromarray(img[..., :3]).convert('L'))
    return img.astype(np.float32)


def to_unit(img):
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


def enhance(img, method='none', gamma=1.0, clahe_clip=0.01):
    """Optional contrast enhancement, returned on a 0-255 scale.
    'none' returns the image unchanged: the model was trained on unenhanced images."""
    if method == 'none':
        return img
    u = to_unit(img)
    if method == 'stretch':  # linear stretch between the 1st and 99th percentiles
        lo, hi = np.percentile(u, (1, 99))
        u = np.clip((u - lo) / max(hi - lo, 1e-6), 0, 1)
    elif method == 'gamma':  # < 1 brightens dark vessels, > 1 darkens the background
        u = u ** gamma
    elif method == 'clahe':  # local (adaptive) histogram equalization
        u = exposure.equalize_adapthist(u, clip_limit=clahe_clip)
    else:
        raise ValueError(f'Unknown enhancement {method}')
    return (u * 255).astype(np.float32)


def rescale(img, scale, order=1):
    if scale == 1:
        return img
    shape = tuple(int(round(s * scale)) for s in img.shape[:2])
    return transform.resize(img, shape, order=order, preserve_range=True, anti_aliasing=scale < 1).astype(np.float32)


# ── Model ─────────────────────────────────────────────────────────────────────
def pick_device(name='auto'):
    import torch
    if name == 'auto':
        if torch.cuda.is_available():
            name = 'cuda'
        elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            name = 'mps'
        else:
            name = 'cpu'
    if name == 'cpu':
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    return torch.device(name)


def load_model(model_dir=DEFAULT_MODEL, folds=(0, 1, 2, 3), device='auto', tta=True):
    model_dir = Path(model_dir)
    if not (model_dir / 'plans.json').exists():
        sys.exit(f'No model in {model_dir}. Download the weights first: python download_weights.py')
    # nnU-Net warns when its data folders are not set; inference does not use them
    for var in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        os.environ.setdefault(var, str(model_dir.parent.parent))
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = pick_device(device)
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=tta,
        perform_everything_on_device=device.type == 'cuda', device=device,
        verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=tuple(folds),
                                                   checkpoint_name='checkpoint_final.pth')
    print(f'Model: {model_dir.name}, folds {list(folds)}, device {device}, mirroring {tta}')
    return predictor


def predict_probabilities(predictor, img):
    """Artery and vein probabilities (H, W) for a grayscale image (H, W)."""
    # Same layout as nnU-Net's reader for 2D PNGs: (channel, 1, H, W), spacing (999, 1, 1)
    x = img.astype(np.float32)[None, None]
    _, probabilities = predictor.predict_single_npy_array(x, {'spacing': (999, 1, 1)}, None, None, True)
    # Region-based model: one sigmoid output per region, shape (2, 1, H, W)
    return probabilities[0, 0].astype(np.float32), probabilities[1, 0].astype(np.float32)


def segment(predictor, img, method='none', gamma=1.0, clahe_clip=0.01, scale=1.0):
    """Probabilities at the size of the input image, and the image given to the model."""
    model_input = rescale(enhance(img, method, gamma, clahe_clip), scale)
    artery, vein = predict_probabilities(predictor, model_input)
    if scale != 1:
        artery = transform.resize(artery, img.shape, order=1, preserve_range=True)
        vein = transform.resize(vein, img.shape, order=1, preserve_range=True)
    return artery, vein, model_input


def uncertain_fraction(artery, vein, threshold=0.5, margin=0.2):
    """Share of the predicted vessel pixels whose probability is close to the threshold.
    No ground truth needed: lower means the model is more decided."""
    p = np.maximum(artery, vein)
    vessel = p > threshold
    if not vessel.any():
        return float('nan')
    near = (np.abs(artery - threshold) < margin) | (np.abs(vein - threshold) < margin)
    return float((near & vessel).mean() / vessel.mean())


# ── Visualization ─────────────────────────────────────────────────────────────
def av_map(artery_mask, vein_mask):
    rgb = np.zeros(artery_mask.shape + (3,), dtype=np.uint8)
    rgb[artery_mask] = ARTERY_COLOUR
    rgb[vein_mask] = VEIN_COLOUR
    rgb[artery_mask & vein_mask] = BOTH_COLOUR
    return rgb


def overlay(img, artery_mask, vein_mask, alpha=0.6):
    gray = (to_unit(img) * 255).astype(np.float32)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    for mask, colour in ((artery_mask & ~vein_mask, ARTERY_COLOUR), (vein_mask & ~artery_mask, VEIN_COLOUR),
                         (artery_mask & vein_mask, BOTH_COLOUR)):
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * colour
    return rgb.astype(np.uint8)


def gray_rgb(img):
    g = (to_unit(img) * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def panel(images, gap=8):
    h = max(i.shape[0] for i in images)
    padded = [np.pad(i, ((0, h - i.shape[0]), (0, gap), (0, 0)), constant_values=255) for i in images]
    return np.concatenate(padded, axis=1)[:, :-gap]


def save_png(array, fn):
    fn.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(fn)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', required=True, help='An image, or a folder of images')
    parser.add_argument('-r', '--recursive', action='store_true', help='Also search the subfolders of --input')
    parser.add_argument('-p', '--pattern', default='*',
                        help='Only images whose file name matches, e.g. "*_SVP_*.png" (default: all)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip images whose masks are already in the output folder (to resume a run)')
    parser.add_argument('-o', '--output', required=True, help='Output folder (created if needed)')
    parser.add_argument('-m', '--model', default=str(DEFAULT_MODEL), help='Model folder (with plans.json and fold_<k>/)')
    parser.add_argument('-f', '--folds', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='Folds to average (default all 4; one fold is about 4x faster)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--no-tta', action='store_true', help='No mirroring at test time (faster, slightly worse)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold of the masks')
    parser.add_argument('--enhance', default='none', choices=ENHANCEMENTS,
                        help='Contrast enhancement before the model (default none, as in training)')
    parser.add_argument('--gamma', type=float, default=0.8, help='Gamma of --enhance gamma')
    parser.add_argument('--clahe-clip', type=float, default=0.01, help='Clip limit of --enhance clahe')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Resize factor before the model, to match the pixel size of training (see README); '
                             'outputs are returned at the original size')
    parser.add_argument('--save-probabilities', action='store_true', help='Also save the probabilities (.npz)')
    args = parser.parse_args()

    files = list_images(args.input, args.recursive, args.pattern)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(fn, output_name(fn, args.input)) for fn in files]
    if args.skip_existing:
        jobs = [(fn, name) for fn, name in jobs
                if not ((out / 'artery' / f'{name}.png').exists() and (out / 'vein' / f'{name}.png').exists())]
    print(f'{len(files)} images found, {len(jobs)} to segment')
    if not jobs:
        return
    predictor = load_model(args.model, args.folds, args.device, tta=not args.no_tta)

    rows, failed = [], []
    for k, (fn, name) in enumerate(jobs, 1):
        try:
            img = load_gray(fn)
            artery, vein, model_input = segment(predictor, img, args.enhance, args.gamma, args.clahe_clip, args.scale)
        except Exception as e:  # one unreadable image should not stop the run
            failed.append(fn)
            print(f'[{k}/{len(jobs)}] FAILED {name}: {e}')
            continue
        a, v = artery > args.threshold, vein > args.threshold

        save_png((a * 255).astype(np.uint8), out / 'artery' / f'{name}.png')
        save_png((v * 255).astype(np.uint8), out / 'vein' / f'{name}.png')
        save_png(av_map(a, v), out / 'av' / f'{name}.png')
        ov = overlay(img, a, v)
        save_png(ov, out / 'overlay' / f'{name}.png')
        parts = [gray_rgb(img)]
        if args.enhance != 'none':
            parts.append(gray_rgb(rescale(model_input, 1 / args.scale) if args.scale != 1 else model_input))
        save_png(panel(parts + [ov, av_map(a, v)]), out / 'panel' / f'{name}.png')
        if args.save_probabilities:
            fn_prob = out / 'probabilities' / f'{name}.npz'
            fn_prob.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fn_prob, artery=artery.astype(np.float16), vein=vein.astype(np.float16))

        rows.append(dict(image=name, height=img.shape[0], width=img.shape[1], enhance=args.enhance,
                         scale=args.scale, artery_fraction=round(float(a.mean()), 5),
                         vein_fraction=round(float(v.mean()), 5),
                         uncertain_fraction=round(uncertain_fraction(artery, vein, args.threshold), 4)))
        print(f'[{k}/{len(jobs)}] {name}: artery {a.mean():.1%}, vein {v.mean():.1%} of the image')

    if rows:
        summary = out / 'predictions.csv'
        append = args.skip_existing and summary.exists()  # resumed run: keep the earlier rows
        with open(summary, 'a' if append else 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            if not append:
                writer.writeheader()
            writer.writerows(rows)
    print(f'Done: {len(rows)} segmented, {len(failed)} failed -> {out}')
    for fn in failed:
        print(f'  failed: {fn}')


if __name__ == '__main__':
    main()
