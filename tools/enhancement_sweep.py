"""Compare preprocessing options (contrast enhancement, rescaling) on your own images.

Usage:
    python tools/enhancement_sweep.py -i <folder of images> -o <output folder> [--labels <folder>]

Each image is segmented once per variant (see VARIANTS, plus --scales). It writes:
    <output>/sweep.csv           one row per image and variant
    <output>/sweep_summary.csv   mean per variant (the table printed at the end)
    <output>/sheets/<name>.png   one overlay per variant, side by side, for visual review

With --labels (a folder with artery/ and vein/ subfolders of binary masks, named like the
images), each variant gets Dice scores: that is the way to choose a preprocessing.
Without labels there are two indirect signals only:
    uncertain_fraction  share of vessel pixels with probability near the threshold (lower
                        is more decided, but a confident model can still be wrong)
    vessel_fraction     share of the image predicted as vessel (a sudden change between
                        variants means the preprocessing changes what is segmented)
and the sheets, to check by eye. A few annotated images are worth much more than these.
"""
from pathlib import Path
import argparse
import csv
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # predict.py is in the repository root
from predict import (DEFAULT_MODEL, list_images, load_gray, load_model, overlay, panel, save_png, segment,
                     uncertain_fraction)

# name: (enhancement, gamma, clahe clip)
VARIANTS = {
    'none': ('none', 1.0, 0.01),
    'stretch': ('stretch', 1.0, 0.01),
    'gamma0.7': ('gamma', 0.7, 0.01),
    'gamma1.4': ('gamma', 1.4, 0.01),
    'clahe0.01': ('clahe', 1.0, 0.01),
    'clahe0.03': ('clahe', 1.0, 0.03),
}


def read_mask(fn):
    m = np.array(Image.open(fn))
    if m.ndim == 3:
        m = m[..., :3].max(axis=2)
    return m > (m.max() / 2) if m.max() > 0 else m > 0


def find_mask(folder, stem):
    for fn in folder.glob(f'{stem}.*'):
        return read_mask(fn)
    return None


def dice(pred, gt):
    s = pred.sum() + gt.sum()
    return float(2 * (pred & gt).sum() / s) if s else 1.0


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).rectangle([0, 0, 8 + 7 * len(text), 16], fill=(0, 0, 0))
    ImageDraw.Draw(im).text((4, 2), text, fill=(255, 255, 255))
    return np.array(im)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', required=True, help='Folder of images (or one image)')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--labels', help='Folder with artery/ and vein/ masks, to compute Dice')
    parser.add_argument('--scales', type=float, nargs='*', default=[],
                        help='Also try these resize factors, without enhancement (e.g. 0.75 1.33)')
    parser.add_argument('-m', '--model', default=str(DEFAULT_MODEL))
    parser.add_argument('-f', '--folds', type=int, nargs='+', default=[0],
                        help='Folds to use (default fold 0 only, to keep the sweep fast)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    variants = {name: (*v, 1.0) for name, v in VARIANTS.items()}
    variants.update({f'scale{s:g}': ('none', 1.0, 0.01, s) for s in args.scales})
    labels = Path(args.labels) if args.labels else None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    predictor = load_model(args.model, args.folds, args.device, tta=False)

    rows = []
    for fn in list_images(args.input):
        img = load_gray(fn)
        gt_a = find_mask(labels / 'artery', fn.stem) if labels else None
        gt_v = find_mask(labels / 'vein', fn.stem) if labels else None
        tiles = []
        for name, (method, gamma, clip, scale) in variants.items():
            artery, vein, _ = segment(predictor, img, method, gamma, clip, scale)
            a, v = artery > args.threshold, vein > args.threshold
            row = dict(image=fn.name, variant=name, vessel_fraction=round(float((a | v).mean()), 5),
                       uncertain_fraction=round(uncertain_fraction(artery, vein, args.threshold), 4))
            if gt_a is not None and gt_v is not None:
                row.update(dice_artery=round(dice(a, gt_a), 4), dice_vein=round(dice(v, gt_v), 4))
            rows.append(row)
            tiles.append(label(overlay(img, a, v), name))
        save_png(panel(tiles), out / 'sheets' / f'{fn.stem}.png')
        print(f'{fn.name}: {len(variants)} variants')

    fields = ['image', 'variant', 'vessel_fraction', 'uncertain_fraction']
    fields += [k for k in ('dice_artery', 'dice_vein') if any(k in r for r in rows)]
    with open(out / 'sweep.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics = [k for k in ('dice_artery', 'dice_vein', 'uncertain_fraction', 'vessel_fraction') if k in fields]
    summary = []
    for name in variants:
        sel = [r for r in rows if r['variant'] == name]
        summary.append({'variant': name, **{m: round(float(np.nanmean([r.get(m, np.nan) for r in sel])), 4)
                                            for m in metrics}})
    with open(out / 'sweep_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['variant'] + metrics)
        writer.writeheader()
        writer.writerows(summary)

    print('\n' + f'{"variant":<12}' + ''.join(f'{m:>20}' for m in metrics))
    for s in summary:
        print(f'{s["variant"]:<12}' + ''.join(f'{s[m]:>20}' for m in metrics))
    print(f'\nSheets: {out / "sheets"}')


if __name__ == '__main__':
    main()
