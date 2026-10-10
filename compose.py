import glob
import json
import os
import sys

import cv2

SRC = sys.argv[1] if len(sys.argv) > 1 else r"D:\VsPro\CPP\v58-onnxrt-v2\output\dji9_clip_0150_0230"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"D:\VsPro\CPP\v58-onnxrt-v2\output\dji9_clip_0150_0230_composed.mp4"
FPS = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0
LABELS = r"D:\VsPro\CPP\Zmodel\dronev5.txt"   # 类别名表（与 C++ 项目同一份，label->id 决定颜色）


def draw_pred(img, label, x1, y1, x2, y2, cls_id):
    """与 C++ YolovOnnx::DrawPred 相同的画法，字号缩小适配 720p/D1 小图"""
    scale, thickness = 0.35, 1
    idx = cls_id + 3
    color = (37 * idx * idx % 255, 17 * idx * idx % 255, 29 * idx * idx % 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, scale, thickness)
    cv2.rectangle(img, (x1, y1 - label_size[1] - 2), (x1 + label_size[0], y1 - 2), color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, scale,
                (225, 255, 255), thickness, cv2.LINE_AA)


def load_labels(path):
    names = {}
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                name = line.strip()
                if name:
                    names[name] = i
    except OSError as e:
        print(f"[warn] cannot read labels {path}: {e}")
    return names


def shape_rect(points):
    """rectangle 点集 -> (x1, y1, x2, y2)；非数值输入抛 ValueError 由调用方跳过"""
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def main():
    label2id = load_labels(LABELS)
    jpgs = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
    if not jpgs:
        raise SystemExit("no jpgs in " + SRC)
    first = cv2.imread(jpgs[0])
    if first is None:
        raise SystemExit("first frame unreadable: " + jpgs[0])
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter.fourcc(*"mp4v"), FPS, (w, h))
    if not vw.isOpened():
        raise SystemExit("VideoWriter open failed")

    n_box = 0
    for jp in jpgs:
        img = cv2.imread(jp)
        if img is None:
            print(f"[warn] skip unreadable {jp}")
            continue
        jpath = os.path.join(SRC, os.path.basename(jp)[:-4] + ".json")
        if os.path.exists(jpath):
            try:
                with open(jpath, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[warn] skip bad json {jpath}: {e}")
                data = {"shapes": []}
            for s in data.get("shapes", []):
                if s.get("shape_type") != "rectangle":
                    continue
                try:
                    x1, y1, x2, y2 = shape_rect(s["points"])
                except (ValueError, TypeError, KeyError, IndexError) as e:
                    print(f"[warn] skip bad shape in {jpath}: {e}")
                    continue
                label = str(s.get("label", ""))
                draw_pred(img, label, x1, y1, x2, y2, label2id.get(label, 0))
                n_box += 1
        vw.write(img)
    vw.release()
    print(f"frames={len(jpgs)} boxes={n_box} -> {OUT}")


if __name__ == "__main__":
    main()
