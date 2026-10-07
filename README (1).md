# VEK280 Network Setup – Ubuntu Host (eno2)

Commands to configure the Ubuntu host for direct connection to the VEK280 board over `eno2`.

## 1. Configure eno2

```bash
sudo ip link set eno2 up
sudo ip addr flush dev eno2
sudo ip addr add 192.168.1.52/24 dev eno2
```

## 2. Add the direct route

```bash
sudo ip route add 192.168.1.0/24 dev eno2 src 192.168.1.52
```

> If it reports `File exists`, that is fine — the route is already present.

## 3. Clear the old ARP entry

```bash
sudo ip neigh flush dev eno2
```

## 4. Test connectivity to the VEK280

```bash
ping -I eno2 -c 4 192.168.1.85
```

## 5. SSH into the board

```bash
ssh root@192.168.1.85
```

---

### Board-side reference (for completeness)

On the VEK280 the corresponding interface setup is:

```bash
ip link set end0 up
ip addr add 192.168.1.85/24 dev end0
ip route add 192.168.1.0/24 dev end0
```

In normal operation `end0` is already up and ping works; only the host-side steps above are required when recovering the link from the Ubuntu side.

---

# NPU YOLOv7 Demo – Full Procedure (board-side)

The commands work after a fresh start **only if the same files exist** on the new system.  
The scripts, the 100 images and the video live in `/root`. If you re-flash the SD card or use a new root filesystem, they are gone.

The NPU-only part needs no special setup — it comes from `npu_only=True` inside the scripts.

---

## Step 1: Check the system (every time)

```bash
source /etc/vai.sh
export SNAP=/run/media/mmcblk0p1/snapshot.VE2802_NPU_IP_O00_A304_M3.yolov7.TF
ls $SNAP | head -3
ls /dev/video0
ls /root/yolo100/*.jpg | wc -l
ls -la /root/yellow_traffic_signs.avi
df -h / /run/media/mmcblk0p1
python3 -c "import VART, cv2, numpy; print('python libs OK')"
```

If `source /etc/vai.sh` prints **"only 1 processor has been power up"**, run `reboot` and start again (as the warning advises).

You need all of these to exist:

* the snapshot folder
* `/dev/video0`
* 100 or more JPGs in `/root/yolo100`
* the `.avi` file
* free space on both filesystems

If the images or video are missing on a new system, copy them from your laptop:

```bash
scp -r yolo100 root@192.168.1.85:/root/
scp yellow_traffic_signs.avi root@192.168.1.85:/root/
```

---

## Step 2: Create the scripts (once per fresh system)

Paste each block separately — long pastes get garbled in the terminal.

### 2a. Helper modules

```bash
mkdir -p /root /run/media/mmcblk0p1/demo_results
cd /root
cat > yolo_decode.py <<'EOF'
import numpy as np

ANCHORS = [[(12,16),(19,36),(40,28)], [(36,75),(76,55),(72,146)], [(142,110),(192,243),(459,401)]]
STRIDES = [8, 16, 32]

def decode(npu_outs, conf=0.25, coeff=8.0):
    t = np.log(conf / (1 - conf))
    thr = int(np.ceil(t * coeff))
    boxes, scores, cls = [], [], []
    for l, o in enumerate(npu_outs):
        o = o[0]
        H, W, _ = o.shape
        o = o[..., :255].reshape(H, W, 3, 85)
        mask = o[..., 4] >= thr
        if not mask.any():
            continue
        ys, xs, a = np.nonzero(mask)
        p = 1 / (1 + np.exp(-(o[ys, xs, a].astype(np.float32) / coeff)))
        anc = np.array(ANCHORS[l], np.float32)[a]
        xy = (p[:, :2] * 2 - 0.5 + np.stack([xs, ys], 1)) * STRIDES[l]
        wh = (p[:, 2:4] * 2) ** 2 * anc
        c = p[:, 5:]
        ci = c.argmax(1)
        s = p[:, 4] * c[np.arange(len(ci)), ci]
        keep = s >= conf
        boxes.append(np.concatenate([xy - wh / 2, xy + wh / 2], 1)[keep])
        scores.append(s[keep]); cls.append(ci[keep])
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(cls)
EOF
cat > coco_names.py <<'EOF'
NAMES = ("person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,"
"fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,"
"bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,"
"sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,"
"bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,"
"carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,"
"tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,"
"refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush").split(",")
EOF
python3 -c "from coco_names import NAMES; print(len(NAMES), NAMES[0], NAMES[79])"
```

The last line should print: `80 person toothbrush`

### 2b. Image batch runner

```bash
cd /root
cat > run_native_save.py <<'EOF'
import os, sys, glob, time, numpy as np, cv2, VART
from yolo_decode import decode
from coco_names import NAMES

SNAP = "/run/media/mmcblk0p1/snapshot.VE2802_NPU_IP_O00_A304_M3.yolov7.TF"
folder = sys.argv[1]; N = int(sys.argv[2]); reduced = sys.argv[3] == "1"
OUT = sys.argv[4]; os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(folder + "/*.jpg"))[:N]
flag = cv2.IMREAD_REDUCED_COLOR_2 if reduced else cv2.IMREAD_COLOR

lut = np.clip(np.round(np.arange(256) / 255.0 * 128.0), 0, 127).astype(np.uint8)
r = VART.Runner(snapshot_dir=SNAP, npu_only=True)
assert not r.set_input_native()
buf = np.zeros((1, 640, 640, 4), np.int8); view = buf[..., :3]
canvas = np.full((640, 640, 3), 114, np.uint8)
T = dict(imread=0, letterbox=0, prep=0, npu=0, decode=0, nms_draw=0, save=0)
ndet = 0
r([view])  # warm-up

t_all = time.time()
for f in files:
    t = time.time(); img = cv2.imread(f, flag); T["imread"] += time.time() - t
    t = time.time()
    h, w = img.shape[:2]; sc = min(640 / h, 640 / w)
    nh, nw = int(round(h * sc)), int(round(w * sc)); dy, dx = (640 - nh) // 2, (640 - nw) // 2
    canvas[:] = 114; canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh))
    T["letterbox"] += time.time() - t
    t = time.time()
    view[0] = cv2.LUT(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), lut).view(np.int8)
    T["prep"] += time.time() - t
    t = time.time(); outs = r([view]); T["npu"] += time.time() - t
    t = time.time(); b, s, c = decode(outs, conf=0.25); T["decode"] += time.time() - t
    t = time.time()
    if len(s):
        xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), 0.25, 0.45)).reshape(-1)
        b, s, c = b[idx], s[idx], c[idx]
        b[:, [0, 2]] = (b[:, [0, 2]] - dx) / sc; b[:, [1, 3]] = (b[:, [1, 3]] - dy) / sc
        for (x1, y1, x2, y2), p, ci in zip(b, s, c):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, "%s %.2f" % (NAMES[int(ci)], p), (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    ndet += len(s)
    T["nms_draw"] += time.time() - t
    t = time.time()
    if not cv2.imwrite(os.path.join(OUT, os.path.basename(f)), img): print("SAVE FAILED:", f)
    T["save"] += time.time() - t

dt = time.time() - t_all; n = len(files)
print("images %d  reduced=%s  END-TO-END %.2f FPS (%.1f ms/frame)" % (n, reduced, n / dt, 1000 * dt / n))
for k, v in T.items(): print("%-9s %.1f ms" % (k, 1000 * v / n))
print("avg detections after NMS: %.1f" % (ndet / n))
print("saved to:", OUT)
EOF
```

### 2c. Video runner

```bash
cd /root
cat > run_video_native.py <<'EOF'
import sys, time, numpy as np, cv2, VART
from yolo_decode import decode
from coco_names import NAMES

SNAP = "/run/media/mmcblk0p1/snapshot.VE2802_NPU_IP_O00_A304_M3.yolov7.TF"
src = sys.argv[1]; out = sys.argv[2]
first = out.rsplit("/", 1)[0] + "/video_first_frame.jpg"
cap = cv2.VideoCapture(src)
if not cap.isOpened(): raise SystemExit("cannot open " + src)
vfps = cap.get(cv2.CAP_PROP_FPS) or 25
lut = np.clip(np.round(np.arange(256) / 255.0 * 128.0), 0, 127).astype(np.uint8)
r = VART.Runner(snapshot_dir=SNAP, npu_only=True); assert not r.set_input_native()
buf = np.zeros((1, 640, 640, 4), np.int8); view = buf[..., :3]
canvas = np.full((640, 640, 3), 114, np.uint8)
vw = None; n = 0; ndet = 0; t_npu = 0.0; t0 = time.time()
while True:
    ok, img = cap.read()
    if not ok: break
    h, w = img.shape[:2]; sc = min(640 / h, 640 / w)
    nh, nw = int(round(h * sc)), int(round(w * sc)); dy, dx = (640 - nh) // 2, (640 - nw) // 2
    canvas[:] = 114; canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh))
    view[0] = cv2.LUT(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), lut).view(np.int8)
    t = time.time(); outs = r([view]); t_npu += time.time() - t
    b, s, c = decode(outs, conf=0.25)
    if len(s):
        xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        i = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), 0.25, 0.45)).reshape(-1)
        b, s, c = b[i], s[i], c[i]
        b[:, [0, 2]] = (b[:, [0, 2]] - dx) / sc; b[:, [1, 3]] = (b[:, [1, 3]] - dy) / sc
        for (x1, y1, x2, y2), p, ci in zip(b, s, c):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, "%s %.2f" % (NAMES[int(ci)], p), (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    ndet += len(s)
    if vw is None: vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"MJPG"), vfps, (w, h))
    vw.write(img)
    if n == 0: cv2.imwrite(first, img)
    n += 1
dt = time.time() - t0
cap.release()
if vw: vw.release()
print("frames %d  END-TO-END %.2f FPS (%.1f ms/frame, includes video write)  npu %.1f ms  avg det %.1f" % (n, n / dt, 1000 * dt / n, 1000 * t_npu / n, ndet / n))
print("saved:", out)
EOF
```

### 2d. Webcam runner (with live HTTP stream)

```bash
cd /root
cat > run_cam_save.py <<'EOF'
import sys, time, threading, numpy as np, cv2, VART
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from yolo_decode import decode
from coco_names import NAMES

SNAP = "/run/media/mmcblk0p1/snapshot.VE2802_NPU_IP_O00_A304_M3.yolov7.TF"
dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
OUT = sys.argv[3] if len(sys.argv) > 3 else "/run/media/mmcblk0p1/demo_results/webcam_out.avi"

cond = threading.Condition(); latest = {"frame": None}

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with cond:
                    cond.wait()
                    f = latest["frame"]
                ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                self.wfile.write(jpg.tobytes()); self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
    def log_message(self, *a): pass

srv = ThreadingHTTPServer(("0.0.0.0", port), H); srv.daemon_threads = True
threading.Thread(target=srv.serve_forever, daemon=True).start()

cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened(): raise SystemExit("cannot open camera %d" % dev)

lut = np.clip(np.round(np.arange(256) / 255.0 * 128.0), 0, 127).astype(np.uint8)
r = VART.Runner(snapshot_dir=SNAP, npu_only=True); assert not r.set_input_native()
buf = np.zeros((1, 640, 640, 4), np.int8); view = buf[..., :3]
canvas = np.full((640, 640, 3), 114, np.uint8)
print("OPEN IN YOUR PC BROWSER:  http://<board-ip>:%d   (find the IP with: ip -4 addr)" % port, flush=True)

vw = None
n = 0; fps = 0.0; t_win = time.time()
try:
    while True:
        ok, img = cap.read()
        if not ok: print("read failed"); break
        h, w = img.shape[:2]; sc = min(640 / h, 640 / w)
        nh, nw = int(round(h * sc)), int(round(w * sc)); dy, dx = (640 - nh) // 2, (640 - nw) // 2
        canvas[:] = 114; canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh))
        view[0] = cv2.LUT(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), lut).view(np.int8)
        b, s, c = decode(r([view]), conf=0.25)
        if len(s):
            xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
            i = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), 0.25, 0.45)).reshape(-1)
            b, s, c = b[i], s[i], c[i]
            b[:, [0, 2]] = (b[:, [0, 2]] - dx) / sc; b[:, [1, 3]] = (b[:, [1, 3]] - dy) / sc
            for (x1, y1, x2, y2), p, ci in zip(b, s, c):
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, "%s %.2f" % (NAMES[int(ci)], p), (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, "%.1f FPS  %d det" % (fps, len(s)), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if vw is None: vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"MJPG"), 20, (img.shape[1], img.shape[0]))
        vw.write(img)
        with cond:
            latest["frame"] = img; cond.notify_all()
        n += 1
        if time.time() - t_win >= 1.0:
            fps = n / (time.time() - t_win); print("FPS %.1f" % fps, flush=True); n = 0; t_win = time.time()
except KeyboardInterrupt:
    pass
cap.release()
if vw: vw.release()
print("saved:", OUT)
EOF
ls -la /root/*.py
```

You should see **5 files**:

* `yolo_decode.py`
* `coco_names.py`
* `run_native_save.py`
* `run_video_native.py`
* `run_cam_save.py`

---

## Step 3: Run the three tests (the demo)

```bash
cd /root
source /etc/vai.sh
D=/run/media/mmcblk0p1/demo_results
mkdir -p $D

# 1. Webcam – open http://<board-ip>:8080 on your laptop, Ctrl+C after 10-20 s
python3 run_cam_save.py 0 8080 $D/webcam_out.avi 2>&1 | tee $D/webcam_log.txt

# 2. Video
python3 run_video_native.py /root/yellow_traffic_signs.avi $D/video_out.avi 2>&1 | tee $D/video_log.txt

# 3. 100 images
python3 run_native_save.py /root/yolo100 100 1 $D/images 2>&1 | tee $D/images_log.txt
```

Each test should print:

```
NPU only mode set. Skipping node wrp_network_embd_export_2_cpu_subgraph_call
```

That line is proof that the CPU subgraph was skipped.  
All three scripts run **NPU-only** because `npu_only=True` is hardcoded.

---

## Step 4: Show the results

```bash
cd /run/media/mmcblk0p1/demo_results && python3 -m http.server 8000
```

On your laptop browser open:

```
http://<board-ip>:8000/images/
```

The board IP was `192.168.1.85`. On a new system, check it with:

```bash
ip -4 addr
```

For the videos, copy them to your laptop with `scp` and play them in VLC (browsers cannot play MJPG `.avi`):

```bash
scp root@192.168.1.85:/run/media/mmcblk0p1/demo_results/*.avi .
```

---

## Notes

* Before demo day, run the full procedure once on the fresh system.
* The one thing that cannot be guaranteed is the camera — a different webcam may not support MJPG at 640×480 or may not appear as `/dev/video0`.
