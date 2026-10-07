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

# NPU Snapshot – Run YOLOv7 on VEK280 (board-side)

Run these commands **on the VEK280 board** (after SSH).

## Step 1 — Set the snapshot variable

Copy only this line and press Enter:

```bash
export SNAPSHOT=/run/media/mmcblk0p1/snapshot.VE2802_NPU_IP_O00_A304_M3.yolov7.TF
```

## Step 2 — Verify it

```bash
echo "$SNAPSHOT"
```

Expected output:

```
/run/media/mmcblk0p1/snapshot.VE2802_NPU_IP_O00_A304_M3.yolov7.TF
```

## Step 3 — Verify the directory

```bash
ls -ld "$SNAPSHOT"
```

Expected output (similar to):

```
drwxr-xr-x ... /run/media/mmcblk0p1/snapshot.VE2802_NPU_IP_O00_A304_M3.yolov7.TF
```

> Note: `]ls -ld "$SNAPSHOT"` is a typo. The correct command is `ls -ld "$SNAPSHOT"`.

## Step 4 — Run NPU-only

Only after the previous steps succeed, run:

```bash
python3 /usr/bin/vart_ml_runner.py --snapshot "$SNAPSHOT" --npu_only
```

Do not type anything else on the same line.
