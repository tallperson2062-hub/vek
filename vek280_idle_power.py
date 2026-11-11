#!/usr/bin/env python3
"""
VEK280 power profiler (live per-rail display, corrected shunt map)

Reads INA226 shunt-voltage + bus-voltage registers through the TCA9548A mux.
Current = measured_shunt_voltage / Rshunt  (calibration registers are ignored).

- Live: every sample shows each rail (V, A, W) and the running total.
- End of run (duration reached or Ctrl+C): summary with per-rail avg/min/max.
- Reports the sum of the 19 INA226-monitored rails only, NOT total board power.
Run as root on the VEK280.
"""

import argparse
import csv
import fcntl
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

I2C_DEVICE = "/dev/i2c-0"
I2C_SLAVE  = 0x0703
MUX_ADDR   = 0x74

# (mux_channel, i2c_addr, rail_name, shunt_ohms)  -- AMD VEK280 revA device-tree
RAILS = [
    (1, 0x40, "VCCINT",           0.0005),
    (1, 0x41, "VCC_SOC",          0.0005),
    (1, 0x42, "VCC_PMC",          0.005),
    (1, 0x43, "VCC_RAM",          0.005),
    (1, 0x44, "VCC_PSLP_CPM5",    0.005),
    (1, 0x45, "VCC_PSFP",         0.005),
    (1, 0x46, "VCCO_HDIO_3V3",    0.005),
    (3, 0x40, "VCCAUX",           0.005),
    (3, 0x41, "VCCAUX_PMC",       0.005),
    (3, 0x42, "MGTAVCC",          0.005),
    (3, 0x43, "VCC1V5",           0.005),
    (3, 0x45, "VCCO_MIO",         0.005),
    (3, 0x46, "MGTAVTT",          0.0005),
    (3, 0x47, "VCCO_502",         0.005),
    (3, 0x48, "MGTVCCAUX",        0.005),
    (3, 0x49, "VCC1V1_LP4",       0.0005),
    (3, 0x4A, "VADJ_FMC",         0.005),
    (3, 0x4B, "LPDMGTYAVCC",      0.005),
    (3, 0x4C, "LPDMGTYAVTT",      0.005),
    (3, 0x4D, "LPDMGTYVCCAUX",    0.005),
]

REG_SHUNT_VOLTAGE = 0x01
REG_BUS_VOLTAGE   = 0x02
SHUNT_LSB_V       = 2.5e-6
BUS_LSB_V         = 1.25e-3


def select_device(fd, address):
    fcntl.ioctl(fd, I2C_SLAVE, address)


def read_register(fd, address, register):
    select_device(fd, address)
    os.write(fd, bytes([register]))
    data = os.read(fd, 2)
    if len(data) != 2:
        raise OSError(f"Short I2C read from 0x{address:02X} reg 0x{register:02X}")
    return (data[0] << 8) | data[1]


def select_mux_channel(fd, channel):
    select_device(fd, MUX_ADDR)
    os.write(fd, bytes([1 << channel]))
    time.sleep(0.002)


def signed16(value):
    return value - 65536 if value & 0x8000 else value


def read_rail(fd, rail):
    channel, address, name, shunt_ohms = rail
    select_mux_channel(fd, channel)
    shunt_raw = read_register(fd, address, REG_SHUNT_VOLTAGE)
    bus_raw   = read_register(fd, address, REG_BUS_VOLTAGE)
    shunt_v = signed16(shunt_raw) * SHUNT_LSB_V
    bus_v   = bus_raw * BUS_LSB_V
    current = shunt_v / shunt_ohms
    power   = bus_v * current
    return {
        "rail": name,
        "mux_channel": channel,
        "i2c_address": f"0x{address:02X}",
        "shunt_ohms": shunt_ohms,
        "bus_voltage_v": bus_v,
        "shunt_voltage_v": shunt_v,
        "current_a": current,
        "power_w": power,
    }


def render_live(sample_no, elapsed, results, total, stats, totals, clear):
    """Show every rail for the current sample + running total."""
    lines = []
    lines.append(f"VEK280 power | sample {sample_no} | elapsed {elapsed:7.1f} s | Ctrl+C to stop")
    lines.append(f"{'Rail':<15}{'V':>8}{'I (A)':>10}{'P (W)':>10}{'Avg P (W)':>12}")
    lines.append("-" * 55)
    for r in results:
        st = stats[r["rail"]]
        avg_p = st["sum_p"] / st["n"]
        lines.append(f"{r['rail']:<15}{r['bus_voltage_v']:8.3f}{r['current_a']:10.4f}"
                     f"{r['power_w']:10.4f}{avg_p:12.4f}")
    lines.append("-" * 55)
    avg_t = sum(totals) / len(totals)
    lines.append(f"{'TOTAL (19 mon. rails)':<33}{total:10.3f}{avg_t:12.3f}   W")
    lines.append(f"total now {total:.3f} W | min {min(totals):.3f} | max {max(totals):.3f}"
                 " | monitored rails only")
    if clear:
        sys.stdout.write("\033[H\033[2J")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="VEK280 INA226 monitored-rail power (live per-rail display)"
    )
    parser.add_argument("--duration", type=float, default=60.0,
                        help="measurement duration in seconds (default 60)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="sampling interval in seconds (default 1)")
    parser.add_argument("--csv", default="vek280_idle_power.csv")
    parser.add_argument("--json", default="vek280_idle_summary.json")
    parser.add_argument("--scroll", action="store_true",
                        help="print a new table every sample instead of refreshing in place")
    args = parser.parse_args()

    if args.duration <= 0 or args.interval <= 0:
        parser.error("--duration and --interval must be positive")

    try:
        fd = os.open(I2C_DEVICE, os.O_RDWR)
    except OSError as exc:
        sys.exit(f"Cannot open {I2C_DEVICE}: {exc}. Run as root.")

    clear = sys.stdout.isatty() and not args.scroll

    totals = []
    stats = {r[2]: {"sum_p": 0.0, "min_p": float("inf"), "max_p": float("-inf"),
                    "sum_i": 0.0, "sum_v": 0.0, "n": 0} for r in RAILS}
    sample_count = 0
    sample_times = []
    start = time.monotonic()
    end   = start + args.duration

    fieldnames = [
        "timestamp_utc", "elapsed_s", "rail", "mux_channel",
        "i2c_address", "shunt_ohms", "bus_voltage_v",
        "shunt_voltage_v", "current_a", "power_w"
    ]

    try:
        with open(args.csv, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            while time.monotonic() < end:
                t0 = time.monotonic()
                ts = datetime.now(timezone.utc).isoformat()
                elapsed = t0 - start

                results = []
                errors = []
                for rail in RAILS:
                    try:
                        res = read_rail(fd, rail)
                        res["timestamp_utc"] = ts
                        res["elapsed_s"] = round(elapsed, 3)
                        writer.writerow(res)
                        results.append(res)
                    except (OSError, ValueError) as e:
                        errors.append(f"{rail[2]}: {e}")
                csv_file.flush()

                if len(results) == len(RAILS):
                    total = sum(r["power_w"] for r in results)
                    totals.append(total)
                    sample_times.append(t0)
                    sample_count += 1
                    for r in results:
                        st = stats[r["rail"]]
                        st["sum_p"] += r["power_w"]
                        st["sum_i"] += r["current_a"]
                        st["sum_v"] += r["bus_voltage_v"]
                        st["min_p"] = min(st["min_p"], r["power_w"])
                        st["max_p"] = max(st["max_p"], r["power_w"])
                        st["n"] += 1
                    render_live(sample_count, elapsed, results, total, stats, totals, clear)
                else:
                    print("\nIncomplete sample (omitted from totals):")
                    for e in errors:
                        print("  ", e)

                remaining = args.interval - (time.monotonic() - t0)
                if remaining > 0:
                    time.sleep(min(remaining, max(0.0, end - time.monotonic())))

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            select_device(fd, MUX_ADDR)
            os.write(fd, b"\x00")
        except OSError:
            pass
        os.close(fd)

    if not totals:
        sys.exit("No complete samples collected. Check I2C / rail map.")

    # Energy from the real timestamps
    energy_j = 0.0
    if len(sample_times) >= 2:
        for i in range(1, len(sample_times)):
            energy_j += totals[i - 1] * (sample_times[i] - sample_times[i - 1])
        energy_j += totals[-1] * (sample_times[-1] - sample_times[-2])
    else:
        energy_j = totals[0] * args.interval

    avg_total = statistics.mean(totals)
    per_rail = {}
    for name, st in stats.items():
        if st["n"]:
            per_rail[name] = {
                "avg_voltage_v": st["sum_v"] / st["n"],
                "avg_current_a": st["sum_i"] / st["n"],
                "avg_power_w": st["sum_p"] / st["n"],
                "min_power_w": st["min_p"],
                "max_power_w": st["max_p"],
            }

    summary = {
        "board": "AMD VEK280",
        "measurement": "sum of 19 INA226-monitored rails (corrected shunt map)",
        "total_board_input_power": False,
        "duration_requested_s": args.duration,
        "samples_completed": sample_count,
        "average_monitored_rail_power_w": avg_total,
        "minimum_monitored_rail_power_w": min(totals),
        "maximum_monitored_rail_power_w": max(totals),
        "population_stddev_w": statistics.pstdev(totals) if len(totals) > 1 else 0.0,
        "energy_j": energy_j,
        "energy_wh": energy_j / 3600.0,
        "per_rail": per_rail,
        "per_rail_average_power_w": {k: v["avg_power_w"] for k, v in per_rail.items()},
        "shunt_source": "zynqmp-sc-vek280-revA.dtso (AMD)",
        "warning": ("Sum of the configured INA226 rails only. Excludes the 12 V input, "
                    "unmonitored rails and losses. Not total board power."),
    }
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- final summary (only at the end) ----
    print("\n========== POWER SUMMARY ==========")
    print(f"Samples: {sample_count}")
    print(f"{'Rail':<15}{'Avg V':>8}{'Avg A':>10}{'Avg W':>10}{'Min W':>10}{'Max W':>10}{'% total':>9}")
    print("-" * 72)
    for name, v in sorted(per_rail.items(), key=lambda x: -x[1]["avg_power_w"]):
        share = 100.0 * v["avg_power_w"] / avg_total if avg_total else 0.0
        print(f"{name:<15}{v['avg_voltage_v']:8.3f}{v['avg_current_a']:10.4f}"
              f"{v['avg_power_w']:10.4f}{v['min_power_w']:10.4f}{v['max_power_w']:10.4f}{share:8.1f}%")
    print("-" * 72)
    print(f"Average monitored power: {avg_total:.4f} W")
    print(f"Minimum monitored power: {min(totals):.4f} W")
    print(f"Maximum monitored power: {max(totals):.4f} W")
    print(f"Energy (timestamp int.): {energy_j:.4f} J")
    print(f"CSV : {args.csv}")
    print(f"JSON: {args.json}")
    print("\nNOTE: monitored-rail sum only - not total board input power.")


if __name__ == "__main__":
    main()
