RM500Q-GL drive-test logger and coverage map
Logs cellular metrics and GPS position from a Quectel RM500Q-GL
(RMU500-EVQ1 / RMU500EK evaluation kit) over its AT port, then renders a
coverage map with a heat map from the log.
File	Purpose
`quectel.py`	AT port discovery, network attach, GNSS, response parsers
`netmetrics.py`	RTT and throughput over the data connection
`log_drive.py`	the logger: writes one CSV row per sample
`coverage_map.py`	turns a CSV into an interactive HTML map
Install
```bash
python -m pip install -r requierements.txt
```
This machine runs Python 3.8, so `xyzservices` is pinned — newer releases use
`str.removesuffix()`, which is 3.9+. On Python 3.9 or later the pin is skipped.
Before you start
The module has to expose a serial AT port. Right now it is enumerated as
`USB\VID_2C7C&PID_0800` in MBIM mode with only the network interface bound
(`cxwmbclass`), so no COM port exists yet. Install the Quectel Windows USB
driver for the RM500 series; that binds the serial interfaces and you get
"Quectel USB AT Port (COMxx)". Then check:
```bash
python log_drive.py --list-ports
```
Also make sure both main/diversity cellular antennas and the active GNSS
antenna are connected — GNSS gets no fix without its own antenna, and a cold
fix outdoors takes a few minutes.
Logging
```bash
python log_drive.py --apn internet --interval 5
```
It sets the APN, enables the radio (`AT+CFUN=1`), waits for registration, starts
GNSS, then samples in a loop until Ctrl+C and writes
`logs/drive_<timestamp>.csv`. Each line is also echoed to the console:
```
14:03:21  NR5G-SA   RSRP -92    RSRQ -11   SINR 14    RSSI -70    CQI -   RTT 24.3 ms  DL 187.5 Mbps GPS 48.13743,11.57549
```
Useful options:
Option	Meaning
`--port COM12`	skip auto-detection
`--interval 2`	seconds between samples (default 5)
`--duration 600`	stop after N seconds
`--require-fix`	drop rows that have no GPS position
`--no-gps` / `--no-rtt`	skip those measurements
`--speedtest-every 10`	throughput test every Nth sample, `0` to disable
`--download-bytes` / `--upload-bytes`	test sizes (10 MB / 2 MB)
Rows are written even without a GPS fix (unless `--require-fix`), so the radio
metrics are never lost while the receiver is still searching.
Where each metric comes from
Metric	Source
RSRP, RSRQ, SINR (SNR), CQI, TX power, cell identity	`AT+QENG="servingcell"`
RSSI	`AT+QENG` on LTE; `AT+QCSQ` / `AT+CSQ` on NR5G-SA, which report no RSSI in QENG
Fallback RSRP/RSRQ/SINR	`AT+QCSQ` (its raw SINR index is converted with `dB = raw/5 - 20`)
Position, altitude, speed, HDOP, satellites	`AT+QGPSLOC=2` (signed decimal degrees)
RTT	median of 3 TCP handshakes to 1.1.1.1:443
Throughput	HTTP download/upload against `speed.cloudflare.com`
Three things worth knowing:
CQI is an LTE-only field. On an NR5G-SA cell the module does not report it,
so `cqi` stays empty. It is populated on LTE and on the LTE anchor in EN-DC.
In EN-DC (NR5G-NSA) the module prints an LTE line plus an NR line; the NR
values win and `rat` reads `NR5G-NSA`. The full untouched responses are kept in
the `raw_qeng` / `raw_qcsq` columns, so nothing is lost if your firmware orders
the fields differently.
RTT and throughput need a working IP path. Windows brings that up through
the module's MBIM/RNDIS network adapter — the AT port alone is not enough. If
the laptop has no cellular IP route, those columns stay empty (or measure your
Wi-Fi instead, if that is the active route); use `--no-rtt --speedtest-every 0`
to skip them. RTT is measured as a TCP handshake rather than ICMP so it needs
no privileges and no parsing of localised `ping` output.
Coverage map
```bash
python coverage_map.py logs/drive_YYYYMMDD_HHMMSS.csv --metric rsrp_dbm
python coverage_map.py logs/drive_x.csv --metric sinr_db
```
Metrics: `rsrp_dbm`, `rsrq_db`, `sinr_db`, `rssi_dbm`. Output is a self-contained HTML file with three toggleable
layers:
Heat map — a weighted density surface: intensity mixes how good the signal
was with how many samples you took there, so a spot you drove past slowly
glows more. Read trends from it, not exact values.
Measurement points — one dot per sample on a five-step scale, light =
weak, dark = strong. Hover for the value, click for the whole reading.
Route — the drive track, off by default.
The legend lists each band with its value range and how many points fell in it.
Exact per-point numbers live in the popups and in the CSV itself.
Colour scales are fixed so two runs are comparable; the ends are set in
`METRICS` at the top of `coverage_map.py` (RSRP -120..-70 dBm, SINR -5..30 dB,
and so on). Change them there if your network needs different thresholds.
`--radius` controls the heat map's spread and `--blur` controls its blur.
Known limitation
The colour palette is the pre-validated sequential blue ramp from the reference
design system. The palette validator that would re-check it is a Node script and
Node is not installed here, so the ramp was taken as documented rather than
re-measured.