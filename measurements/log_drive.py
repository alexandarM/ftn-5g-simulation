"""Drive-test logger for the Quectel RM500Q-GL (RMU500-EVQ1 kit).

Attaches to the network, starts GNSS, then logs position and radio metrics
(RSRP, RSRQ, SINR/SNR, RSSI, CQI, RTT, throughput) to a CSV file.

    python log_drive.py --apn internet --interval 5
    python log_drive.py --list-ports
"""

import argparse
import csv
import datetime
import os
import sys
import time

import serial

import netmetrics
import quectel

CSV_FIELDS = [
    "timestamp_utc", "timestamp_local",
    "lat", "lon", "altitude_m", "hdop", "satellites", "speed_kmh",
    "course_deg", "gps_fix",
    "rat", "operator", "mcc", "mnc", "tac", "cell_id", "pci",
    "band", "arfcn", "earfcn", "dl_bw",
    "rsrp_dbm", "rsrq_db", "sinr_db", "rssi_dbm", "cqi", "tx_power_dbm",
    "csq_rssi_dbm", "csq_ber",
    "rtt_ms", "dl_mbps", "ul_mbps",
    "raw_qeng", "raw_qcsq",
]


def build_row(cell, signal, position, operator, rtt, down, up):
    """Flatten the modem readings into one CSV row.

    QENG is the primary source; QCSQ fills the gaps (NR5G-SA, for example,
    reports no RSSI in QENG). CQI is only reported on LTE, so it stays empty
    on a 5G SA cell.
    """
    now = datetime.datetime.now()
    row = {
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_local": now.strftime("%Y-%m-%d %H:%M:%S"),
        "operator": operator,
        "rat": cell.get("rat") or signal.get("sysmode", ""),
        "mcc": cell.get("mcc", ""),
        "mnc": cell.get("mnc", ""),
        "tac": cell.get("tac", ""),
        "cell_id": cell.get("cell_id", ""),
        "pci": cell.get("pci", ""),
        "band": cell.get("band", ""),
        "arfcn": cell.get("arfcn", ""),
        "earfcn": cell.get("earfcn", ""),
        "dl_bw": cell.get("dl_bw", ""),
        "rsrp_dbm": _first(cell.get("rsrp"), signal.get("qcsq_rsrp")),
        "rsrq_db": _first(cell.get("rsrq"), signal.get("qcsq_rsrq")),
        "sinr_db": _first(cell.get("sinr"), signal.get("qcsq_sinr")),
        "rssi_dbm": _first(cell.get("rssi"), signal.get("qcsq_rssi"),
                           signal.get("csq_rssi")),
        "cqi": cell.get("cqi", ""),
        "tx_power_dbm": cell.get("tx_power", ""),
        "csq_rssi_dbm": signal.get("csq_rssi", ""),
        "csq_ber": signal.get("csq_ber", ""),
        "rtt_ms": rtt if rtt is not None else "",
        "dl_mbps": down if down is not None else "",
        "ul_mbps": up if up is not None else "",
        "raw_qeng": cell.get("raw_qeng", ""),
        "raw_qcsq": signal.get("raw_qcsq", ""),
    }
    for key in ("lat", "lon", "altitude_m", "hdop", "satellites", "speed_kmh",
                "course_deg", "gps_fix"):
        value = position.get(key)
        row[key] = value if value is not None else ""
    return row


def _first(*values):
    for value in values:
        if value is not None:
            return value
    return ""


def summarise(row):
    def show(key, unit=""):
        value = row.get(key, "")
        return "%s%s" % (value, unit) if value != "" else "-"

    position = "-"
    if row["lat"] != "" and row["lon"] != "":
        position = "%.5f,%.5f" % (row["lat"], row["lon"])

    return ("%s  %-9s RSRP %-7s RSRQ %-6s SINR %-6s RSSI %-7s CQI %-3s "
            "RTT %-8s DL %-8s GPS %s") % (
        row["timestamp_local"][11:],
        row["rat"] or "-",
        show("rsrp_dbm"), show("rsrq_db"), show("sinr_db"), show("rssi_dbm"),
        show("cqi"), show("rtt_ms", " ms"), show("dl_mbps", " Mbps"), position,
    )


def connect(modem, apn, wait):
    print("Module : %s" % modem.initialise())
    print("Port   : %s" % modem.port)
    print("SIM    : %s" % modem.sim_status())
    if apn:
        modem.set_apn(apn)
        print("APN    : %s" % apn)
    modem.radio_on()

    print("Waiting for network registration ...")
    stat, source = modem.wait_for_registration(timeout=wait)
    if stat is None:
        print("Not registered after %.0fs - logging anyway; check SIM, APN and "
              "antennas." % wait)
    else:
        print("Attached: %s stat=%d, operator %s"
              % (source, stat, modem.operator() or "unknown"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="AT port, e.g. COM12 (auto-detected if omitted)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--apn", help="APN to program into PDP context 1")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="seconds between samples (default 5)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop after N seconds (default: run until Ctrl+C)")
    parser.add_argument("--csv", help="output file (default logs/drive_<timestamp>.csv)")
    parser.add_argument("--reg-timeout", type=float, default=90.0,
                        help="how long to wait for network registration")
    parser.add_argument("--no-gps", action="store_true", help="skip GNSS entirely")
    parser.add_argument("--require-fix", action="store_true",
                        help="only write rows that have a GPS position")
    parser.add_argument("--rtt-host", default="1.1.1.1")
    parser.add_argument("--rtt-port", type=int, default=443)
    parser.add_argument("--rtt-count", type=int, default=3)
    parser.add_argument("--no-rtt", action="store_true")
    parser.add_argument("--speedtest-every", type=int, default=10,
                        help="run a throughput test every N samples (0 = never)")
    parser.add_argument("--download-bytes", type=int, default=10000000)
    parser.add_argument("--upload-bytes", type=int, default=2000000,
                        help="0 disables the upload test")
    parser.add_argument("--list-ports", action="store_true",
                        help="list the Quectel serial ports and exit")
    args = parser.parse_args(argv)

    if args.list_ports:
        ports = quectel.candidate_ports()
        print("\n".join(ports) if ports else "No Quectel serial port found.")
        return 0

    path = args.csv or os.path.join(
        "logs", "drive_%s.csv" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)

    try:
        modem = quectel.open_at_port(args.port, args.baud)
    except quectel.ATError as exc:
        print("Error: %s" % exc)
        return 1

    handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()

    samples = 0
    written = 0
    status = 0
    try:
        connect(modem, args.apn, args.reg_timeout)
        operator = modem.operator()

        if not args.no_gps:
            modem.gnss_start()
            print("GNSS   : started (a cold fix can take a few minutes)")

        print("Logging to %s - press Ctrl+C to stop.\n" % path)
        started = time.time()

        while True:
            loop_start = time.time()
            samples += 1

            try:
                cell = modem.serving_cell()
            except quectel.ATError as exc:
                print("QENG failed: %s" % exc)
                cell = {}
            signal = modem.signal_quality()
            position = {} if args.no_gps else modem.location()
            if not operator:
                # Attach may have completed after startup.
                operator = modem.operator()

            rtt = None
            if not args.no_rtt:
                rtt = netmetrics.tcp_rtt_ms(args.rtt_host, args.rtt_port,
                                            args.rtt_count)

            down = up = None
            if args.speedtest_every and (samples - 1) % args.speedtest_every == 0:
                print("  running throughput test ...")
                down = netmetrics.download_mbps(args.download_bytes)
                if args.upload_bytes:
                    up = netmetrics.upload_mbps(args.upload_bytes)

            row = build_row(cell, signal, position, operator, rtt, down, up)

            if args.require_fix and row["lat"] == "":
                print("%s  no GPS fix yet - row skipped" % row["timestamp_local"][11:])
            else:
                writer.writerow(row)
                handle.flush()
                written += 1
                print(summarise(row))

            if args.duration and time.time() - started >= args.duration:
                break

            remaining = args.interval - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except (quectel.ATError, serial.SerialException) as exc:
        print("\nModem error: %s" % exc)
        status = 1
    finally:
        if not args.no_gps:
            modem.gnss_stop()
        modem.close()
        handle.close()
        print("\n%d rows written to %s" % (written, path))
        if written:
            print("Now build the map:  python coverage_map.py \"%s\"" % path)
    return status


if __name__ == "__main__":
    sys.exit(main())
