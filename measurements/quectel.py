"""AT-command interface for the Quectel RM500Q-GL (RMU500EK / RMU500-EVQ1 kit).

Only what the drive-test logger needs: open the AT port, attach to the network,
start GNSS, and read signal metrics.
"""

import time

import serial
import serial.tools.list_ports

QUECTEL_VID = 0x2C7C

# Fields of AT+QENG="servingcell", per radio access technology.
# The response starts with "servingcell" for the serving-cell line; the extra
# NR5G-NSA line emitted in EN-DC mode starts with the RAT name directly.
QENG_FIELDS = {
    "LTE": [
        "state", "rat", "is_tdd", "mcc", "mnc", "cell_id", "pci", "earfcn",
        "band", "ul_bw", "dl_bw", "tac", "rsrp", "rsrq", "rssi", "sinr",
        "cqi", "tx_power", "srxlev",
    ],
    "NR5G-SA": [
        "state", "rat", "duplex", "mcc", "mnc", "cell_id", "pci", "tac",
        "arfcn", "band", "dl_bw", "rsrp", "rsrq", "sinr", "scs", "srxlev",
    ],
    "NR5G-NSA": [
        "rat", "mcc", "mnc", "pci", "rsrp", "sinr", "rsrq",
    ],
    "WCDMA": [
        "state", "rat", "mcc", "mnc", "lac", "cell_id", "uarfcn", "psc",
        "rac", "rscp", "ecio",
    ],
    "GSM": [
        "state", "rat", "mcc", "mnc", "lac", "cell_id", "bsic", "arfcn",
        "band", "rxlev",
    ],
}

# Identifiers that must stay text (hex cell IDs, zero-padded MNCs, "TDD", ...).
STRING_FIELDS = {"state", "rat", "is_tdd", "duplex", "mcc", "mnc", "cell_id",
                 "tac", "lac", "band"}


class ATError(Exception):
    """An AT command returned ERROR / +CME ERROR, or timed out."""

    def __init__(self, message, response=""):
        Exception.__init__(self, message)
        self.response = response


def _num(token):
    """Parse an AT field into int/float, or None if it carries no value."""
    token = token.strip().strip('"')
    if token in ("", "-", "--"):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return None


class Modem(object):
    def __init__(self, port, baudrate=115200, timeout=1.0):
        self.port = port
        self.serial = serial.Serial(port, baudrate, timeout=timeout)

    def close(self):
        try:
            self.serial.close()
        except Exception:
            pass

    def send(self, command, timeout=5.0):
        """Send one AT command and return its response lines (without "OK")."""
        self.serial.reset_input_buffer()
        self.serial.write((command + "\r\n").encode("ascii"))
        self.serial.flush()

        lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self.serial.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", "replace").strip()
            if not line or line == command:
                continue
            if line == "OK":
                return lines
            if line == "ERROR" or line.startswith(("+CME ERROR", "+CMS ERROR")):
                raise ATError("%s -> %s" % (command, line), line)
            lines.append(line)
        raise ATError("%s -> no response within %.1fs" % (command, timeout))

    # setup

    def initialise(self):
        """Quiet echo, verbose errors, and return the module identification."""
        self.send("ATE0")
        self.send("AT+CMEE=2")
        return " | ".join(self.send("ATI"))

    def sim_status(self):
        try:
            for line in self.send("AT+CPIN?"):
                if line.startswith("+CPIN:"):
                    return line.split(":", 1)[1].strip()
        except ATError as exc:
            return "not ready (%s)" % exc.response
        return "unknown"

    def set_apn(self, apn, cid=1, pdp_type="IPV4V6"):
        self.send('AT+CGDCONT=%d,"%s","%s"' % (cid, pdp_type, apn))

    def radio_on(self):
        """Make sure the radio is enabled (CFUN=1) so the module can attach."""
        for line in self.send("AT+CFUN?"):
            if line.startswith("+CFUN:") and _num(line.split(":", 1)[1]) == 1:
                return
        self.send("AT+CFUN=1", timeout=15.0)

    def registration(self):
        """Return (stat, source) for the first registered domain found.

        stat 1 = registered (home), 5 = registered (roaming).
        """
        for command in ("AT+C5GREG?", "AT+CEREG?", "AT+CREG?"):
            try:
                lines = self.send(command)
            except ATError:
                continue
            for line in lines:
                if ":" not in line:
                    continue
                parts = line.split(":", 1)[1].split(",")
                stat = _num(parts[1]) if len(parts) > 1 else None
                if stat in (1, 5):
                    return stat, command.strip("AT+?")
        return None, None

    def wait_for_registration(self, timeout=90.0, poll=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            stat, source = self.registration()
            if stat is not None:
                return stat, source
            time.sleep(poll)
        return None, None

    def operator(self):
        try:
            for line in self.send("AT+COPS?"):
                if line.startswith("+COPS:"):
                    parts = line.split(":", 1)[1].split(",")
                    if len(parts) >= 3:
                        return parts[2].strip().strip('"')
        except ATError:
            pass
        return ""

    #  GNSS

    def gnss_start(self):
        """Turn GNSS on, unless a session is already running."""
        try:
            for line in self.send("AT+QGPS?"):
                if line.startswith("+QGPS:") and _num(line.split(":", 1)[1]) == 1:
                    return
        except ATError:
            pass
        self.send("AT+QGPS=1", timeout=10.0)

    def gnss_stop(self):
        try:
            self.send("AT+QGPSEND")
        except Exception:
            pass  # called during shutdown; a dead port must not mask the exit

    def location(self):
        """Current fix as a dict, or {} while the module has no fix yet."""
        try:
            return parse_qgpsloc(self.send("AT+QGPSLOC=2"))
        except ATError:
            # +CME ERROR: 516 = "not fixed yet"; anything else is equally
            # non-fatal for a drive test, so just report "no position".
            return {}

    # -- radio metrics -------------------------------------------------------

    def serving_cell(self):
        lines = self.send('AT+QENG="servingcell"')
        data = parse_qeng(lines)
        data["raw_qeng"] = " ; ".join(lines)
        return data

    def signal_quality(self):
        data = {}
        raw = []
        try:
            lines = self.send("AT+QCSQ")
            raw = lines
            data.update(parse_qcsq(lines))
        except ATError:
            pass
        try:
            data.update(parse_csq(self.send("AT+CSQ")))
        except ATError:
            pass
        data["raw_qcsq"] = " ; ".join(raw)
        return data


#  response parsers

def parse_qeng(lines):
    """Merge every +QENG line into one dict of named fields.

    In EN-DC the module prints the LTE anchor first and the NR5G-NSA line
    second, so the NR values (rsrp/rsrq/sinr/pci) overwrite the LTE ones and
    the resulting "rat" says NR5G-NSA.
    """
    merged = {}
    for line in lines:
        if not line.startswith("+QENG") or ":" not in line:
            continue
        tokens = [t.strip().strip('"') for t in line.split(":", 1)[1].split(",")]
        if tokens and tokens[0] == "servingcell":
            tokens = tokens[1:]
            rat_index = 1
        else:
            rat_index = 0
        if len(tokens) <= rat_index:
            continue
        rat = tokens[rat_index].upper()
        fields = QENG_FIELDS.get(rat)
        if fields is None:
            continue
        for name, token in zip(fields, tokens):
            if token in ("", "-"):
                continue
            value = token if name in STRING_FIELDS else _num(token)
            if value is not None:
                merged[name] = value
    return merged


def parse_qcsq(lines):
    """+QCSQ: "<sysmode>",<rssi>,<rsrp>,<sinr>,<rsrq>

    The QCSQ <sinr> is a raw 0..250 index; dB = raw / 5 - 20.
    """
    for line in lines:
        if not line.startswith("+QCSQ") or ":" not in line:
            continue
        tokens = [t.strip().strip('"') for t in line.split(":", 1)[1].split(",")]
        sysmode = tokens[0]
        values = [_num(t) for t in tokens[1:]]
        data = {"sysmode": sysmode}
        if sysmode.upper().startswith(("LTE", "NR5G")) and len(values) >= 4:
            data["qcsq_rssi"] = values[0]
            data["qcsq_rsrp"] = values[1]
            if values[2] is not None:
                data["qcsq_sinr"] = round(values[2] / 5.0 - 20.0, 1)
            data["qcsq_rsrq"] = values[3]
        elif values:
            data["qcsq_rssi"] = values[0]
        return data
    return {}


def parse_csq(lines):
    """+CSQ: <rssi_index>,<ber>  -- index 0..31 maps to -113..-51 dBm, 99 = unknown."""
    for line in lines:
        if not line.startswith("+CSQ") or ":" not in line:
            continue
        tokens = line.split(":", 1)[1].split(",")
        index = _num(tokens[0])
        ber = _num(tokens[1]) if len(tokens) > 1 else None
        data = {}
        if index is not None and index != 99:
            data["csq_rssi"] = -113 + 2 * index
        if ber is not None and ber != 99:
            data["csq_ber"] = ber
        return data
    return {}


def parse_qgpsloc(lines):
    """+QGPSLOC: <utc>,<lat>,<lon>,<hdop>,<alt>,<fix>,<cog>,<spkm>,<spkn>,<date>,<nsat>

    Read with AT+QGPSLOC=2, so latitude/longitude are signed decimal degrees.
    """
    for line in lines:
        if not line.startswith("+QGPSLOC") or ":" not in line:
            continue
        tokens = [t.strip() for t in line.split(":", 1)[1].split(",")]

        def field(index):
            return tokens[index] if index < len(tokens) else ""

        return {
            "gps_utc": field(0),
            "lat": _num(field(1)),
            "lon": _num(field(2)),
            "hdop": _num(field(3)),
            "altitude_m": _num(field(4)),
            "gps_fix": _num(field(5)),
            "course_deg": _num(field(6)),
            "speed_kmh": _num(field(7)),
            "gps_date": field(9),
            "satellites": _num(field(10)),
        }
    return {}


#port discovery

def candidate_ports():
    """Serial ports that look like they belong to the Quectel module."""
    found = []
    for port in serial.tools.list_ports.comports():
        text = "%s %s" % (port.description or "", port.hwid or "")
        if port.vid == QUECTEL_VID or "quectel" in text.lower():
            found.append(port.device)
    return found


def open_at_port(port=None, baudrate=115200):
    """Open the AT port, probing the Quectel ports if none was given."""
    ports = [port] if port else candidate_ports()
    if not ports:
        raise ATError(
            "No Quectel serial port found. Plug in the RMU500 kit and install "
            "the Quectel USB driver, or pass --port COMxx."
        )

    errors = []
    for device in ports:
        try:
            modem = Modem(device, baudrate, timeout=0.5)
        except (serial.SerialException, OSError) as exc:
            errors.append("%s: %s" % (device, exc))
            continue
        try:
            modem.send("AT", timeout=2.0)
            modem.serial.timeout = 1.0
            return modem
        except (ATError, serial.SerialException) as exc:
            errors.append("%s: %s" % (device, exc))
            modem.close()

    raise ATError("No port answered AT (tried %s)" % "; ".join(errors))
