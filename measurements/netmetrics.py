"""Latency and throughput measured over the module's data connection.

RTT is a TCP-connect round trip instead of ICMP: it needs no privileges and no
parsing of localised `ping` output, and it still measures the cellular path.
"""

import socket
import statistics
import time

import requests

DOWNLOAD_URL = "https://speed.cloudflare.com/__down?bytes={bytes}"
UPLOAD_URL = "https://speed.cloudflare.com/__up"


def tcp_rtt_ms(host="1.1.1.1", port=443, count=3, timeout=2.0):
    """Median TCP handshake round-trip time in ms, or None if all attempts fail."""
    samples = []
    for _ in range(count):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        start = time.perf_counter()
        try:
            sock.connect((host, port))
            samples.append((time.perf_counter() - start) * 1000.0)
        except OSError:
            pass
        finally:
            sock.close()
    if not samples:
        return None
    return round(statistics.median(samples), 1)


def download_mbps(num_bytes=10000000, url=DOWNLOAD_URL, timeout=30.0):
    """Download num_bytes and return the rate in Mbit/s, or None on failure.

    Timing starts at the first received chunk so that DNS, TCP and TLS setup do
    not count against the throughput figure.
    """
    if "{bytes}" in url:
        url = url.format(bytes=num_bytes)
    received = 0
    start = None
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        for chunk in response.iter_content(65536):
            if start is None:
                start = time.perf_counter()
            received += len(chunk)
        response.close()
    except Exception:
        return None
    if start is None or received == 0:
        return None
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return None
    return round(received * 8 / elapsed / 1e6, 2)


def upload_mbps(num_bytes=2000000, url=UPLOAD_URL, timeout=30.0):
    """Upload num_bytes and return the rate in Mbit/s, or None on failure.

    The figure includes the server's response time, so it is slightly
    pessimistic for small payloads.
    """
    payload = b"\0" * num_bytes
    start = time.perf_counter()
    try:
        response = requests.post(
            url,
            data=payload,
            timeout=timeout,
            headers={"Content-Type": "application/octet-stream"},
        )
        response.raise_for_status()
    except Exception:
        return None
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return None
    return round(num_bytes * 8 / elapsed / 1e6, 2)
