"""EdgeTech JSF reader: real 16-byte message header, message 80 traces, 16-bit samples, port/starboard pairing."""

import io
import struct

import numpy as np
import pytest

from utils.sonar_raw_ingestion import JSFReader, ingest_raw_sonar_file


def _msg(mtype: int, channel: int, payload: bytes) -> bytes:
    return struct.pack("<HBBHBBBBHI", 0x1601, 1, 0, mtype, 0, 21, channel, 0, 0, len(payload)) + payload


def _trace(t: int, ping: int, samples: np.ndarray, wexp: int = 7) -> bytes:
    h = bytearray(240)
    struct.pack_into("<iii", h, 0, t, 0, ping)
    struct.pack_into("<h", h, 34, 0)                       # 16-bit envelope
    struct.pack_into("<H", h, 114, len(samples))
    struct.pack_into("<I", h, 116, 15360)
    struct.pack_into("<h", h, 168, wexp)
    return bytes(h) + samples.astype("<u2").tobytes()


def make_jsf(n_pings=40, n_samp=1500, nav=False, junk_messages=True) -> bytes:
    rng = np.random.default_rng(0)
    out = bytearray()
    if junk_messages:
        out += _msg(426, 0, b"\x00" * 8)
    for i in range(n_pings):
        for ch in (0, 1):
            s = rng.gamma(2.0, 800.0, n_samp)
            s[700:720] = 20000 if (ch == 1 and 10 <= i < 20) else s[700:720]     # a bright starboard target
            tr = bytearray(_trace(1439370275 + i // 6, 100 + (i * 7) % 13, s))     # ping numbers deliberately not monotonic
            if nav:
                struct.pack_into("<iih", tr, 80, int(80.27 * 600000), int(13.08 * 600000), 2)
            out += _msg(80, ch, bytes(tr))
        if junk_messages and i == 5:
            out += _msg(182, 0, b"\x01" * 300)
    return bytes(out)


def test_real_format_decodes_and_pairs_channels_in_order():
    wf, rec, meta = JSFReader().read(make_jsf())
    assert meta["num_pings"] == 40 and len(rec) == 40 and meta["resyncs"] == 0
    assert wf.shape == (40, 2 * 1024 + 6, 3) and wf.dtype == np.uint8
    assert meta["native_samples_per_channel"] == 1500 and meta["sample_interval_ns"] == 15360
    assert meta["max_slant_range_m"] == pytest.approx(1500 * 15360e-9 * 750, rel=1e-3)


def test_bright_target_lands_on_the_starboard_side_only():
    wf, _, _ = JSFReader().read(make_jsf())
    g = wf[..., 0].astype(float)
    port, stbd = g[10:20, :1024], g[10:20, 1030:]
    col = int(700 / 1500 * 1024)
    assert stbd[:, col:col + 10].mean() > stbd.mean() + 30 and port[:, 1024 - col - 14:1024 - col].mean() < port.mean() + 30


def test_ping_times_are_real_and_navigation_flag_reflects_the_header():
    _, rec, meta = JSFReader().read(make_jsf())
    assert rec[0].timestamp == 1439370275 and meta["navigation_synthetic"] is True
    _, rec2, meta2 = JSFReader().read(make_jsf(nav=True))
    assert meta2["navigation_synthetic"] is False and rec2[0].latitude == pytest.approx(13.08, abs=1e-3)


def test_dispatcher_accepts_bytes_bytesio_and_path(tmp_path):
    raw = make_jsf()
    p = tmp_path / "a.jsf"
    p.write_bytes(raw)
    shapes = {ingest_raw_sonar_file(x, "a.jsf")[0].shape for x in (raw, io.BytesIO(raw), p)}
    assert len(shapes) == 1


def test_corruption_is_resynchronised_and_non_sidescan_files_are_rejected():
    raw = make_jsf()
    damaged = raw[:3000] + b"\xde\xad\xbe\xef" * 50 + raw[3000:]
    _, _, meta = JSFReader().read(damaged)
    assert meta["resyncs"] >= 1 and meta["num_pings"] >= 30
    with pytest.raises(ValueError):
        JSFReader().read(_msg(426, 0, b"\x00" * 8))
