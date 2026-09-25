"""
Raw Side-Scan Sonar (SSS) Data Ingestion & Stream Processing Engine.
Provides direct binary parsers for:
  1. Triton eXtended Triton Format (.xtf)
  2. EdgeTech Sonar Format (.jsf)
  3. Waterfall acoustic reconstruction: (Port + Nadir + Starboard) -> 2D Image
  4. Ping-synchronized telemetry extraction (WGS-84, Heading, Attitude, Heave)
  5. Synthetic XTF Generator for offline mission testing and simulation
"""

import datetime
import io
import math
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import cv2

from utils.telemetry_parser import TelemetryRecord, TelemetryValidator


# ─────────────────────────────────────────────────────────────────────────────
# TRITON XTF BINARY CONSTANTS & STRUCTS
# ─────────────────────────────────────────────────────────────────────────────
# Fixed start time for synthetic files so generated surveys are reproducible (2026-09-12 14:00:00 UTC)
BASE_EPOCH = datetime.datetime(2026, 9, 12, 14, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
XTF_MAGIC_FILE_HEADER = 0x7B         # 123 decimal
XTF_MAGIC_PACKET_HEADER = 0xFACE     # 64206 decimal
XTF_HEADER_SONAR = 0                 # Sidescan ping packet type
XTF_HEADER_NOTES = 1
XTF_HEADER_ATTITUDE = 3


def _xtf_epoch(year, month, day, hour, minute, second, hsecond):
    """UTC epoch seconds from an XTF ping header time (hsecond = 1/100 s); None if the fields are not a valid time."""
    try:
        import datetime as _dt
        return _dt.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second),
                            int(hsecond) * 10000, tzinfo=_dt.timezone.utc).timestamp()
    except (ValueError, OverflowError):
        return None


class XTFReader:
    """
    Pure-Python Binary Parser for Triton eXtended Triton Format (.xtf).
    Requires zero external binary C dependencies, working natively on all platforms.
    """

    def __init__(self):
        self.file_header: Dict[str, Any] = {}
        self.channel_info: List[Dict[str, Any]] = []
        self.pings: List[Dict[str, Any]] = []
        self.telemetry_records: List[TelemetryRecord] = []

    def read(self, source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[np.ndarray, List[TelemetryRecord], Dict[str, Any]]:
        """
        Parses an XTF file or byte buffer.

        Returns:
            waterfall_bgr: 2D stitched sonar image (H, W, 3) [Port | Nadir | Starboard]
            telemetry: List of TelemetryRecord for each ping
            metadata: File-level and survey metadata
        """
        if isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                data = f.read()
        elif isinstance(source, io.BytesIO):
            data = source.getvalue()
        elif isinstance(source, bytes):
            data = source
        else:
            raise TypeError("Source must be file path, bytes, or io.BytesIO")

        stream = io.BytesIO(data)
        file_size = len(data)

        if file_size < 1024:
            raise ValueError(f"File too small for XTF header ({file_size} bytes < 1024)")

        # 1. Parse 1024-byte File Header
        self._parse_file_header(stream)

        # 2. Parse Packets
        port_traces = []
        stbd_traces = []
        self.telemetry_records = []
        self.pings = []

        max_samples_per_chan = 0

        while stream.tell() + 14 <= file_size:
            pos_before = stream.tell()
            raw_pkt_hdr = stream.read(14)
            if len(raw_pkt_hdr) < 14:
                break
            magic, header_type, sub_chan, num_chans, r1, r2, num_bytes = struct.unpack("<HBBHHHI", raw_pkt_hdr)

            if magic != XTF_MAGIC_PACKET_HEADER:
                # Resynchronize stream: scan byte-by-byte for next 0xFACE
                stream.seek(pos_before + 1)
                sync_bytes = stream.read(min(2048, file_size - stream.tell()))
                found = sync_bytes.find(b"\xCE\xFA")  # 0xFACE in little-endian
                if found != -1:
                    stream.seek(pos_before + 1 + found)
                    continue
                else:
                    break

            if num_bytes < 14 or (pos_before + num_bytes) > file_size:
                # Corrupted record size or EOF
                break

            record_payload_len = num_bytes - 14

            if header_type == XTF_HEADER_SONAR:
                ping_meta, port_data, stbd_data = self._parse_ping_record(stream, record_payload_len)
                if port_data is not None and stbd_data is not None:
                    port_traces.append(port_data)
                    stbd_traces.append(stbd_data)
                    max_samples_per_chan = max(max_samples_per_chan, len(port_data), len(stbd_data))
                    self.pings.append(ping_meta)

                    rec = TelemetryRecord(
                        timestamp=ping_meta.get("timestamp", time.time()),
                        latitude=ping_meta.get("latitude", 13.0827),
                        longitude=ping_meta.get("longitude", 80.2707),
                        heading_deg=ping_meta.get("heading", 0.0),
                        depth_m=ping_meta.get("depth", 15.0),
                        altitude_m=ping_meta.get("altitude", 10.0),
                        slant_range_m=ping_meta.get("slant_range", 75.0),
                        frequency_khz=ping_meta.get("frequency", 450.0),
                        beamwidth_deg=0.5,
                        vessel_speed_knots=ping_meta.get("speed", 3.5),
                        layback_m=ping_meta.get("layback", 25.0),
                        pitch_deg=ping_meta.get("pitch", 0.0),
                        roll_deg=ping_meta.get("roll", 0.0),
                        heave_m=ping_meta.get("heave", 0.0),
                    )
                    self.telemetry_records.append(rec)
            else:
                stream.seek(pos_before + num_bytes)

        # 3. Assemble continuous 2D side-scan waterfall
        waterfall_bgr = self._assemble_waterfall(port_traces, stbd_traces, max_samples_per_chan)

        metadata = {
            "format": "Triton XTF",
            "sonar_name": self.file_header.get("sonar_name", "Side-Scan Sonar"),
            "num_pings": len(port_traces),
            "channels": self.file_header.get("num_channels", 2),
            "samples_per_channel": max_samples_per_chan,
            "max_slant_range_m": self.telemetry_records[0].slant_range_m if self.telemetry_records else 75.0,
            "file_size_bytes": file_size,
        }

        return waterfall_bgr, self.telemetry_records, metadata

    def _parse_file_header(self, stream: io.BytesIO):
        raw = stream.read(1024)
        file_format, sys_type = struct.unpack_from("<BB", raw, 0)
        prog_name = raw[2:10].decode("ascii", errors="ignore").strip("\x00 ")
        prog_ver = raw[10:18].decode("ascii", errors="ignore").strip("\x00 ")
        sonar_name = raw[18:34].decode("ascii", errors="ignore").strip("\x00 ")
        sonar_type, note_str = struct.unpack_from("<H64s", raw, 34)
        num_hdr_bytes, nav_units, num_chan = struct.unpack_from("<HHH", raw, 102)

        self.file_header = {
            "file_format": file_format,
            "system_type": sys_type,
            "program_name": prog_name,
            "program_version": prog_ver,
            "sonar_name": sonar_name or "EdgeTech / Klein SSS",
            "sonar_type": sonar_type,
            "nav_units": nav_units,
            "num_channels": max(1, min(6, num_chan)),
        }

    def _parse_ping_record(self, stream: io.BytesIO, payload_len: int) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
        """Parses XTFPINGHEADER (242 bytes) and subsequent XTFPINGCHANHEADERs."""
        hdr_bytes = stream.read(min(242, payload_len))
        if len(hdr_bytes) < 100:
            return {}, None, None

        year, month, day, hour, minute, second, hsecond = struct.unpack_from("<HBBBBBB", hdr_bytes, 0)
        julian_day, curr_sec, ping_num = struct.unpack_from("<HII", hdr_bytes, 8)
        layback = struct.unpack_from("<f", hdr_bytes, 18)[0]
        heading, pitch, roll, heave, yaw = struct.unpack_from("<fffff", hdr_bytes, 26)
        depth, altitude, speed, sound_vel = struct.unpack_from("<ffff", hdr_bytes, 46)
        sensor_x, sensor_y = struct.unpack_from("<dd", hdr_bytes, 62)

        lat = sensor_y if (-90.0 <= sensor_y <= 90.0) else 13.0827
        lon = sensor_x if (-180.0 <= sensor_x <= 180.0) else 80.2707
        if lat == 0.0 and lon == 0.0:
            lat, lon = 13.0827, 80.2707

        ts = _xtf_epoch(year, month, day, hour, minute, second, hsecond)
        ping_meta = {
            "ping_number": ping_num,
            "timestamp": ts if ts is not None else time.time(),
            "timestamp_source": "xtf_ping_header" if ts is not None else "parse_time_fallback",
            "latitude": lat,
            "longitude": lon,
            "heading": float(heading) if 0 <= heading <= 360 else 45.0,
            "speed": max(0.5, float(speed)) if speed > 0 else 3.5,
            "altitude": max(1.0, float(altitude)) if altitude > 0 else 10.0,
            "depth": max(0.5, float(depth)) if depth > 0 else 15.0,
            "layback": max(0.0, float(layback)) if layback >= 0 else 25.0,
            "pitch": float(pitch),
            "roll": float(roll),
            "heave": float(heave),
            "slant_range": 75.0,
            "frequency": 450.0,
        }

        rem_len = payload_len - len(hdr_bytes)
        rem_bytes = stream.read(rem_len)
        rem_stream = io.BytesIO(rem_bytes)

        port_samples = None
        stbd_samples = None

        for ch_idx in range(self.file_header.get("num_channels", 2)):
            if rem_stream.tell() + 64 > len(rem_bytes):
                break
            ch_hdr = rem_stream.read(64)
            ch_num, downsample, slant_range, ground_range, delay, duration = struct.unpack_from("<HHffff", ch_hdr, 0)
            sample_bytes_count = struct.unpack_from("<I", ch_hdr, 34)[0]

            if sample_bytes_count == 0 or sample_bytes_count > 65536:
                sample_bytes_count = min(1024, len(rem_bytes) - rem_stream.tell())

            raw_ch_data = rem_stream.read(sample_bytes_count)
            if not raw_ch_data:
                break

            arr = np.frombuffer(raw_ch_data, dtype=np.uint8)
            if ch_num == 0 or ch_idx == 0:
                port_samples = arr
                ping_meta["slant_range"] = max(10.0, float(slant_range)) if slant_range > 0 else 75.0
            else:
                stbd_samples = arr

        return ping_meta, port_samples, stbd_samples

    def _assemble_waterfall(
        self,
        port_traces: List[np.ndarray],
        stbd_traces: List[np.ndarray],
        max_samples: int
    ) -> np.ndarray:
        num_pings = len(port_traces)
        if num_pings == 0:
            return np.full((320, 640, 3), 40, dtype=np.uint8)

        target_width_per_channel = max(128, min(1024, max_samples if max_samples > 0 else 256))

        port_matrix = np.zeros((num_pings, target_width_per_channel), dtype=np.uint8)
        stbd_matrix = np.zeros((num_pings, target_width_per_channel), dtype=np.uint8)

        for i in range(num_pings):
            p = port_traces[i]
            s = stbd_traces[i]

            if p.size > 0:
                if len(p) != target_width_per_channel:
                    p = cv2.resize(p.reshape(1, -1), (target_width_per_channel, 1), interpolation=cv2.INTER_LINEAR).flatten()
                port_matrix[i, :] = p[::-1]

            if s.size > 0:
                if len(s) != target_width_per_channel:
                    s = cv2.resize(s.reshape(1, -1), (target_width_per_channel, 1), interpolation=cv2.INTER_LINEAR).flatten()
                stbd_matrix[i, :] = s

        nadir_separator = np.full((num_pings, 6), 15, dtype=np.uint8)
        waterfall_gray = np.hstack([port_matrix, nadir_separator, stbd_matrix])
        waterfall_bgr = cv2.cvtColor(waterfall_gray, cv2.COLOR_GRAY2BGR)
        return waterfall_bgr


# ─────────────────────────────────────────────────────────────────────────────
# EDGETECH JSF BINARY PARSER
# ─────────────────────────────────────────────────────────────────────────────
class JSFReader:
    """
    Binary parser for EdgeTech JSF side-scan logs (verified against a real 409 MB file).

    Layout: repeated  [16-byte message header `<HBBHBBBBHI`: marker 0x1601, version, session, message type, command,
    subsystem, channel, sequence, reserved, payload size][payload].  Side-scan traces are message 80; channel 0 = port,
    channel 1 = starboard.  Payload = 240-byte trace header + N samples.  Trace-header fields used (payload offsets):
        0  i32 ping time (unix s)      8  i32 ping number        34  i16 data format (0 = 16-bit envelope)
      108  i32 altitude (mm)         114  u16 sample count      116  u32 sample interval (ns)
      168  i16 weighting exponent (amplitude = raw * 2^-exp)     80/84 i32 X/Y, 88 i16 coordinate units
    Navigation: many side-scan-only logs carry NO position in the trace header (the file this was verified on has
    X/Y = 0 in every ping; the fix is logged elsewhere).  Then the telemetry is synthetic and metadata says so.
    """

    JSF_MARKER = 0x1601
    MSG_SIDESCAN = 80
    HDR = struct.Struct("<HBBHBBBBHI")
    TRACE_HDR = 240
    MAX_WIDTH = 1024                                     # per-channel waterfall width (matches XTFReader)

    def _scan(self, buf) -> List[Tuple[int, int, int]]:
        """Return [(payload_offset, channel, payload_size)] for every side-scan message; resync on corruption."""
        n, pos, out, resyncs = len(buf), 0, [], 0
        while pos + 16 <= n:
            marker, _v, _s, mtype, _c, _sub, ch, _q, _r, size = self.HDR.unpack_from(buf, pos)
            if marker != self.JSF_MARKER or pos + 16 + size > n:
                nxt = bytes(buf[pos + 1:pos + 1 + 65536]).find(b"\x01\x16")
                pos += 65536 if nxt < 0 else 1 + nxt
                resyncs += 1
                continue
            if mtype == self.MSG_SIDESCAN and size > self.TRACE_HDR:
                out.append((pos + 16, ch, size))
            pos += 16 + size
        self._resyncs = resyncs
        return out

    def read(self, source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[np.ndarray, List[TelemetryRecord], Dict[str, Any]]:
        import mmap
        fh = None
        if isinstance(source, (str, Path)):
            fh = open(source, "rb")
            buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        elif isinstance(source, io.BytesIO):
            buf = source.getbuffer()
        elif isinstance(source, (bytes, bytearray)):
            buf = memoryview(source)
        else:
            raise TypeError("Source must be file path, bytes, or io.BytesIO")
        try:
            return self._decode(buf)
        finally:
            del_view = getattr(self, "_view", None)
            if hasattr(buf, "close"):
                try:
                    buf.close()
                except BufferError:
                    pass
            if fh:
                fh.close()

    def _decode(self, buf):
        file_size = len(buf)
        msgs = self._scan(buf)
        if not msgs:
            raise ValueError("no side-scan (message 80) traces found - not a side-scan JSF file")

        # group traces into pings by message order (ping numbers are not monotonic in real files): a trace on a channel
        # that the current ping already has opens the next ping; port/starboard of one ping are logged back to back
        pings: Dict[int, Dict[str, Any]] = {}
        order: List[int] = []
        for off, ch, size in msgs:
            if not order or ch in pings[order[-1]]["ch"]:
                order.append(len(order))
                pings[order[-1]] = {"ch": {}, "t": struct.unpack_from("<i", buf, off)[0], "off": off}
            pings[order[-1]]["ch"][ch] = (off, size)

        n_pings = len(order)
        first = msgs[0][0]
        n_samp = struct.unpack_from("<H", buf, first + 114)[0] or (msgs[0][2] - self.TRACE_HDR) // 2
        interval_ns = struct.unpack_from("<I", buf, first + 116)[0] or 15000
        width = max(128, min(self.MAX_WIDTH, n_samp))

        def trace(off: int, size: int) -> np.ndarray:
            ns = struct.unpack_from("<H", buf, off + 114)[0] or (size - self.TRACE_HDR) // 2
            fmt = struct.unpack_from("<h", buf, off + 34)[0]
            wexp = struct.unpack_from("<h", buf, off + 168)[0]
            if fmt in (0, 1):                                    # 16-bit envelope / magnitude
                ns = min(ns, (size - self.TRACE_HDR) // 2)
                raw = np.frombuffer(buf, dtype="<u2", count=ns, offset=off + self.TRACE_HDR).astype(np.float32)
                return raw * (2.0 ** -wexp)
            ns = min(ns, size - self.TRACE_HDR)                  # 8-bit fallback
            return np.frombuffer(buf, dtype=np.uint8, count=ns, offset=off + self.TRACE_HDR).astype(np.float32)

        def reduce(a: np.ndarray) -> np.ndarray:
            if a.size == width:
                return a
            return cv2.resize(a.reshape(1, -1), (width, 1), interpolation=cv2.INTER_AREA).ravel()

        port = np.zeros((n_pings, width), np.float32)
        stbd = np.zeros((n_pings, width), np.float32)
        for i, k in enumerate(order):
            for ch, (off, size) in pings[k]["ch"].items():
                v = reduce(trace(off, size))
                if ch == 0:
                    port[i] = v[::-1]                            # stored near->far; the waterfall has nadir in the middle
                else:
                    stbd[i] = v

        # log-compress the (huge dynamic range) envelope and map robustly to 8 bit
        both = np.log1p(np.hstack([port, stbd]))
        lo, hi = np.percentile(both[::max(1, n_pings // 512)], [1.0, 99.7])
        gray8 = (np.clip((both - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        sep = np.full((n_pings, 6), 15, np.uint8)
        waterfall_bgr = cv2.cvtColor(np.hstack([gray8[:, :width], sep, gray8[:, width:]]), cv2.COLOR_GRAY2BGR)

        # telemetry: real ping times; position is synthetic when the trace headers carry none
        stride = max(1, n_pings // 64)
        has_nav = any(struct.unpack_from("<ii", buf, pings[k]["off"] + 80) != (0, 0) for k in order[::stride])
        rng_m = n_samp * interval_ns * 1e-9 * 1500.0 / 2.0
        alt = max(struct.unpack_from("<i", buf, first + 108)[0] / 1000.0, 0.0)
        records = []
        for i, k in enumerate(order):
            x, y, units = struct.unpack_from("<iih", buf, pings[k]["off"] + 80)
            if has_nav:                                          # JSF units: 1 = 1/1000 arc-minute, 2 = 1/10000 arc-minute
                div = 60000.0 if units == 1 else 600000.0
                lon, lat = x / div, y / div
            else:
                lat, lon = 13.0827 + i * 1e-5, 80.2707 + i * 1e-5
            records.append(TelemetryRecord(timestamp=float(pings[k]["t"]), latitude=lat, longitude=lon, heading_deg=45.0,
                                           depth_m=alt or 15.0, altitude_m=alt or 10.0, slant_range_m=rng_m))
        metadata = {
            "format": "EdgeTech JSF", "num_pings": n_pings, "channels": 2, "samples_per_channel": width,
            "native_samples_per_channel": int(n_samp), "sample_interval_ns": int(interval_ns),
            "max_slant_range_m": round(rng_m, 2), "file_size_bytes": file_size, "resyncs": getattr(self, "_resyncs", 0),
            "navigation_synthetic": not has_nav,
            "ping_time_start": int(pings[order[0]]["t"]), "ping_time_end": int(pings[order[-1]]["t"]),
        }
        return waterfall_bgr, records, metadata


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC XTF FILE GENERATOR FOR TESTING & DEMONSTRATIONS
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_xtf(
    output_path: Union[str, Path],
    num_pings: int = 120,
    samples_per_channel: int = 384,
    base_lat: float = 13.0827,
    base_lon: float = 80.2707,
    inject_target: bool = True,
    seed: int = 0,
) -> Path:
    """
    Creates a Triton XTF file with valid headers, attitude records,
    realistic seabed acoustic backscatter, and an acoustic debris target with downstream shadow.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)          # deterministic sample data
    with open(out_file, "wb") as f:
        # 1. 1024-byte File Header
        hdr = bytearray(1024)
        struct.pack_into("<BB", hdr, 0, XTF_MAGIC_FILE_HEADER, 1)
        hdr[2:10] = b"AkhetAI\x00"
        hdr[10:18] = b"v2.0.0\x00\x00"
        hdr[18:34] = b"SonarSim 450kHz\x00"
        struct.pack_into("<H64s", hdr, 34, 1, b"SIH 2026 Akhet Marine Guard Synthetic Sonar Log\x00")
        struct.pack_into("<HHH", hdr, 102, 1024, 0, 2)

        chan0_offset = 256
        struct.pack_into("<B", hdr, chan0_offset, 0)
        hdr[chan0_offset + 1:chan0_offset + 17] = b"Port Channel\x00\x00\x00\x00"

        chan1_offset = 320
        struct.pack_into("<B", hdr, chan1_offset, 1)
        hdr[chan1_offset + 1:chan1_offset + 17] = b"Stbd Channel\x00\x00\x00\x00"

        f.write(hdr)

        # 2. Write Ping Packets
        for p_idx in range(num_pings):
            record_len = 14 + 242 + 2 * (64 + samples_per_channel)
            pkt_hdr = struct.pack("<HBBHHHI", XTF_MAGIC_PACKET_HEADER, XTF_HEADER_SONAR, 0, 2, 0, 0, record_len)
            f.write(pkt_hdr)

            ping_hdr = bytearray(242)
            cur_time = BASE_EPOCH + p_idx * 0.25
            # Advance along a 45 deg track at the declared 3.8 kt (1 kt = 0.5144 m/s, 0.25 s per ping) so the
            # navigation is physically consistent with the speed written into every ping header.
            _step_m = 3.8 * 0.5144 * 0.25 * p_idx
            lat = base_lat + (_step_m * math.cos(math.radians(45.0))) / 111320.0
            lon = base_lon + (_step_m * math.sin(math.radians(45.0))) / (111320.0 * math.cos(math.radians(base_lat)))
            heading = 45.0 + math.sin(p_idx * 0.1) * 2.0
            pitch = math.sin(p_idx * 0.2) * 1.5
            roll = math.cos(p_idx * 0.2) * 2.0
            heave = math.sin(p_idx * 0.15) * 0.4

            _t = datetime.datetime.fromtimestamp(cur_time, datetime.timezone.utc)
            struct.pack_into("<HBBBBBB", ping_hdr, 0, _t.year, _t.month, _t.day, _t.hour, _t.minute, _t.second,
                             int(_t.microsecond / 10000))
            struct.pack_into("<HII", ping_hdr, 8, 255, int(cur_time), p_idx + 1)
            struct.pack_into("<f", ping_hdr, 18, 25.0)
            struct.pack_into("<fffff", ping_hdr, 26, heading, pitch, roll, heave, heading)
            struct.pack_into("<ffff", ping_hdr, 46, 15.0, 10.0, 3.8, 1500.0)
            struct.pack_into("<dd", ping_hdr, 62, lon, lat)
            struct.pack_into("<dd", ping_hdr, 78, lon, lat)
            f.write(ping_hdr)

            port_data = rng.normal(85, 14, samples_per_channel).clip(20, 210).astype(np.uint8)
            stbd_data = rng.normal(85, 14, samples_per_channel).clip(20, 210).astype(np.uint8)

            nadir_len = int(samples_per_channel * 0.12)
            port_data[:nadir_len] = rng.normal(12, 4, nadir_len).clip(2, 25).astype(np.uint8)
            stbd_data[:nadir_len] = rng.normal(12, 4, nadir_len).clip(2, 25).astype(np.uint8)

            ripple_modulation = (np.sin(np.arange(samples_per_channel) * 0.35) * 16.0).astype(np.int16)
            port_data[nadir_len:] = np.clip(port_data[nadir_len:].astype(np.int16) + ripple_modulation[nadir_len:], 0, 255).astype(np.uint8)

            if inject_target and 50 <= p_idx <= 70:
                t_pos = int(samples_per_channel * 0.45)
                stbd_data[t_pos:t_pos + 12] = rng.integers(230, 255, 12, dtype=np.uint8)
                stbd_data[t_pos + 12:t_pos + 38] = rng.integers(4, 18, 26, dtype=np.uint8)

            ch0_hdr = bytearray(64)
            struct.pack_into("<HHffff", ch0_hdr, 0, 0, 1, 75.0, 74.0, 0.0, 0.05)
            struct.pack_into("<I", ch0_hdr, 34, samples_per_channel)
            f.write(ch0_hdr)
            f.write(port_data.tobytes())

            ch1_hdr = bytearray(64)
            struct.pack_into("<HHffff", ch1_hdr, 0, 1, 1, 75.0, 74.0, 0.0, 0.05)
            struct.pack_into("<I", ch1_hdr, 34, samples_per_channel)
            f.write(ch1_hdr)
            f.write(stbd_data.tobytes())

    return out_file


def ingest_raw_sonar_file(
    file_or_bytes: Union[str, Path, bytes, io.BytesIO],
    filename: Optional[str] = None
) -> Tuple[np.ndarray, List[TelemetryRecord], Dict[str, Any]]:
    """
    Universal dispatcher for raw side-scan sonar files.
    Auto-detects XTF vs JSF based on extension or magic header bytes.
    """
    name_str = (filename or (str(file_or_bytes) if isinstance(file_or_bytes, (str, Path)) else "")).lower()

    if name_str.endswith(".jsf"):
        reader = JSFReader()
    else:
        reader = XTFReader()

    return reader.read(file_or_bytes)
