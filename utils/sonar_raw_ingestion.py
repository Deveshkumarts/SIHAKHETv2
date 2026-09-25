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
    Binary Parser for EdgeTech Sonar Format (.jsf).
    Reads 16-byte message envelopes and extracts Message 2080 (Side-Scan Acoustic Data).
    """

    JSF_MARKER = 0x1601
    MSG_SIDESCAN = 2080

    def read(self, source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[np.ndarray, List[TelemetryRecord], Dict[str, Any]]:
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

        port_lines = []
        stbd_lines = []
        records = []

        while stream.tell() + 16 <= file_size:
            pos = stream.tell()
            marker, ver, session, msg_type, cmd, subsys, ch, seq, rsvd, data_size = struct.unpack(
                "<HBBHBBBHI", stream.read(16)
            )

            if marker != self.JSF_MARKER or pos + 16 + data_size > file_size:
                stream.seek(pos + 1)
                continue

            if msg_type == self.MSG_SIDESCAN and data_size >= 64:
                sub_bytes = stream.read(data_size)
                sample_count = struct.unpack_from("<H", sub_bytes, 10)[0]
                if sample_count > 0:
                    raw_samples = sub_bytes[240:240 + sample_count]
                    if raw_samples:
                        arr = np.frombuffer(raw_samples, dtype=np.uint8)
                        if ch == 0:
                            port_lines.append(arr)
                        else:
                            stbd_lines.append(arr)

                        if len(records) < len(port_lines):
                            records.append(TelemetryRecord(
                                timestamp=time.time(),
                                latitude=13.0827 + len(records) * 0.0001,
                                longitude=80.2707 + len(records) * 0.0001,
                                heading_deg=45.0,
                                depth_m=15.0,
                                altitude_m=10.0,
                                slant_range_m=75.0,
                            ))
            else:
                stream.seek(pos + 16 + data_size)

        max_len = max([len(p) for p in port_lines] + [len(s) for s in stbd_lines] + [256])
        xtf_reader = XTFReader()
        waterfall_bgr = xtf_reader._assemble_waterfall(port_lines, stbd_lines, max_len)

        metadata = {
            "format": "EdgeTech JSF",
            "num_pings": max(len(port_lines), len(stbd_lines)),
            "channels": 2,
            "samples_per_channel": max_len,
            "file_size_bytes": file_size,
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
