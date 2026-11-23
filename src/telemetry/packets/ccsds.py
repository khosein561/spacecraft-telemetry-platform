"""
CCSDS-style space packet encoding and decoding.

WHY THIS FILE EXISTS: ground-segment software spends its life turning bytes from a spacecraft
into numbers. This is that translation layer. Everything downstream (broker, consumer,
database) depends on it being exactly right and on it NEVER crashing when a link delivers garbage.

Wire format (all big-endian, as in CCSDS):

    primary header, 6 bytes
        bits 0-2    version               (always 0)
        bit  3      type                  (0 = telemetry, 1 = telecommand)
        bit  4      secondary-header flag (we always set 1)
        bits 5-15   APID                  (11 bits; 2047 = idle packet)
        bits 16-17  sequence flags        (3 = unsegmented, the only kind we use)
        bits 18-31  sequence count        (14 bits; wraps 16383 -> 0)
        bits 32-47  data length           (number of bytes in the data field, MINUS ONE)
    data field
        secondary header, 8 bytes: spacecraft time, unsigned microseconds since the Unix epoch
        user data: N readings, each  u16 channel index + f32 value  (6 bytes)
        packet error control, 2 bytes: CRC-16/CCITT-FALSE over everything before it

The "minus one" length rule is the classic trap: a data field of 10 bytes encodes the number 9.
The CRC is part of the data field, so it is counted in the length.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

PRIMARY_HEADER_LEN = 6
SECONDARY_HEADER_LEN = 8
CRC_LEN = 2
READING_LEN = 6
MAX_SEQ = 0x3FFF  # 14 bits
SEQ_MODULO = MAX_SEQ + 1  # 16384
MAX_APID = 0x7FF  # 11 bits
IDLE_APID = MAX_APID  # 2047
SEQ_FLAGS_UNSEGMENTED = 0b11
MAX_DATA_LEN = 0xFFFF + 1  # a 16-bit "length minus one" field can describe 65536 bytes

_PRIMARY = struct.Struct(">HHH")
_TIME = struct.Struct(">Q")
_READING = struct.Struct(">Hf")


class PacketError(ValueError):
    """The bytes are not a valid packet. Subclasses ValueError so callers can catch one type."""


class PacketCRCError(PacketError):
    """Structure was fine but the checksum failed: the packet was corrupted in transit.

    The parsed header is attached so the caller can log WHICH packet was damaged. It must not be
    trusted for anything else (the damage may be in the header itself).
    """

    def __init__(self, message: str, apid: int, seq_count: int):
        super().__init__(message)
        self.apid = apid
        self.seq_count = seq_count


@dataclass(frozen=True)
class PacketHeader:
    """Parsed primary header fields."""

    version: int
    ptype: int
    sec_hdr_flag: int
    apid: int
    seq_flags: int
    seq_count: int
    data_length: int  # as stored on the wire: actual data bytes minus one


@dataclass(frozen=True)
class Packet:
    """A decoded packet."""

    header: PacketHeader
    time_us: int  # spacecraft time from the secondary header
    readings: tuple  # tuple of (channel_index, value)

    @property
    def apid(self) -> int:
        return self.header.apid

    @property
    def seq_count(self) -> int:
        return self.header.seq_count


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF). Check value for b"123456789" is 0x29B1."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def pack_readings(readings) -> bytes:
    """Serialize (channel_index, value) pairs: u16 index + f32 value each, big-endian."""
    out = bytearray()
    for index, value in readings:
        if not 0 <= index <= 0xFFFF:
            raise ValueError(f"channel index out of range: {index}")
        out += _READING.pack(index, value)
    return bytes(out)


def unpack_readings(payload: bytes) -> list:
    """Inverse of pack_readings. The payload length must be a whole number of readings."""
    if len(payload) % READING_LEN != 0:
        raise PacketError(f"reading block of {len(payload)} bytes is not a multiple of {READING_LEN}")
    return [_READING.unpack_from(payload, off) for off in range(0, len(payload), READING_LEN)]


def encode_packet(apid: int, seq_count: int, time_us: int, readings, ptype: int = 0) -> bytes:
    """Build a complete packet: primary header + secondary header + readings + CRC."""
    if not 0 <= apid <= MAX_APID:
        raise ValueError(f"apid must be 0..{MAX_APID}, got {apid}")
    if not 0 <= seq_count <= MAX_SEQ:
        raise ValueError(f"seq_count must be 0..{MAX_SEQ}, got {seq_count}")
    if ptype not in (0, 1):
        raise ValueError("ptype must be 0 or 1")
    if not 0 <= time_us < 2**64:
        raise ValueError("time_us must fit in 64 bits")
    body = _TIME.pack(time_us) + pack_readings(readings)
    data_len = len(body) + CRC_LEN
    if data_len > MAX_DATA_LEN:
        raise ValueError("too many readings for one packet")
    word0 = (0 << 13) | (ptype << 12) | (1 << 11) | apid  # version 0, secondary header present
    word1 = (SEQ_FLAGS_UNSEGMENTED << 14) | seq_count
    head = _PRIMARY.pack(word0, word1, data_len - 1)  # the "minus one" rule
    without_crc = head + body
    return without_crc + struct.pack(">H", crc16_ccitt(without_crc))


def parse_header(buf: bytes) -> PacketHeader:
    """Parse just the 6-byte primary header. Raises PacketError if the buffer is too short."""
    if len(buf) < PRIMARY_HEADER_LEN:
        raise PacketError(f"buffer of {len(buf)} bytes is shorter than the {PRIMARY_HEADER_LEN}-byte header")
    word0, word1, length = _PRIMARY.unpack_from(buf)
    return PacketHeader(
        version=word0 >> 13,
        ptype=(word0 >> 12) & 1,
        sec_hdr_flag=(word0 >> 11) & 1,
        apid=word0 & MAX_APID,
        seq_flags=word1 >> 14,
        seq_count=word1 & MAX_SEQ,
        data_length=length,
    )


def decode_packet(buf: bytes, verify_crc: bool = True) -> Packet:
    """Parse bytes into a Packet.

    Raises PacketError (a ValueError) for anything malformed and PacketCRCError for a checksum
    failure. It never trusts the length field: the buffer must be EXACTLY header + length + 1
    bytes, so a lying length cannot make us read past the end or accept trailing junk.
    """
    header = parse_header(buf)
    if header.version != 0:
        raise PacketError(f"unsupported packet version {header.version}")
    if header.sec_hdr_flag != 1:
        raise PacketError("secondary header flag not set (this project always sends one)")
    if header.seq_flags != SEQ_FLAGS_UNSEGMENTED:
        raise PacketError(f"unsupported sequence flags {header.seq_flags}")
    data_len = header.data_length + 1
    if len(buf) != PRIMARY_HEADER_LEN + data_len:
        raise PacketError(
            f"length field says {data_len} data bytes but buffer holds {len(buf) - PRIMARY_HEADER_LEN}"
        )
    if data_len < SECONDARY_HEADER_LEN + CRC_LEN:
        raise PacketError("data field too short to hold a timestamp and checksum")
    if verify_crc:
        (stored,) = struct.unpack_from(">H", buf, len(buf) - CRC_LEN)
        if crc16_ccitt(buf[:-CRC_LEN]) != stored:
            raise PacketCRCError("checksum mismatch", header.apid, header.seq_count)
    (time_us,) = _TIME.unpack_from(buf, PRIMARY_HEADER_LEN)
    payload = buf[PRIMARY_HEADER_LEN + SECONDARY_HEADER_LEN : len(buf) - CRC_LEN]
    return Packet(header=header, time_us=time_us, readings=tuple(unpack_readings(payload)))
