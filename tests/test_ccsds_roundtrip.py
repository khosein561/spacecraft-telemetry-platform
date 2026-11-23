"""Packet codec: round trips, the 'minus one' length rule, CRC, and never crashing on garbage."""
import random
import struct

import pytest
from hypothesis import given, settings, strategies as st

from telemetry.packets import ccsds
from telemetry.packets.ccsds import (
    PacketCRCError, PacketError, crc16_ccitt, decode_packet, encode_packet, pack_readings, unpack_readings,
)

READING = st.tuples(st.integers(0, 0xFFFF), st.floats(width=32, allow_nan=False, allow_infinity=False))


def test_roundtrip():
    buf = encode_packet(apid=17, seq_count=1234, time_us=1_700_000_000_000_000, readings=[(3, 0.5), (4, -1.25)])
    pkt = decode_packet(buf)
    assert (pkt.apid, pkt.seq_count, pkt.time_us) == (17, 1234, 1_700_000_000_000_000)
    assert pkt.readings == ((3, 0.5), (4, -1.25))
    assert pkt.header.ptype == 0 and pkt.header.sec_hdr_flag == 1 and pkt.header.seq_flags == 0b11


@given(
    apid=st.integers(0, 2047), seq=st.integers(0, 16383), t=st.integers(0, 2**64 - 1),
    readings=st.lists(READING, max_size=20),
)
def test_roundtrip_property(apid, seq, t, readings):
    pkt = decode_packet(encode_packet(apid, seq, t, readings))
    assert (pkt.apid, pkt.seq_count, pkt.time_us) == (apid, seq, t)
    assert list(pkt.readings) == readings


def test_length_field_is_data_bytes_minus_one():
    # data field = 8 (time) + 6*N (readings) + 2 (CRC). For N=0 that is 10 bytes -> field stores 9.
    buf = encode_packet(1, 0, 0, [])
    assert len(buf) == 6 + 10
    assert struct.unpack(">H", buf[4:6])[0] == 9
    buf2 = encode_packet(1, 0, 0, [(0, 1.0)])
    assert struct.unpack(">H", buf2[4:6])[0] == 15


def test_header_bit_layout_by_hand():
    buf = encode_packet(apid=0x555, seq_count=0x2AAA, time_us=0, readings=[])
    word0, word1, _ = struct.unpack(">HHH", buf[:6])
    assert word0 == 0b000_0_1_10101010101   # version 0, type 0, sec-hdr 1, apid 0x555
    assert word1 == 0b11_10101010101010     # flags 11, count 0x2AAA


def test_crc_known_vector():
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_short_buffer_raises():
    for n in range(0, 16):
        with pytest.raises(ValueError):
            decode_packet(b"\x00" * n)


def test_length_field_that_lies_is_rejected_both_ways():
    good = encode_packet(5, 1, 0, [(0, 1.0)])
    with pytest.raises(PacketError):
        decode_packet(good + b"\x00")          # trailing junk
    with pytest.raises(PacketError):
        decode_packet(good[:-1])               # truncated
    lying = bytearray(good)
    lying[5] = 0xFF                            # claims 256 data bytes
    with pytest.raises(PacketError):
        decode_packet(bytes(lying))


def test_corruption_is_caught_by_crc_and_reports_the_header():
    good = bytearray(encode_packet(9, 77, 5, [(1, 2.0)]))
    good[12] ^= 0x01                           # flip one payload bit
    with pytest.raises(PacketCRCError) as info:
        decode_packet(bytes(good))
    assert (info.value.apid, info.value.seq_count) == (9, 77)
    assert decode_packet(bytes(good), verify_crc=False).apid == 9


def test_field_validation_on_encode():
    for kwargs in (dict(apid=2048), dict(apid=-1), dict(seq_count=16384), dict(seq_count=-1), dict(ptype=2)):
        args = dict(apid=1, seq_count=0, time_us=0, readings=[])
        args.update(kwargs)
        with pytest.raises(ValueError):
            encode_packet(**args)
    with pytest.raises(ValueError):
        pack_readings([(70000, 1.0)])


def test_unsupported_version_rejected():
    buf = bytearray(encode_packet(1, 0, 0, []))
    buf[0] |= 0b0010_0000                      # set a version bit
    with pytest.raises(PacketError):
        decode_packet(bytes(buf), verify_crc=False)


def test_unpack_requires_whole_readings():
    assert unpack_readings(pack_readings([(1, 0.25)])) == [(1, 0.25)]
    with pytest.raises(PacketError):
        unpack_readings(b"\x00" * 7)


def test_fuzz_never_crashes_seeded():
    """1000 random byte strings either decode or raise ValueError - nothing else escapes."""
    rng = random.Random(1234)
    for _ in range(1000):
        blob = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 40)))
        try:
            decode_packet(blob)
        except ValueError:
            pass


@settings(max_examples=300)
@given(st.binary(max_size=80))
def test_fuzz_arbitrary_bytes(blob):
    for verify in (True, False):
        try:
            decode_packet(blob, verify_crc=verify)
        except ValueError:
            pass


@settings(max_examples=200)
@given(apid=st.integers(0, 2047), seq=st.integers(0, 16383), readings=st.lists(READING, max_size=5), pos=st.integers(0, 10_000), bit=st.integers(0, 7))
def test_any_single_bit_flip_never_decodes_silently_wrong(apid, seq, readings, pos, bit):
    """Flip any one bit of a valid packet: decoding either raises ValueError or (rarely, e.g. a flipped
    reserved header bit the CRC also covers -> still caught) never returns different content."""
    original = encode_packet(apid, seq, 42, readings)
    corrupted = bytearray(original)
    corrupted[pos % len(corrupted)] ^= 1 << bit
    try:
        pkt = decode_packet(bytes(corrupted))
    except ValueError:
        return
    assert pkt == decode_packet(original)      # CRC-16 detects all single-bit errors, so this is unreachable
