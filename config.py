"""
config.py
Shared configuration and binary schema definitions.
Optimized for Python 3.14t (Free-Threaded).
"""
import struct

# --- DATA STORAGE ---
BASE_DIR = "data"
SYMBOL = "BTCUSDT"

# --- BINARY SCHEMA (AGG2) ---
# Row: <trade_id, px, qty, first_id, count, flags, ts_ms, side, padding>
# 48 bytes per row
AGG_ROW_STRUCT = struct.Struct("<QQQQHHqB3x")
AGG_ROW_SIZE = AGG_ROW_STRUCT.size

# Header: Magic + Version + Day + Reserved + Count + MinTS + MaxTS
AGG_HDR_STRUCT = struct.Struct("<4sBBHQqq16x")
AGG_HDR_SIZE = AGG_HDR_STRUCT.size
AGG_HDR_MAGIC = b"AGG2"

# Index: Day + Offset + Length
INDEX_ROW_STRUCT = struct.Struct("<HQQ")
INDEX_ROW_SIZE = INDEX_ROW_STRUCT.size

# --- SCALING ---
PX_SCALE = 100_000_000.0
QT_SCALE = 100_000_000.0

# --- SIMULATION ---
TAKER_FEE_BPS = 4.0
SLIPPAGE_BPS = 1.0
COST_BASIS_BPS = TAKER_FEE_BPS + SLIPPAGE_BPS

# --- HARDWARE OPTIMIZATION ---
# Ryzen 9 7900X (24 Logical Threads)
WORKERS = 24