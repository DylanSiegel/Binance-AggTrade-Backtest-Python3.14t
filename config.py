"""
config.py
Shared Configuration & Schemas.
"""
import struct

# --- PATHS ---
BASE_DIR = "data"
SYMBOL = "BTCUSDT"

# --- RAW DATA (AGG3) - Input for Builder ---
# Row: <trade_id, px, qty, fi, cnt, flags, ts_ms, padding>
AGG_ROW_STRUCT = struct.Struct("<QQQQIHq2x")
AGG_HDR_STRUCT = struct.Struct("<4sBBHQqq16x")
AGG_HDR_SIZE = 48
AGG_ROW_SIZE = 48
PX_SCALE = 100_000_000.0
QT_SCALE = 100_000_000.0

# --- INDEX SCHEMA (V1) ---
INDEX_MAGIC = b"QIDX"
INDEX_VERSION = 1
INDEX_HDR_STRUCT = struct.Struct("<4sIQ")
INDEX_HDR_SIZE = INDEX_HDR_STRUCT.size
INDEX_ROW_STRUCT = struct.Struct("<HQQQ")
INDEX_ROW_SIZE = INDEX_ROW_STRUCT.size

# --- NOVEL BARS (OUTPUT of Builder) ---
# Format:
# 1. ts_start (d): Unix timestamp float
# 2. ts_end   (d): Unix timestamp float
# 3. open     (d): Price
# 4. high     (d): Price
# 5. low      (d): Price
# 6. close    (d): Price
# 7. vol      (d): Total Volume
# 8. delta    (d): Net Delta (Buy - Sell)
# 9. effic    (d): Efficiency Ratio (0.0 to 1.0)
# 10. impact  (d): Price Impact (Price Change / Delta)
BAR_STRUCT = struct.Struct("<dddddddddd")
BAR_SIZE = BAR_STRUCT.size  # 80 bytes per bar

# --- SETTINGS ---
DELTA_THRESHOLD = 50.0  # BTC Net Delta to trigger a new bar
WORKERS = 24            # Ryzen 7900X

# Simulation
TAKER_FEE_BPS = 4.0
SLIPPAGE_BPS = 1.0
COST_BASIS_BPS = TAKER_FEE_BPS + SLIPPAGE_BPS