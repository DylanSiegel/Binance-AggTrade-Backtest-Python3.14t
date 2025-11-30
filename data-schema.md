## 1. Filesystem / Dataset Layout

Per symbol `SYMBOL` (e.g. `"BTCUSDT"`), data live under:

```text
{BASE_DIR}/{SYMBOL}/{YYYY}/{MM}/
    data.quantdev   # concatenated compressed day blobs
    index.quantdev  # index for day blobs
```

Defaults in code:

* `BASE_DIR = "data"` (via `CONFIG["BASE_DIR"]`)
* `SYMBOL` = `CONFIG["SYMBOL"]`
* `YYYY` = 4-digit year, zero-padded (e.g. `2024`)
* `MM`   = 2-digit month, zero-padded (e.g. `01`)

Example for BTCUSDT, January 2024:

```text
data/BTCUSDT/2024/01/data.quantdev
data/BTCUSDT/2024/01/index.quantdev
```

Each *calendar day* (per symbol) corresponds to one compressed “AGG2” block in `data.quantdev`, with a corresponding index row in `index.quantdev`.

---

## 2. Index File (`index.quantdev`)

Binary file, sequence of fixed-size records:

```text
STRUCT: "<HQQ"  (little-endian)
SIZE  : 18 bytes
```

Fields per record:

| Offset | Type | Name         | Description                                  |
| ------ | ---- | ------------ | -------------------------------------------- |
| 0      | H    | day_of_month | Day of month (1–31)                          |
| 2      | Q    | offset       | Byte offset into `data.quantdev`             |
| 10     | Q    | length       | Compressed blob length in bytes for that day |

Usage:

1. Open `index.quantdev`.

2. Iterate in 18-byte chunks; for each chunk `c`:

   ```python
   day_of_month, offset, length = unpack("<HQQ", c)
   ```

3. For a given `(year, month, day)` you want:

   * Find a record with `day_of_month == day`.
   * Ensure `offset + length <= filesize(data.quantdev)` (otherwise ignore as invalid).
   * Use `(offset, length)` on `data.quantdev` to locate the compressed day blob.

A day is treated as “indexed/valid” only if there is at least one index row for that day with `offset + length <= size(data.quantdev)`.

---

## 3. Data File (`data.quantdev`)

Binary file containing concatenated Zstandard frames for each indexed day:

```text
data.quantdev = zstd_frame(day1) || zstd_frame(day2) || ... || zstd_frame(dayN)
```

For a given index row `(day_of_month, offset, length)`:

1. Open `data.quantdev`.

2. Seek to `offset`, read `length` bytes → `compressed_blob`.

3. Decompress:

   ```python
   raw_blob = zstd.decompress(compressed_blob)
   ```

4. `raw_blob` format:

   ```text
   [ AGG2 header ][ N × AGG2 row ]
   ```

---

## 4. AGG2 Day Blob Format (Uncompressed)

### 4.1 Header

Struct:

```text
AGG_HDR_STRUCT = "<4sBBHQqq16x"   # little-endian
HEADER_SIZE    = 48 bytes
```

Fields:

| Offset | Type | Name         | Description                                            |
| ------ | ---- | ------------ | ------------------------------------------------------ |
| 0      | 4s   | magic        | Must be `b"AGG2"`                                      |
| 4      | B    | version      | Format version (`1`)                                   |
| 5      | B    | day_of_month | Day of month for this block (1–31)                     |
| 6      | H    | reserved     | Reserved, currently `0`                                |
| 8      | Q    | row_count    | Number of rows in this block                           |
| 16     | q    | ts_min_ms    | Minimum timestamp (ms since Unix epoch, signed 64-bit) |
| 24     | q    | ts_max_ms    | Maximum timestamp (ms since Unix epoch, signed 64-bit) |
| 32     | 16x  | padding      | 16 bytes reserved (ignore)                             |

Interpretation:

* Header is always 48 bytes.
* `row_count` is the number of fixed-size rows that immediately follow the header.
* All timestamps are Unix milliseconds.

### 4.2 Row

Struct:

```text
AGG_ROW_STRUCT = "<QQQQHHqB3x"   # little-endian
ROW_SIZE       = 48 bytes
```

Fields:

| Offset | Type | Name     | Meaning                                                                                  |
| ------ | ---- | -------- | ---------------------------------------------------------------------------------------- |
| 0      | Q    | trade_id | Aggregate trade ID (`aggTradeId` from Binance)                                           |
| 8      | Q    | px       | Scaled price                                                                             |
| 16     | Q    | qty      | Scaled quantity                                                                          |
| 24     | Q    | first_id | First underlying trade ID in this aggregate (`firstTradeId`)                             |
| 32     | H    | count    | Number of underlying trades (≈ `lastTradeId - firstTradeId + 1`, clamped to `[0,65535]`) |
| 34     | H    | flags    | Bit flags (currently only bit 0 used: `FLAG_IS_BUYER_MAKER`)                             |
| 36     | q    | ts_ms    | Event timestamp in milliseconds since Unix epoch (signed 64-bit)                         |
| 44     | B    | side     | 0 if `isBuyerMaker` is true (maker), 1 otherwise (taker)                                 |
| 45     | 3x   | padding  | Padding for alignment / future use                                                       |

#### Scaling for Price / Quantity

Constants:

```text
PX_SCALE = 100_000_000
QT_SCALE = 100_000_000
```

Encoding:

```python
px_scaled = int(float(price)    * PX_SCALE)
qty_scaled = int(float(quantity) * QT_SCALE)
```

Decoding:

* `price    = px / PX_SCALE`
* `quantity = qty / QT_SCALE`

#### Trade Count

From source:

```python
fi = firstTradeId
li = lastTradeId
cnt = li - fi + 1
if cnt < 0:        # row dropped before encoding
if cnt > 65535:    cnt = 65535
```

* `count` is `uint16`.
* If `count < 65535`, then `lastTradeId = first_id + count - 1` exactly.
* If `count == 65535`, `lastTradeId` is **at least** `first_id + 65535 - 1` (range was truncated).

#### Flags and Side

Constants and logic:

```python
FLAG_IS_BUYER_MAKER = 1 << 0
is_maker = isBuyerMaker  # from CSV

flags = FLAG_IS_BUYER_MAKER if is_maker else 0
side  = 0 if is_maker else 1
```

Interpretation:

* `flags & FLAG_IS_BUYER_MAKER`:

  * non-zero → `isBuyerMaker == True`
  * zero     → `isBuyerMaker == False`

* `side`:

  * `0` → maker side (`isBuyerMaker` true)
  * `1` → taker side (`isBuyerMaker` false)

---

## 5. Logical Source Schema (Binance `aggTrades`)

After header normalization, CSV columns map to AGG2 fields approximately as:

* `aggTradeId` or `id`    → `trade_id`
* `price`                 → price (scaled to `px`)
* `quantity`              → quantity (scaled to `qty`)
* `firstTradeId`          → `first_id`
* `lastTradeId`           → used to compute `count`
* `transactTime`          → `ts_ms` (and contributes to `ts_min_ms` / `ts_max_ms`)
* `isBuyerMaker`          → `flags` and `side`

Logical per-row schema (before packing):

* `trade_id: int64`
* `price: float`      (stored as scaled `uint64`)
* `quantity: float`   (stored as scaled `uint64`)
* `first_trade_id: int64`
* `last_trade_id: int64`
* `trade_count: int` (derived, clamped to `[0, 65535]`)
* `timestamp_ms: int64`
* `is_buyer_maker: bool`

---

## 6. Minimal “How to Read One Day”

Given: `BASE_DIR`, `SYMBOL`, `YEAR`, `MONTH`, `DAY`.

1. Paths:

   ```text
   dir       = f"{BASE_DIR}/{SYMBOL}/{YEAR:04d}/{MONTH:02d}"
   data_path = dir + "/data.quantdev"
   index_path= dir + "/index.quantdev"
   ```

2. Find index row:

   * Open `index_path` in binary.
   * For each 18-byte chunk:

     ```python
     day_of_month, offset, length = unpack("<HQQ", chunk)
     if day_of_month == DAY and offset + length <= size(data_path):
         # use this (offset, length)
     ```

3. Read + decompress:

   * Open `data_path`.
   * `seek(offset)`, `read(length)` → `compressed_blob`.
   * `raw = zstd.decompress(compressed_blob)`.

4. Parse header:

   ```python
   hdr = raw[:48]
   magic, ver, h_day, _, row_count, ts_min, ts_max = unpack("<4sBBHQqq16x", hdr)
   ```

   * Check: `magic == b"AGG2"`, `ver == 1`, `h_day == DAY`.

5. Parse rows:

   ```python
   rows = raw[48:]
   assert len(rows) == row_count * 48

   for i in range(row_count):
       sl = rows[i*48:(i+1)*48]
       trade_id, px_s, qty_s, first_id, count, flags, ts_ms, side = unpack("<QQQQHHqB3x", sl)

       price = px_s / 1e8
       qty   = qty_s / 1e8
       is_buyer_maker = bool(flags & 1)
   ```