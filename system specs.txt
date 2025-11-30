Here is the simplified README. It assumes your environment is set up and you just need the standard commands.

***

# Adaptive Microstructure Alpha (v4.1)

## Setup
Ensure your Python environment is ready.

## Workflow

**1. Download Data**
```bash
python data.py
```

**2. Verify Data**
Checks integrity and scale (integers `1e8`).
```bash
python verify.py
```

**3. Quick Test**
Runs a small batch to check the code.
```bash
python quick.py
```

**4. Run Backtest**
Executes the full strategy.
```bash
python backtest.py
```