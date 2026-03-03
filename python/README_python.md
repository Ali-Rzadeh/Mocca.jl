# Mocca Python Port

This directory contains a complete Python implementation of the Mocca adsorption process workflow:

- Input parsing from Mocca JSON files
- Case and stage setup (pressurisation / adsorption / blowdown / evacuation)
- Time marching simulator with adsorption kinetics and thermal coupling
- CSV export and plotting utilities

## Install

```bash
pip install -e .
```

## Quick start

```bash
python python/examples/quick_start.py
```
