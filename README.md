# Simtools

A Python utility for calculating production profits in Sim Companies using the Simcotools API.

## Features

- **Real-time Data**: Fetches the latest resource data and market VWAPs (Volume Weighted Average Prices) from the Simcotools API.
- **Profit Calculation**: Calculates hourly profit by accounting for:
  - Selling price (at specific quality levels).
  - 4% market exchange fee.
  - Production wages.
  - Administrative Overhead costs.
  - Input material costs.
  - Transportation costs.
- **Data Persistence**: Automatically saves fetched API data to `resources.json` and `vwaps.json` for inspection.
- **Search & Filter**: Search for specific resources or filter calculations by a target quality level.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1. Clone the repository.
2. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

Run the script using `uv run` or directly with `python` if dependencies are installed.

```bash
uv run main.py [options]
```

### Options

- `-Q`, `--quality`: Specify the quality level to use for market prices (default: `0`).
- `-S`, `--search`: Search for a specific resource by name (case-insensitive).
- `-A`, `--abundance`: Specify the abundance percentage (0-100) for mine and well resources (default: `90`).
- `-O`, `--admin-overhead`: Specify administration overhead percentage to add to wages (default: `0`).

### Examples

**View top 30 most profitable quality 0 resources with 85% abundance:**
```bash
uv run main.py -A 85
```

**Calculate profits for Quality 2 resources:**
```bash
uv run main.py -Q 2
```

**Search for "Electric" related resources at Quality 1:**
```bash
uv run main.py -S Electric -Q 1
```

## Output Explanation

The tool displays a table with the following columns:
- **Resource**: Name of the produced item. An `(*)` indicates it is an abundance-based resource (Mine/Well).
- **Profit/hr**: Net profit per hour after all costs (including abundance adjustments and market fee).
- **Revenue/hr**: Gross income per hour from sales.
- **Fee/hr**: 4% market exchange fee.
- **Costs/hr**: Combined cost of wages (including admin overhead) and input materials.
- **Transp/hr**: Total transportation costs.

### Warning Flags
- `(*)`: Indicates abundance-based resource (e.g., Bauxite, Crude oil, etc.). The production rate is multiplied by the abundance percentage.
- `(!)`: Indicates that one or more input materials for that resource do not have a market price at the specified quality level, which may result in inaccurate profit calculations.

## Data Files

- `resources.json`: Contains raw metadata for all resources (wages, inputs, production rates).
- `vwaps.json`: Contains the latest market Volume Weighted Average Prices for all resources and qualities.
- `abundance_resources.json`: List of resources that use the abundance calculation.

