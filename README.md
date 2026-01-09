# Simtools

A Python utility for calculating production profits in Sim Companies using the Simcotools API.

## Features

- **Rich UI**: Uses the `rich` library for beautiful tables, colors, progress indicators, and formatted logs.
- **Real-time Data**: Fetches the latest resource data and market VWAPs (Volume Weighted Average Prices) from the Simcotools API.
- **Profit Calculation**: Calculates hourly profit by accounting for:
  - Selling price (at specific quality levels).
  - 4% market exchange fee (0% in direct contract mode).
  - Production wages.
  - Administrative Overhead costs.
  - Input material costs.
  - Transportation costs (50% reduction in direct contract mode).
- **Data Persistence**: Automatically saves fetched API data to `resources.json` and `vwaps.json` for inspection.
- **Search & Filter**: Search for specific resources or filter calculations by a target quality level.
- **Direct Contract Mode**: Support for calculating profits when trading directly with other players.

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
- `-S`, `--search`: Search for specific resources by name (case-insensitive). You can provide multiple terms (e.g., `-S power water`).
- `-B`, `--building`: Filter resources by building name. This also displays the construction costs for the matched building(s).
- `-A`, `--abundance`: Specify the abundance percentage (0-100) for mine and well resources (default: `90`).
- `-O`, `--admin-overhead`: Specify administration overhead percentage to add to wages (default: `0`).
- `-C`, `--contract`: Calculate values for direct contracts (0% market fee, 50% transportation cost).
- `-D`, `--debug-unassigned`: List all resources that are not assigned to any building in `buildings.json`.

### Examples

**View top 30 most profitable quality 0 resources with 85% abundance:**
```bash
uv run main.py -A 85
```

**Calculate profits for Quality 2 resources:**
```bash
uv run main.py -Q 2
```

**Search for "Electric" and "Water" related resources at Quality 1:**
```bash
uv run main.py -S Electric Water -Q 1
```

**Filter by building (e.g., Farm) to see its costs and production profits:**
```bash
uv run main.py -B Farm
```

**Calculate direct contract profits (0% fee, 50% transport):**
```bash
uv run main.py --contract
```

## Output Explanation

The tool displays a formatted table using `rich` with the following columns:
- **Resource**: Name of the produced item (**bold white**). A yellow `(*)` indicates it is an abundance-based resource (Mine/Well).
- **Profit/hr**: Net profit per hour. Highlighted in **green** if positive and **red** if negative.
- **Revenue/hr**: Gross income per hour from sales (**white**).
- **Fee/hr**: 4% market exchange fee (**red**).
- **Costs/hr**: Combined cost of wages (including admin overhead) and input materials (**yellow**).
- **Transp/hr**: Total transportation costs (**magenta**). A red `(!)` indicates missing input prices.

### Indicators & Warnings
- **Yellow (*)**: Indicates abundance-based resource (e.g., Bauxite, Crude oil, etc.). The production rate is multiplied by the abundance percentage.
- **Red (!)**: Indicates that one or more input materials for that resource do not have a market price at the specified quality level, which may result in inaccurate profit calculations.

## Data Files

- `resources.json`: Contains raw metadata for all resources (wages, inputs, production rates).
- `vwaps.json`: Contains the latest market Volume Weighted Average Prices for all resources and qualities.
- `abundance_resources.json`: List of resources that use the abundance calculation.
- `buildings.json`: Contains metadata for buildings, including construction costs and the resources they produce.

