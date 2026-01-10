# Simtools

A Python utility for calculating production profits in Sim Companies using the Simcotools API.

## Features

- **Rich UI**: Uses the `rich` library for beautiful tables, colors, progress indicators, and formatted logs.
- **Real-time Data**: Fetches the latest resource data and market VWAPs (Volume Weighted Average Prices) from the Simcotools API.
- **Profit Calculation**: Calculates hourly profit by accounting for:
  - Selling price (at specific quality levels).
  - 4% market exchange fee (0% in direct contract mode).
  - Production wages (with optional 3% reduction for robots).
  - Administrative Overhead costs.
  - Input material costs.
  - Transportation costs (50% reduction in direct contract mode).
- **Data Persistence**: Automatically saves fetched API data to `resources.json` and `vwaps.json` for inspection.
- **Search & Filter**: Search for specific resources or filter calculations by a target quality level.
- **Direct Contract Mode**: Support for calculating profits when trading directly with other players.
- **Prospecting Simulation**: Simulate the probability and expected time/attempts to reach a target abundance level in mines/wells using a Gaussian distribution.
- **Lifecycle ROI Analysis**: Calculate the optimal building level for abundance resources by simulating production from start abundance down to 85%, accounting for build time and scrapping losses (100% recovery for Lv 1-2, 50% for Lv 3+).

## Project Structure

```
simtools/
├── __init__.py          # Package exports
├── models/
│   ├── __init__.py
│   ├── building.py      # Building class - production facilities
│   └── resource.py      # Resource class - producible items
├── api.py               # SimcoAPI client for fetching game data
├── calculator.py        # Profit and ROI calculation logic
├── cli.py               # Command-line interface and display
└── data/                # Static data files
    ├── buildings.json
    ├── abundance_resources.json
    └── seasonal_resources.json
main.py                  # Entry point
```

### Core Classes

- **`Resource`**: Represents a producible item with production rate, wages, inputs, and profit calculation.
- **`Building`**: Represents a production facility with construction costs and the resources it produces.
- **`SimcoAPI`**: Client for fetching resource and price data from the Simcotools API.

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
- `-B`, `--building`: Filter resources by building name.
- `-A`, `--abundance`: Specify the abundance percentage (0-100) for mine and well resources (default: `90`).
- `-O`, `--admin-overhead`: Specify administration overhead percentage to add to wages (default: `0`).
- `-C`, `--contract`: Calculate values for direct contracts (0% market fee, 50% transportation cost).
- `--robots`: Apply 3% wage reduction for buildings with robots installed.
- `-R`, `--roi`: Calculate and display ROI for buildings based on their best performing resource (uses Q0 prices for construction costs).
- `--lifecycle`: Enable lifecycle ROI analysis for abundance resources. This simulates production from a starting abundance (set by `-A`) down to 85%, including build time and scrapping losses.
- `-D`, `--debug-unassigned`: List all resources that are not assigned to any building in `buildings.json`.
- `-E`, `--exclude-seasonal`: Exclude seasonal resources from all calculations.
- `-P`, `--prospect`: Enable prospecting simulation mode.
- `-T`, `--time`: Time in hours for one build attempt (default: 12). Uses the abundance set by `-A` as the target.
- `-L`, `--slots`: Number of simultaneous building slots (default: 1).
- `--max-level`: Maximum building level for ROI analysis (default: 20). Used when filtering by building with `--roi` or with `--lifecycle`.
- `--step-roi`: Calculate ROI based on individual upgrade steps ($L \to L+1$) rather than cumulative investment. Shows ROI for the *additional* profit gained from that specific upgrade.
- `--build-time`: Base construction time in hours (Level 1) for lifecycle analysis. Build time for level $L$ is calculated as $(\sum_{k=1}^{L} k) \times \text{base}$. Abundance decays during this time.

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

**Filter by building (e.g., Farm) to see its production profits:**
```bash
uv run main.py -B Farm
```

**Calculate direct contract profits (0% fee, 50% transport):**
```bash
uv run main.py --contract
```

**Exclude seasonal resources from calculations:**
```bash
uv run main.py -E
```

**Simulate prospecting for 95% abundance with 12h build time:**
```bash
uv run main.py --prospect -A 95 -T 12
```

**Simulate prospecting with 3 parallel slots:**
```bash
uv run main.py --prospect -A 98 -T 12 -L 3
```

**Find the optimal building level for a Mine starting at 95% abundance:**
```bash
uv run main.py --lifecycle -A 95 -B Mine --build-time 2
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

### ROI Analysis Table (when using --roi)
If the `-R` or `--roi` flag is used, a second table is displayed showing:
- **Building**: The name of the building.
- **Best Resource**: The resource produced by that building which yields the highest hourly profit.
- **Building Cost**: Total construction cost using current Q0 market prices.
- **Daily Profit**: Estimated daily profit from producing the best resource (24 hours).
- **ROI (Daily)**: Return on Investment per day as a percentage.
- **Break Even**: Estimated number of days to recover the construction cost.

### Lifecycle Analysis Table (when using --lifecycle)

Displays a simulation of the building's entire useful life from starting abundance down to 85%:
- **Resource**: The resource being produced.
- **Level**: Building level being simulated.
- **Build(h)**: Total time spent on construction and upgrades (hours). Abundance decays during this period.
- **Prod Days**: Total days of active production until abundance hits 85%.
- **Investment**: Total cash spent on construction and upgrades.
- **Unrecoverable**: The net loss on building materials when scrapping (50% loss for Level 3+).
- **Ops Profit**: Total profit from production over the building's life.
- **Net Profit**: `Ops Profit - Unrecoverable`. This is the primary metric for choosing the best upgrade level.

**Level-based ROI Analysis (when using --roi and --building):**
If the `-B` or `--building` flag is used along with `-R`, the ROI analysis will show levels from 1 up to `--max-level` (default 20) for the filtered building. This helps visualize how profit and investment scale as the building is leveled up.


### Prospecting Simulation Tables (when using --prospect)
When using the `-P` or `--prospect` flag, the tool displays:
- **Prospecting Simulation Results**:
    - **Target Abundance**: The percentage you are aiming for.
    - **Probability of Success**: The chance of hitting the target in a single attempt.
    - **Expected Attempts**: The average number of attempts needed (1/p).
    - **Expected Time**: The average time required to hit the target.
    - **Days until 85%**: If target > 85%, shows estimated days until abundance decays to 85% (at 0.032% daily decay).
- **Confidence Intervals (Time to Success)**:
    - Shows the number of attempts and total time required to be X% sure (50%, 80%, 90%, 95%, 99%) of having found the target abundance.

## Data Files

- `resources.json`: Contains raw metadata for all resources (wages, inputs, production rates). Generated from API.
- `vwaps.json`: Contains the latest market Volume Weighted Average Prices for all resources and qualities. Generated from API.
- `simtools/data/abundance_resources.json`: List of resources that use the abundance calculation.
- `simtools/data/seasonal_resources.json`: List of resources that are considered seasonal and can be excluded.
- `simtools/data/buildings.json`: Contains metadata for buildings, including construction costs and the resources they produce.

## Programmatic Usage

The package can also be imported for programmatic use:

```python
from simtools import Resource, Building, SimcoAPI
from simtools.calculator import calculate_all_profits, ProfitConfig

# Fetch data
api = SimcoAPI(realm=0)
resources_data = api.get_resources()
vwaps = api.get_market_vwaps()

# Create resource objects
resources = [Resource.from_api_data(r) for r in resources_data["resources"]]

# Load buildings
buildings = Building.load_all()

# Calculate profits
config = ProfitConfig(quality=0, abundance=90.0)
profits = calculate_all_profits(resources, price_map, transport_price, config)
```
