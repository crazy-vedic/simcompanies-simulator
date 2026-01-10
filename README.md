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
uv run main.py [command] [options]
```

### Commands

Simtools uses a subcommand structure for better organization:

#### `profit` - Calculate production profits (default)
Calculate and display production profits for resources.

**Options:**
- `-q`, `--quality`: Quality level (default: 0)
- `-a`, `--abundance`: Abundance percentage for mine/well resources (default: 90)
- `-b`, `--building`: Filter by building name
- `-s`, `--search`: Search resources by name (case-insensitive)
- `-c`, `--contract`: Direct contract mode (0% market fee, 50% transport)
- `-r`, `--robots`: Apply 3% wage reduction
- `-o`, `--overhead`: Admin overhead percentage (default: 0)
- `-e`, `--no-seasonal`: Exclude seasonal resources

#### `roi` - Building ROI analysis
Analyze return on investment for buildings based on their best performing resource.

**Options:**
- `-b`, `--building`: Filter by building name
- `-l`, `--level`: Maximum building level (default: 20)
- `-p`, `--per-step`: Calculate per-upgrade-step ROI
- Plus inherited: `-q`, `-a`, `-c`, `-r`, `-o`, `-e`

#### `lifecycle` - Abundance decay/lifecycle analysis
Calculate lifecycle ROI for abundance resources (simulates decay from starting abundance to 85%).

**Options:**
- `-b`, `--building`: Filter by building name
- `-l`, `--level`: Maximum building level (default: 20)
- `-t`, `--time`: Base build time in hours
- Plus inherited: `-q`, `-a`, `-c`, `-r`, `-o`, `-e`

#### `prospect` - Prospecting simulation
Simulate prospecting to find target abundance.

**Options:**
- `-t`, `--target`: Target abundance percentage (required)
- `-d`, `--duration`: Build time per attempt in hours (default: 12)
- `-s`, `--slots`: Number of building slots (default: 1)

#### `debug` - Debugging utilities
Various debugging and diagnostic tools.

**Options:**
- `-u`, `--unassigned`: List resources not assigned to any building

### Examples

**View top 30 most profitable quality 0 resources with 85% abundance:**
```bash
uv run main.py profit -a 85
# or without subcommand (defaults to profit):
uv run main.py -a 85
```

**Calculate profits for Quality 2 resources:**
```bash
uv run main.py profit -q 2
```

**Search for "Electric" and "Water" related resources at Quality 1:**
```bash
uv run main.py profit -s Electric Water -q 1
```

**Filter by building (e.g., Farm) to see its production profits:**
```bash
uv run main.py profit -b Farm
```

**Calculate direct contract profits (0% fee, 50% transport):**
```bash
uv run main.py profit -c
# or with building filter:
uv run main.py profit -b Farm -c
```

**Exclude seasonal resources from calculations:**
```bash
uv run main.py profit -e
```

**Building ROI analysis:**
```bash
# ROI for all buildings
uv run main.py roi

# ROI for specific building
uv run main.py roi -b Farm

# ROI with level-by-level breakdown
uv run main.py roi -b Mine -l 15 --per-step
```

**Simulate prospecting for 95% abundance with 12h build time:**
```bash
uv run main.py prospect -t 95 -d 12
```

**Simulate prospecting with 3 parallel slots:**
```bash
uv run main.py prospect -t 98 -d 12 -s 3
```

**Find the optimal building level for a Mine starting at 95% abundance:**
```bash
uv run main.py lifecycle -a 95 -b Mine -t 2
```

**Debug: List unassigned resources:**
```bash
uv run main.py debug -u
```

### Backwards Compatibility

For ease of migration, the tool supports running without specifying a subcommand. When no subcommand is provided, it defaults to the `profit` command. Additionally, common flags (`-q`, `-a`, `-c`, `-r`, `-o`, `-b`, `-s`, `-e`) are available at the top level for backwards compatibility:

```bash
# These are equivalent:
uv run main.py -a 85
uv run main.py profit -a 85
```

However, using the explicit subcommand structure is recommended for clarity and to access all command-specific options.

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
