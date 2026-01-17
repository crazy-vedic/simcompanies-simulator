"""Command-line interface for simtools."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from simtools import __version__
from simtools.api import SimcoAPI
from simtools.calculator import (
    ProfitConfig,
    calculate_all_profits,
    calculate_building_roi,
    calculate_company_building_stats,
    calculate_level_roi,
    calculate_lifecycle_roi,
    calculate_retail_units_per_hour,
    calculate_upgrade_recommendations,
    compare_market_vs_contract,
    simulate_prospecting,
)
from simtools.genetic import (
    GeneticAlgorithm,
    SimulationConfig,
    render_ascii_graph,
)
from simtools.models.building import Building, build_resource_to_building_map
from simtools.models.resource import Resource
from simtools.models.market import MarketData

console = Console()


def get_data_path(filename: str) -> Path:
    """Get the path to a data file.

    First checks the simtools/data directory, then falls back to the workspace root.

    Args:
        filename: Name of the data file.

    Returns:
        Path to the data file.
    """
    # Check package data directory first
    package_data = Path(__file__).parent / "data" / filename
    if package_data.exists():
        return package_data

    # Fall back to workspace root
    return Path(filename)


def load_json_list(filepath: Path) -> list[str]:
    """Load a JSON file containing a list of strings.

    Args:
        filepath: Path to the JSON file.

    Returns:
        List of strings, or empty list if file doesn't exist.
    """
    if not filepath.exists():
        return []
    with open(filepath, "r") as f:
        return json.load(f)


def display_prospecting_results(results: dict) -> None:
    """Display prospecting simulation results.

    Args:
        results: Results from simulate_prospecting().
    """
    if results.get("impossible"):
        console.print(
            f"[bold red]Target abundance {results['target_abundance']*100:.1f}% "
            f"is impossible with the current distribution.[/bold red]"
        )
        return

    table_width = 60

    table = Table(
        title="Prospecting Simulation Results",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        width=table_width,
    )
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Target Abundance", f"{results['target_abundance']*100:.1f}%")
    table.add_row("Build Time per Attempt", f"{results['attempt_time']:.1f} hours")
    table.add_row("Number of Slots", f"{results['slots']}")
    table.add_row("Prob. Success (Single)", f"{results['p_success_single']*100:.4f}%")
    table.add_row("Prob. Success (Block)", f"{results['p_success_block']*100:.4f}%")
    table.add_row("Expected Blocks", f"{results['expected_blocks']:.2f}")
    table.add_row(
        "Expected Time",
        f"{results['expected_time']:.2f} hours ({results['expected_time']/24:.2f} days)",
    )

    if results.get("days_to_85") is not None:
        table.add_row("Days until 85%", f"{results['days_to_85']:.1f} days")

    console.print(table)

    # Confidence intervals table
    conf_table = Table(
        title="Confidence Intervals (Time to Success)",
        show_header=True,
        header_style="bold blue",
        box=box.ROUNDED,
        width=table_width,
    )
    conf_table.add_column("Confidence Level", style="cyan")
    conf_table.add_column("Required Blocks", justify="right", style="yellow")
    conf_table.add_column("Required Time", justify="right", style="green")

    for ci in results["confidence_intervals"]:
        conf_table.add_row(
            f"{ci['confidence']*100:.0f}%",
            f"{ci['blocks']}",
            f"{ci['time_hours']:.1f}h ({ci['time_days']:.1f}d)",
        )

    console.print(conf_table)


def display_profits_table(
    profits: list[dict],
    transport_price: float,
    config: ProfitConfig,
    building_level: int = 1,
    search_terms: list[str] | None = None,
    building_terms: list[str] | None = None,
) -> None:
    """Display the profits table.

    Args:
        profits: List of profit dictionaries.
        transport_price: Price per transport unit.
        config: Profit calculation configuration.
        building_level: Building level used for calculations.
        search_terms: Search terms used for filtering (for header).
        building_terms: Building terms used for filtering (for header).
    """
    # Build header title
    header_title = f"Top 30 Most Profitable Resources (Level {building_level})"
    if search_terms or building_terms:
        parts = []
        if search_terms:
            parts.append(f"search: '{', '.join(search_terms)}'")
        if building_terms:
            parts.append(f"building: '{', '.join(building_terms)}'")
        header_title = f"Results for {' & '.join(parts)} (Level {building_level})"

    if config.is_contract:
        header_title += " (Direct Contract Mode)"

    console.print(f"\n[bold blue]{header_title}[/bold blue]")
    market_fee_display = "0%" if config.is_contract else "4%"
    console.print(
        f"Quality: [bold cyan]{config.quality}[/bold cyan] | "
        f"Transport: [bold cyan]${transport_price:.3f}[/bold cyan] | "
        f"Market Fee: [bold cyan]{market_fee_display}[/bold cyan] | "
        f"Admin Overhead: [bold cyan]{config.admin_overhead}%[/bold cyan] | "
        f"Robots: [bold cyan]{'Yes' if config.has_robots else 'No'}[/bold cyan]"
    )

    table = Table(
        show_header=True,
        header_style="bold white on blue",
        box=box.ROUNDED,
        border_style="bright_black",
    )
    table.add_column("Resource", style="bold white", width=25)
    table.add_column("Profit/hr", justify="right")
    table.add_column("Revenue/hr", justify="right", style="white")
    table.add_column("Fee/hr", justify="right", style="red")
    table.add_column("Costs/hr", justify="right", style="yellow")
    table.add_column("Transp/hr", justify="right", style="magenta")

    display_count = 30 if not search_terms else len(profits)
    for p in profits[:display_count]:
        warn = " [bold red](!)[/bold red]" if p["missing_input_price"] else ""
        abundance_mark = " [bold yellow](*)[/bold yellow]" if p["is_abundance_res"] else ""

        profit_style = "bold green" if p["profit_per_hour"] >= 0 else "bold red"
        
        # Scale values by building level
        profit_hr = p["profit_per_hour"] * building_level
        revenue_hr = p["revenue_per_hour"] * building_level
        fee_hr = p["market_fee_per_hour"] * building_level
        costs_hr = p["costs_per_hour"] * building_level
        transport_hr = p["transport_costs_per_hour"] * building_level

        table.add_row(
            f"{p['name']}{abundance_mark}",
            f"[{profit_style}]${profit_hr:,.2f}[/{profit_style}]",
            f"${revenue_hr:,.2f}",
            f"${fee_hr:,.2f}",
            f"${costs_hr:,.2f}",
            f"${transport_hr:,.2f}{warn}",
        )

    console.print(table)

    if any(p["is_abundance_res"] for p in profits[:display_count]):
        console.print(
            f"\n[bold yellow](*)[/bold yellow] indicates abundance-based resource "
            f"(applied {config.abundance}% abundance)"
        )
    if any(p["missing_input_price"] for p in profits[:display_count]):
        console.print(
            f"[bold red](!)[/bold red] indicates one or more source materials had no "
            f"Quality {config.quality} market price"
        )


def display_roi_table(roi_data: list[dict]) -> None:
    """Display the ROI analysis table.

    Args:
        roi_data: List of ROI dictionaries.
    """
    roi_table = Table(
        title="Building ROI Analysis",
        show_header=True,
        header_style="bold green",
        box=box.ROUNDED,
    )
    roi_table.add_column("Building", style="bold white")
    if roi_data and "level" in roi_data[0]:
        col_name = "Step/Lv" if any("→" in str(d.get("level", "")) for d in roi_data) else "Lv"
        roi_table.add_column(col_name, justify="right", style="cyan")
    roi_table.add_column("Best Resource", style="cyan")
    roi_table.add_column("Building Cost", justify="right", style="magenta")
    roi_table.add_column("Daily Profit", justify="right", style="green")
    roi_table.add_column("ROI (Daily)", justify="right", style="bold yellow")
    roi_table.add_column("Break Even", justify="right", style="white")

    for d in roi_data:
        if d["break_even"] == float("inf"):
            break_even_str = "∞"
        elif d["daily_profit"] < 0:
            break_even_str = "Never"
        else:
            break_even_str = f"{d['break_even']:.1f} days"

        warn = " (!)" if d["missing_cost"] else ""

        row_data = [
            d["building"],
        ]
        if "level" in d:
            row_data.append(str(d["level"]))
        
        row_data.extend([
            d["resource"],
            f"${d['cost']:,.0f}{warn}",
            f"${d['daily_profit']:,.0f}",
            f"{d['roi']:.2f}%",
            break_even_str,
        ])
        
        roi_table.add_row(*row_data)

    console.print("\n")
    console.print(roi_table)
    if any(d["missing_cost"] for d in roi_data):
        console.print(
            "[yellow](!) Warning: Some building costs calculated with missing "
            "material prices (assumed $0).[/yellow]"
        )


def display_lifecycle_table(results: list[dict], start_abundance: float) -> None:
    """Display lifecycle ROI analysis table.

    Args:
        results: List of lifecycle result dictionaries.
        start_abundance: Starting abundance percentage.
    """
    table = Table(
        title=f"Lifecycle Analysis (Abundance {start_abundance}% -> 85%)",
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
    )
    table.add_column("Resource", style="bold white")
    table.add_column("Level", justify="right", style="cyan")
    table.add_column("Build(h)", justify="right", style="blue")
    table.add_column("Prod Days", justify="right", style="white")
    table.add_column("Investment", justify="right", style="magenta")
    table.add_column("Unrecoverable", justify="right", style="red")
    table.add_column("Ops Profit", justify="right", style="green")
    table.add_column("Net Profit", justify="right", style="bold yellow")
    
    # Show top 30
    for res in results[:30]:
        warn = " (!)" if res["missing_cost"] else ""
        table.add_row(
            res["resource"],
            str(res["level"]),
            f"{res['build_time_hours']:.1f}",
            str(res["days"]),
            f"${res['investment']:,.0f}{warn}",
            f"${res['unrecoverable']:,.0f}",
            f"${res['operational_profit']:,.0f}",
            f"${res['net_profit']:,.0f}",
        )
        
    console.print("\n")
    console.print(table)
    if any(r["missing_cost"] for r in results):
        console.print(
            "[yellow](!) Warning: Some costs/profits calculated with missing prices.[/yellow]"
        )


def display_compare_table(
    comparisons: list[dict],
    transport_price: float,
    config: ProfitConfig,
) -> None:
    """Display market vs contract comparison table.

    Args:
        comparisons: List of comparison dictionaries from compare_market_vs_contract.
        transport_price: Price per transport unit.
        config: Profit calculation configuration.
    """
    console.print(f"\n[bold blue]Market vs Contract Comparison[/bold blue]")
    console.print(
        f"Quality: [bold cyan]{config.quality}[/bold cyan] | "
        f"Transport: [bold cyan]${transport_price:.3f}[/bold cyan] | "
        f"Abundance: [bold cyan]{config.abundance}%[/bold cyan] | "
        f"Admin Overhead: [bold cyan]{config.admin_overhead}%[/bold cyan] | "
        f"Robots: [bold cyan]{'Yes' if config.has_robots else 'No'}[/bold cyan]"
    )

    table = Table(
        show_header=True,
        header_style="bold white on blue",
        box=box.ROUNDED,
        border_style="bright_black",
    )
    
    # Resource name column
    table.add_column("Resource", style="bold white")
    
    # Market columns
    table.add_column("Mkt Price", justify="right", style="cyan")
    table.add_column("Mkt Fee/u", justify="right", style="red")
    table.add_column("Mkt Trans/u", justify="right", style="magenta")
    table.add_column("Mkt Net/u", justify="right", style="white")
    table.add_column("Mkt $/hr", justify="right", style="white")
    
    # Contract columns
    table.add_column("Cnt Price", justify="right", style="cyan")
    table.add_column("Cnt Fee/u", justify="right", style="red")
    table.add_column("Cnt Trans/u", justify="right", style="magenta")
    table.add_column("Cnt Net/u", justify="right", style="white")
    table.add_column("Cnt $/hr", justify="right", style="white")
    
    # Difference columns
    table.add_column("Diff/u", justify="right")
    table.add_column("Diff/hr", justify="right")

    for comp in comparisons:
        warn = " [bold red](!)[/bold red]" if comp["missing_input_price"] else ""
        abundance_mark = " [bold yellow](*)[/bold yellow]" if comp["is_abundance_res"] else ""
        
        # Determine styling for differences
        if comp["diff_per_unit"] > 0:
            diff_unit_style = "bold green"
            diff_unit_prefix = "+"
        elif comp["diff_per_unit"] < 0:
            diff_unit_style = "bold red"
            diff_unit_prefix = ""
        else:
            diff_unit_style = "white"
            diff_unit_prefix = ""
            
        if comp["diff_per_hour"] > 0:
            diff_hour_style = "bold green"
            diff_hour_prefix = "+"
        elif comp["diff_per_hour"] < 0:
            diff_hour_style = "bold red"
            diff_hour_prefix = ""
        else:
            diff_hour_style = "white"
            diff_hour_prefix = ""

        table.add_row(
            f"{comp['name']}{abundance_mark}{warn}",
            f"${comp['market']['price']:.2f}",
            f"${comp['market']['fee_per_unit']:.2f}",
            f"${comp['market']['transport_per_unit']:.2f}",
            f"${comp['market']['net_per_unit']:.2f}",
            f"${comp['market']['profit_per_hour']:.2f}",
            f"${comp['contract']['price']:.2f}",
            f"${comp['contract']['fee_per_unit']:.2f}",
            f"${comp['contract']['transport_per_unit']:.2f}",
            f"${comp['contract']['net_per_unit']:.2f}",
            f"${comp['contract']['profit_per_hour']:.2f}",
            f"[{diff_unit_style}]{diff_unit_prefix}${comp['diff_per_unit']:.2f}[/{diff_unit_style}]",
            f"[{diff_hour_style}]{diff_hour_prefix}${comp['diff_per_hour']:.2f}[/{diff_hour_style}]",
        )

    console.print(table)

    if any(comp["is_abundance_res"] for comp in comparisons):
        console.print(
            f"\n[bold yellow](*)[/bold yellow] indicates abundance-based resource "
            f"(applied {config.abundance}% abundance)"
        )
    if any(comp["missing_input_price"] for comp in comparisons):
        console.print(
            f"[bold red](!)[/bold red] indicates one or more source materials had no "
            f"Quality {config.quality} market price"
        )


def display_company_analysis(
    company_data: dict,
    building_stats: list[dict],
    config: ProfitConfig,
) -> None:
    """Display company analysis results.

    Args:
        company_data: Raw company data from API.
        building_stats: List of building statistics from calculate_company_building_stats.
        config: Profit calculation configuration.
    """
    company = company_data.get("company", {})

    console.print("\n[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]              COMPANY ANALYSIS                                 [/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")

    # Company info
    console.print("[bold cyan]Company Info:[/bold cyan]")
    console.print(f"  • Name: [yellow]{company.get('name', 'N/A')}[/yellow]")
    console.print(f"  • Level: [yellow]{company.get('level', 'N/A')}[/yellow]")
    console.print(f"  • Rating: [yellow]{company.get('rating', 'N/A')}[/yellow]")
    console.print(f"  • Total Buildings: [yellow]{company.get('totalBuildings', 'N/A')}[/yellow]")
    console.print(f"  • Workers: [yellow]{company.get('workers', 'N/A')}[/yellow]")
    console.print(f"  • Building Value: [yellow]${company.get('buildingValue', 0):,.0f}[/yellow]")
    console.print()

    # Configuration used
    console.print("[bold cyan]Analysis Configuration:[/bold cyan]")
    market_fee_display = "0%" if config.is_contract else "4%"
    console.print(
        f"  • Quality: [yellow]{config.quality}[/yellow] | "
        f"Abundance: [yellow]{config.abundance}%[/yellow] | "
        f"Market Fee: [yellow]{market_fee_display}[/yellow] | "
        f"Admin Overhead: [yellow]{config.admin_overhead}%[/yellow] | "
        f"Robots: [yellow]{'Yes' if config.has_robots else 'No'}[/yellow]"
    )
    console.print()

    if not building_stats:
        console.print("[yellow]No building data available for analysis.[/yellow]")
        return

    # Buildings table
    table = Table(
        title="Building Performance Analysis",
        show_header=True,
        header_style="bold white on green",
        box=box.ROUNDED,
    )
    table.add_column("Building", style="bold white")
    table.add_column("Lv", justify="center", style="cyan")
    table.add_column("Best Resource", style="magenta")
    table.add_column("$/hour", justify="right", style="green")
    table.add_column("$/day", justify="right", style="green")
    table.add_column("Value", justify="right", style="yellow")
    table.add_column("ROI/day", justify="right", style="bold cyan")
    table.add_column("Break Even", justify="right", style="white")

    total_hourly = 0.0
    total_daily = 0.0
    total_value = 0.0

    for stat in building_stats:
        total_hourly += stat["hourly_profit"]
        total_daily += stat["daily_profit"]
        total_value += stat["building_value"]

        if stat["break_even_days"] == float("inf"):
            break_even_str = "∞"
        elif stat["daily_profit"] < 0:
            break_even_str = "Never"
        else:
            break_even_str = f"{stat['break_even_days']:.1f} days"

        warn = " (!)" if stat["missing_cost"] else ""
        profit_style = "green" if stat["hourly_profit"] >= 0 else "red"

        table.add_row(
            stat["building_name"],
            str(stat["level"]),
            stat["best_resource"] or "N/A",
            f"[{profit_style}]${stat['hourly_profit']:,.2f}[/{profit_style}]",
            f"[{profit_style}]${stat['daily_profit']:,.2f}[/{profit_style}]",
            f"${stat['building_value']:,.0f}{warn}",
            f"{stat['roi_daily']:.2f}%",
            break_even_str,
        )

    console.print(table)

    # Summary
    console.print("\n[bold magenta]Summary:[/bold magenta]")
    profit_style = "green" if total_hourly >= 0 else "red"
    console.print(f"  • Total Hourly Profit: [{profit_style}]${total_hourly:,.2f}[/{profit_style}]")
    console.print(f"  • Total Daily Profit: [{profit_style}]${total_daily:,.2f}[/{profit_style}]")
    console.print(f"  • Total Building Value: [yellow]${total_value:,.0f}[/yellow]")
    if total_value > 0:
        overall_roi = (total_daily / total_value) * 100
        console.print(f"  • Overall Daily ROI: [cyan]{overall_roi:.2f}%[/cyan]")
        if total_daily > 0:
            overall_break_even = total_value / total_daily
            console.print(f"  • Overall Break Even: [white]{overall_break_even:.1f} days[/white]")

    if any(stat["missing_cost"] for stat in building_stats):
        console.print(
            "\n[yellow](!) Warning: Some building costs calculated with missing "
            "material prices (assumed $0).[/yellow]"
        )


def display_retail_results(
    resource_name: str,
    units_per_hour: float,
    price: float,
    quality: int,
    building_level: int,
    sales_speed_bonus: float,
    retail_info: dict | None = None,
) -> None:
    """Display retail calculation results.

    Args:
        resource_name: Name of the resource being retailed.
        units_per_hour: Calculated units sold per hour.
        price: Retail price per unit.
        quality: Quality level used.
        building_level: Building level used.
        sales_speed_bonus: Sales speed bonus percentage used.
        retail_info: Retail info dict used for calculation (optional).
    """
    console.print("\n[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]              RETAIL CALCULATION RESULTS                       [/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")

    table = Table(
        show_header=True,
        header_style="bold white on blue",
        box=box.ROUNDED,
        border_style="bright_black",
    )
    table.add_column("Parameter", style="bold white", width=30)
    table.add_column("Value", justify="right", style="cyan")

    table.add_row("Resource", resource_name)
    table.add_row("Retail Price", f"${price:.2f}")
    table.add_row("Quality", f"Q{quality}")
    table.add_row("Building Level", str(building_level))
    table.add_row("Sales Speed Bonus", f"{sales_speed_bonus:.1f}%")
    if retail_info:
        table.add_row("Market Saturation", f"{retail_info.get('saturation', 0):.4f}")

    if math.isnan(units_per_hour):
        table.add_row("Units/Hour", "[bold red]Invalid (NaN)[/bold red]")
        table.add_row("Units/Day", "[bold red]Invalid[/bold red]")
        table.add_row("Revenue/Hour", "[bold red]Invalid[/bold red]")
        table.add_row("Revenue/Day", "[bold red]Invalid[/bold red]")
    else:
        units_per_day = units_per_hour * 24
        revenue_per_hour = units_per_hour * price
        revenue_per_day = units_per_day * price

        table.add_row("Units/Hour", f"{units_per_hour:.3f}")
        table.add_row("Units/Day", f"{units_per_day:.3f}")
        table.add_row("Revenue/Hour", f"${revenue_per_hour:,.2f}")
        table.add_row("Revenue/Day", f"${revenue_per_day:,.2f}")

    console.print(table)


def display_retail_table(
    profits: list[dict],
    config: ProfitConfig,
    building_level: int = 1,
) -> None:
    """Display retail profits table ordered by PPH.

    Args:
        profits: List of retail profit dictionaries.
        config: Profit calculation configuration.
        building_level: Building level used for calculations.
    """
    console.print(f"\n[bold blue]Retail Sales - Ordered by Profit/Hour (Level {building_level})[/bold blue]")
    console.print(
        f"Quality: [bold cyan]{config.quality}[/bold cyan] | "
        f"Admin Overhead: [bold cyan]{config.admin_overhead}%[/bold cyan] | "
        f"Sales Speed Bonus: [bold cyan]{config.sales_speed_bonus * 100:.1f}%[/bold cyan]"
    )

    table = Table(
        show_header=True,
        header_style="bold white on blue",
        box=box.ROUNDED,
        border_style="bright_black",
    )
    table.add_column("Resource", style="bold white", width=25)
    table.add_column("Profit/hr", justify="right")
    table.add_column("Revenue/hr", justify="right", style="white")
    table.add_column("Input Cost", justify="right", style="yellow")
    table.add_column("Units/hr", justify="right", style="cyan")
    table.add_column("Sell Price", justify="right", style="green")

    for p in profits[:30]:  # Show top 30
        # Extract retail-specific data
        name = p["name"].replace(" (Retail)", "")  # Remove suffix for cleaner display
        profit_style = "bold green" if p["profit_per_hour"] >= 0 else "bold red"
        
        units_sold = p.get("units_sold_per_hour", 0)
        retail_price = p.get("retail_price", 0)
        wages = p.get("wages_per_hour", 0)
        
        # Calculate input cost per hour from total costs
        input_cost_per_hour = p["costs_per_hour"] - wages
        
        table.add_row(
            name,
            f"[{profit_style}]${p['profit_per_hour']:,.2f}[/{profit_style}]",
            f"${p['revenue_per_hour']:,.2f}",
            f"${input_cost_per_hour:,.2f}",
            f"{units_sold:.2f}",
            f"${retail_price:.2f}",
        )

    console.print(table)


def display_scenario_table(rows: list[dict], building_level: int) -> None:
    """Display scenario comparisons for production/retail modes."""
    if not rows:
        console.print("[yellow]No scenarios to display.[/yellow]")
        return

    table = Table(
        title=f"Scenario Comparison (Level {building_level})",
        show_header=True,
        header_style="bold white on blue",
        box=box.ROUNDED,
        border_style="bright_black",
    )
    table.add_column("Mode", style="bold white", width=12)
    table.add_column("Resource", style="bold white", width=18)
    table.add_column("Q", justify="center", style="cyan")
    table.add_column("Price", justify="right")
    table.add_column("Profit/hr", justify="right")
    table.add_column("Revenue/hr", justify="right")
    table.add_column("Costs/hr", justify="right")
    table.add_column("Fee/hr", justify="right")
    table.add_column("Transp/hr", justify="right")
    table.add_column("Units/hr", justify="right")

    for row in rows:
        profit_style = "bold green" if row["profit_per_hour"] >= 0 else "bold red"
        table.add_row(
            row["mode"],
            row["resource"],
            f"Q{row['quality']}",
            f"${row['price']:.2f}",
            f"[{profit_style}]${row['profit_per_hour']:,.2f}[/{profit_style}]",
            f"${row['revenue_per_hour']:,.2f}",
            f"${row['costs_per_hour']:,.2f}",
            f"${row['fee_per_hour']:,.2f}",
            f"${row['transport_per_hour']:,.2f}",
            f"{row['units_per_hour']:.2f}",
        )

    console.print(table)


def display_upgrade_recommendations(
    recommendations: list[dict],
    config: ProfitConfig,
    top_n: int = 10,
) -> None:
    """Display upgrade recommendations.

    Args:
        recommendations: List of upgrade recommendations sorted by marginal ROI.
        config: Profit calculation configuration.
        top_n: Number of top recommendations to display.
    """
    console.print("\n[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]              UPGRADE RECOMMENDATIONS                          [/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")

    if not recommendations:
        console.print("[yellow]No upgrade recommendations available.[/yellow]")
        return

    console.print(
        "[bold cyan]Recommendation based on marginal ROI:[/bold cyan] "
        "The best upgrade is the one that gives the highest return on the upgrade cost."
    )
    console.print()

    table = Table(
        title=f"Top {min(top_n, len(recommendations))} Upgrade Recommendations",
        show_header=True,
        header_style="bold white on blue",
        box=box.ROUNDED,
    )
    table.add_column("#", justify="center", style="bold white")
    table.add_column("Building", style="bold white")
    table.add_column("Upgrade", justify="center", style="cyan")
    table.add_column("Best Resource", style="magenta")
    table.add_column("Upgrade Cost", justify="right", style="yellow")
    table.add_column("+$/day", justify="right", style="green")
    table.add_column("Marginal ROI", justify="right", style="bold cyan")
    table.add_column("Break Even", justify="right", style="white")

    for i, rec in enumerate(recommendations[:top_n], 1):
        if rec["marginal_break_even"] == float("inf"):
            break_even_str = "∞"
        elif rec["additional_daily_profit"] < 0:
            break_even_str = "Never"
        else:
            break_even_str = f"{rec['marginal_break_even']:.1f} days"

        warn = " (!)" if rec["missing_cost"] else ""
        profit_style = "green" if rec["additional_daily_profit"] >= 0 else "red"

        # Highlight top recommendation
        rank_style = "bold green" if i == 1 else "white"

        table.add_row(
            f"[{rank_style}]{i}[/{rank_style}]",
            rec["building_name"],
            f"Lv{rec['current_level']}→{rec['next_level']}",
            rec["best_resource"],
            f"${rec['upgrade_cost']:,.0f}{warn}",
            f"[{profit_style}]+${rec['additional_daily_profit']:,.2f}[/{profit_style}]",
            f"{rec['marginal_roi']:.2f}%",
            break_even_str,
        )

    console.print(table)

    if recommendations:
        best = recommendations[0]
        console.print(
            f"\n[bold green]★ Recommended next upgrade:[/bold green] "
            f"[bold]{best['building_name']}[/bold] from Level {best['current_level']} to "
            f"Level {best['next_level']}"
        )
        console.print(
            f"  Cost: [yellow]${best['upgrade_cost']:,.0f}[/yellow] → "
            f"Adds [green]+${best['additional_daily_profit']:,.2f}/day[/green] → "
            f"ROI: [cyan]{best['marginal_roi']:.2f}%[/cyan]"
        )

    if any(rec["missing_cost"] for rec in recommendations[:top_n]):
        console.print(
            "\n[yellow](!) Warning: Some upgrade costs calculated with missing "
            "material prices (assumed $0).[/yellow]"
        )


def prompt_building_levels(
    building_ids: dict[str, int],
    buildings: list[Building],
) -> dict[str, int]:
    """Prompt the user to enter building levels interactively.

    Args:
        building_ids: Dict of building ID to count from API.
        buildings: List of all Building instances.

    Returns:
        Dict of building ID to level.
    """
    # Create lookup from building ID to building name
    id_to_name = {b.id: b.name for b in buildings}

    console.print("\n[bold cyan]Enter building levels:[/bold cyan]")
    console.print("(Press Enter to use default level 1)")
    console.print()

    building_levels = {}

    for building_id, count in building_ids.items():
        building_name = id_to_name.get(building_id)

        # Skip buildings that are not in buildings.json
        if building_name is None:
            console.print(f"[yellow]Skipping unknown building (ID: {building_id}) - not in buildings.json[/yellow]")
            continue

        if count > 1:
            console.print(f"[bold]{building_name}[/bold] (x{count}):")
            for i in range(count):
                while True:
                    try:
                        level_input = console.input(f"  Building #{i+1} level: ")
                        if level_input.strip() == "":
                            level = 1
                        else:
                            level = int(level_input)
                        if level < 1:
                            console.print("[red]Level must be at least 1[/red]")
                            continue
                        break
                    except ValueError:
                        console.print("[red]Please enter a valid number[/red]")

                # Use building_id with index for multiple buildings of same type
                building_levels[f"{building_id}_{i}"] = level
        else:
            while True:
                try:
                    level_input = console.input(f"[bold]{building_name}[/bold] level: ")
                    if level_input.strip() == "":
                        level = 1
                    else:
                        level = int(level_input)
                    if level < 1:
                        console.print("[red]Level must be at least 1[/red]")
                        continue
                    break
                except ValueError:
                    console.print("[red]Please enter a valid number[/red]")

            building_levels[building_id] = level

    return building_levels


def display_genetic_results(
    best_individual,
    fitness_history: list[float],
    config: SimulationConfig,
    ga: GeneticAlgorithm,
) -> None:
    """Display genetic algorithm results.

    Args:
        best_individual: The best individual from the genetic algorithm.
        fitness_history: List of best fitness values per generation.
        config: Simulation configuration used.
        ga: The GeneticAlgorithm instance for calculating costs.
    """
    console.print("\n[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]              GENETIC ALGORITHM RESULTS                        [/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")

    # Configuration summary
    console.print("[bold cyan]Configuration:[/bold cyan]")
    console.print(f"  • Building Slots: [yellow]{config.slots}[/yellow]")
    console.print(f"  • Max Budget: [yellow]${config.budget:,.0f}[/yellow]")
    console.print(f"  • Population Size: [yellow]{config.population_size}[/yellow]")
    console.print(f"  • Generations: [yellow]{config.generations}[/yellow]")
    console.print(f"  • Max Building Level: [yellow]{config.max_level}[/yellow]")
    console.print(f"  • Mutation Rate: [yellow]{config.mutation_rate:.1%}[/yellow]")
    console.print(f"  • Crossover Rate: [yellow]{config.crossover_rate:.1%}[/yellow]")
    console.print()

    # Best configuration
    console.print("[bold green]Best Building Configuration:[/bold green]")

    if best_individual.genes:
        table = Table(
            show_header=True,
            header_style="bold white on green",
            box=box.ROUNDED,
        )
        table.add_column("Building", style="bold white")
        table.add_column("Produces", style="magenta")
        table.add_column("Level", justify="center", style="cyan")
        table.add_column("Cost", justify="right", style="yellow")

        for gene in best_individual.genes:
            cost = ga.calculate_building_cost(gene.building_name, gene.level)
            # Get the resource this building will produce (most profitable one)
            resource = ga.get_best_resource_for_building(gene.building_name)
            resource_name = resource.name if resource else "N/A"
            table.add_row(
                gene.building_name,
                resource_name,
                str(gene.level),
                f"${cost:,.0f}",
            )

        console.print(table)
    else:
        console.print("  [red]No buildings in best configuration[/red]")

    console.print()

    # Summary statistics
    budget_status = "WITHIN" if best_individual.total_cost <= config.budget else "OVER"
    budget_style = "green" if budget_status == "WITHIN" else "red"

    console.print("[bold magenta]Results Summary:[/bold magenta]")
    console.print(f"  • Total Investment: [yellow]${best_individual.total_cost:,.0f}[/yellow]")
    console.print(f"  • Budget Status: [{budget_style}]{budget_status}[/{budget_style}] ({best_individual.total_cost / config.budget * 100:.1f}% of budget)")
    console.print(f"  • Buildings Used: [yellow]{len(best_individual.genes)}[/yellow] / {config.slots} slots")

    profit_style = "green" if best_individual.fitness >= 0 else "red"
    console.print(f"  • 48-Hour Profit: [{profit_style}]${best_individual.fitness:,.2f}[/{profit_style}]")

    if best_individual.fitness > 0:
        hourly = best_individual.fitness / 48
        daily = hourly * 24
        console.print(f"  • Hourly Profit: [green]${hourly:,.2f}[/green]")
        console.print(f"  • Daily Profit: [green]${daily:,.2f}[/green]")
        if best_individual.total_cost > 0:
            roi_days = best_individual.total_cost / daily
            console.print(f"  • ROI Break-even: [cyan]{roi_days:.1f} days[/cyan]")

    console.print()

    # Fitness graph
    if fitness_history:
        console.print("[bold blue]Fitness Evolution Graph:[/bold blue]")
        # Adapt width to data size (min 30, max 60)
        graph_width = min(60, max(30, len(fitness_history)))
        graph_lines = render_ascii_graph(fitness_history, width=graph_width, height=12)
        for line in graph_lines:
            console.print(f"  {line}")

    console.print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments using subcommands.

    Returns:
        Parsed arguments namespace.
    """
    # Create parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "-q", "--quality", type=int, default=0, help="Quality level (default: 0)"
    )
    parent_parser.add_argument(
        "-a",
        "--abundance",
        type=float,
        default=90,
        help="Abundance percentage for mine/well resources (default: 90)",
    )
    parent_parser.add_argument(
        "-c",
        "--contract",
        action="store_true",
        help="Direct contract mode (0%% market fee, 50%% transport)",
    )
    parent_parser.add_argument(
        "-r",
        "--robots",
        action="store_true",
        help="Apply 3%% wage reduction",
    )
    parent_parser.add_argument(
        "-o",
        "--overhead",
        type=float,
        default=0,
        dest="admin_overhead",
        help="Admin overhead percentage (default: 0)",
    )
    parent_parser.add_argument(
        "-e",
        "--no-seasonal",
        action="store_true",
        dest="exclude_seasonal",
        help="Exclude seasonal resources",
    )

    # Main parser (includes common options so they work before subcommands)
    parser = argparse.ArgumentParser(
        parents=[parent_parser],
        description="Simtools - Sim Companies calculation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-b", "--building", type=str, nargs="+", help="Filter by building name"
    )
    parser.add_argument(
        "-s",
        "--search",
        type=str,
        nargs="+",
        help="Search resources by name (case-insensitive)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Make the subparser optional by setting required=False
    subparsers.required = False

    # profit subcommand - default/main functionality
    profit_parser = subparsers.add_parser(
        "profit",
        help="Calculate production profits",
        description="Calculate and display production profits for resources",
    )
    profit_parser.add_argument(
        "-b", "--building", type=str, nargs="+", help="Filter by building name"
    )
    profit_parser.add_argument(
        "-s",
        "--search",
        type=str,
        nargs="+",
        help="Search resources by name (case-insensitive)",
    )
    profit_parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=1,
        dest="building_level",
        help="Building level for production calculations (default: 1)",
    )

    # roi subcommand
    roi_parser = subparsers.add_parser(
        "roi",
        help="Building ROI analysis",
        description="Analyze return on investment for buildings",
    )
    roi_parser.add_argument(
        "-b", "--building", type=str, nargs="+", help="Filter by building name"
    )
    roi_parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=20,
        dest="max_level",
        help="Maximum building level (default: 20)",
    )
    roi_parser.add_argument(
        "-p",
        "--per-step",
        action="store_true",
        dest="step_roi",
        help="Calculate per-upgrade-step ROI",
    )

    # lifecycle subcommand
    lifecycle_parser = subparsers.add_parser(
        "lifecycle",
        help="Abundance decay/lifecycle analysis",
        description="Calculate lifecycle ROI for abundance resources",
    )
    lifecycle_parser.add_argument(
        "-b", "--building", type=str, nargs="+", help="Filter by building name"
    )
    lifecycle_parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=20,
        dest="max_level",
        help="Maximum building level (default: 20)",
    )
    lifecycle_parser.add_argument(
        "-t",
        "--time",
        type=float,
        default=0.0,
        dest="build_time",
        help="Base build time in hours",
    )

    # prospect subcommand
    prospect_parser = subparsers.add_parser(
        "prospect",
        help="Prospecting simulation",
        description="Simulate prospecting to find target abundance",
    )
    prospect_parser.add_argument(
        "-t",
        "--target",
        type=float,
        required=True,
        dest="abundance",
        help="Target abundance percentage",
    )
    prospect_parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=12,
        dest="time",
        help="Build time per attempt in hours (default: 12)",
    )
    prospect_parser.add_argument(
        "-s",
        "--slots",
        type=int,
        default=1,
        help="Number of building slots (default: 1)",
    )

    # debug subcommand
    debug_parser = subparsers.add_parser(
        "debug",
        help="Debugging utilities",
        description="Various debugging and diagnostic tools",
    )
    debug_parser.add_argument(
        "-u",
        "--unassigned",
        action="store_true",
        dest="debug_unassigned",
        help="List resources not assigned to any building",
    )

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare market vs contract sales",
        description="Compare selling on the market vs selling via contracts with a custom contract price",
    )
    compare_parser.add_argument(
        "-s",
        "--search",
        type=str,
        nargs="+",
        required=True,
        help="Search resources by name (case-insensitive)",
    )
    compare_parser.add_argument(
        "-p",
        "--price",
        type=float,
        required=True,
        dest="contract_price",
        help="Contract price per unit (e.g., 97.5)",
    )

    # scenario subcommand (compare multiple custom price/quality pairs)
    scenario_parser = subparsers.add_parser(
        "scenario",
        help="Compare custom price/quality scenarios for production and retail",
        description=(
            "Evaluate profit/revenue for a resource across multiple (quality, price) pairs "
            "with optional retail/production filtering"
        ),
    )
    scenario_parser.add_argument(
        "resource",
        type=str,
        nargs="+",
        help="Resource name(s) (case-insensitive)",
    )
    scenario_parser.add_argument(
        "-q",
        "--quality",
        dest="scenario_quality",
        type=int,
        action="append",
        required=True,
        help="Quality for a scenario (repeatable, order must match --price)",
    )
    scenario_parser.add_argument(
        "-p",
        "--price",
        dest="scenario_price",
        type=float,
        action="append",
        required=True,
        help="Price for a scenario (repeatable, order must match --quality)",
    )
    scenario_parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=1,
        dest="building_level",
        help="Building level to use (default: 1)",
    )
    scenario_parser.add_argument(
        "-R",
        "--retail",
        dest="retail_only",
        action="store_true",
        help="Show only retail scenarios",
    )
    scenario_parser.add_argument(
        "-P",
        "--production",
        dest="production_only",
        action="store_true",
        help="Show only production scenarios",
    )

    # genetic subcommand
    genetic_parser = subparsers.add_parser(
        "genetic",
        help="Genetic algorithm optimization",
        description="Use genetic algorithm to find optimal building configuration for maximum profit",
    )
    genetic_parser.add_argument(
        "-s",
        "--slots",
        type=int,
        default=5,
        help="Number of building slots (default: 5)",
    )
    genetic_parser.add_argument(
        "-b",
        "--budget",
        type=float,
        default=100000,
        help="Maximum investment budget (default: 100000)",
    )
    genetic_parser.add_argument(
        "-p",
        "--population",
        type=int,
        default=50,
        dest="population_size",
        help="Population size (default: 50)",
    )
    genetic_parser.add_argument(
        "-g",
        "--generations",
        type=int,
        default=100,
        help="Number of generations (default: 100)",
    )
    genetic_parser.add_argument(
        "-m",
        "--mutation-rate",
        type=float,
        default=0.1,
        dest="mutation_rate",
        help="Mutation rate 0.0-1.0 (default: 0.1)",
    )
    genetic_parser.add_argument(
        "-x",
        "--crossover-rate",
        type=float,
        default=0.7,
        dest="crossover_rate",
        help="Crossover rate 0.0-1.0 (default: 0.7)",
    )
    genetic_parser.add_argument(
        "-l",
        "--max-level",
        type=int,
        default=10,
        dest="max_level",
        help="Maximum building level (default: 10)",
    )
    genetic_parser.add_argument(
        "-t",
        "--tournament-size",
        type=int,
        default=3,
        dest="tournament_size",
        help="Tournament size for selection (default: 3)",
    )
    genetic_parser.add_argument(
        "--elitism",
        type=int,
        default=2,
        help="Number of best individuals to preserve (default: 2)",
    )
    genetic_parser.add_argument(
        "--budget-penalty",
        type=float,
        default=2.0,
        dest="budget_penalty_factor",
        help="Penalty factor for budget overage (default: 2.0)",
    )

    # retail subcommand
    retail_parser = subparsers.add_parser(
        "retail",
        help="Retail sales calculation",
        description="Calculate units sold per hour for retail sales and profit analysis",
    )
    retail_parser.add_argument(
        "-i",
        "--item",
        type=str,
        nargs="+",
        required=False,
        dest="item_name",
        help="Resource name(s) to calculate retail for (case-insensitive). If not provided, shows ordered list of all retail items",
    )
    retail_parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=1,
        dest="building_level",
        help="Building level (default: 1)",
    )
    # analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze",
        parents=[parent_parser],
        help="Interactive company analysis",
        description="Analyze a player's company setup and provide upgrade recommendations",
    )
    analyze_parser.add_argument(
        "-u",
        "--user-id",
        type=int,
        required=True,
        dest="user_id",
        help="User ID to fetch company data for",
    )
    analyze_parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=10,
        dest="top_n",
        help="Number of upgrade recommendations to show (default: 10)",
    )

    args = parser.parse_args()
    
    # Set default command to 'profit' if none specified
    if args.command is None:
        args.command = "profit"
    
    return args


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    # Handle prospecting simulation
    if args.command == "prospect":
        results = simulate_prospecting(args.abundance / 100, args.time, args.slots)
        display_prospecting_results(results)
        return

    # Handle retail command
    if args.command == "retail":
        # Load data files
        abundance_resources = load_json_list(get_data_path("abundance_resources.json"))
        seasonal_resources = load_json_list(get_data_path("seasonal_resources.json"))
        buildings = Building.load_all(get_data_path("buildings.json"), include_retail=True)
        resource_to_building = build_resource_to_building_map(buildings)

        # Fetch API data
        api = SimcoAPI(realm=0)
        try:
            resources_data = api.get_resources()
            raw_resources = resources_data.get("resources", [])
            
            vwaps_data = api.get_market_vwaps()
            market = MarketData.from_api_response(
                vwaps_data=vwaps_data,
                resources_data=raw_resources,
            )

            # Fetch retail info from API
            retail_info_api_list = api.get_retail_info()
            retail_info_api = {}
            for r in retail_info_api_list:
                if isinstance(r, dict) and "dbLetter" in r:
                    retail_info_api[r["dbLetter"]] = r

            # Create Resource objects
            resources = [
                Resource.from_api_data(
                    data,
                    abundance_resources=[r.lower() for r in abundance_resources],
                    seasonal_resources=[r.lower() for r in seasonal_resources],
                )
                for data in raw_resources
            ]

            # Link resources to buildings
            resource_by_name = {r.name.lower(): r for r in resources}
            for building in buildings:
                building.link_resources(resource_by_name)

            # Set building names on resources
            for res in resources:
                building_name = resource_to_building.get(res.name.lower())
                if building_name:
                    res.building_name = building_name

            # If -i/--item specified, show details for specific items
            if hasattr(args, 'item_name') and args.item_name:
                item_names_lower = [name.lower() for name in args.item_name]
                for item_name_lower in item_names_lower:
                    # Find resource
                    resource = resource_by_name.get(item_name_lower)
                    if not resource:
                        console.print(f"[bold red]Resource '{item_name_lower}' not found.[/bold red]")
                        continue

                    if not resource.retail_info:
                        console.print(f"[bold red]{resource.name} has no retail info.[/bold red]")
                        continue

                    # Get retail info for quality
                    retail_info_entry = None
                    for ri in resource.retail_info:
                        if ri.get("quality") == args.quality:
                            retail_info_entry = ri.copy()
                            break
                    
                    if not retail_info_entry:
                        console.print(f"[bold red]No retail info for {resource.name} at Q{args.quality}.[/bold red]")
                        continue

                    # Merge with API data
                    api_data = retail_info_api.get(resource.id, {})
                    if api_data:
                        retail_info_entry.update({
                            "saturation": api_data.get("saturation", retail_info_entry.get("saturation", 1.0)),
                            "averagePrice": api_data.get("averagePrice", retail_info_entry.get("averagePrice", 0)),
                        })

                    retail_price = retail_info_entry.get("averagePrice", 0)
                    if retail_price <= 0:
                        console.print(f"[bold red]No retail price for {resource.name}.[/bold red]")
                        continue

                    input_cost = market.get_price(resource.id, args.quality)
                    building_level = getattr(args, 'building_level', 1)

                    # Calculate units per hour
                    units_per_hour = calculate_retail_units_per_hour(
                        retail_info=retail_info_entry,
                        price=retail_price,
                        quality=args.quality,
                        building_level=building_level,
                        sales_speed_bonus=getattr(args, 'sales_speed_bonus', 0.0),
                        acceleration_multiplier=getattr(args, 'acceleration_multiplier', 1.0),
                        weather_multiplier=getattr(args, 'weather_multiplier', 1.0),
                    )

                    # Calculate profit
                    sales_wages = retail_info_entry.get("salesWages", 0)
                    admin_overhead = getattr(args, 'admin_overhead', 0.0)
                    wages_per_hour = sales_wages * building_level * (1.0 + admin_overhead / 100.0)
                    revenue_per_hour = retail_price * units_per_hour
                    costs_per_hour = wages_per_hour + (input_cost * units_per_hour)
                    profit_per_hour = revenue_per_hour - costs_per_hour

                    # Display detailed results
                    display_retail_results(
                        resource_name=resource.name,
                        units_per_hour=units_per_hour,
                        price=retail_price,
                        quality=args.quality,
                        building_level=building_level,
                        sales_speed_bonus=getattr(args, 'sales_speed_bonus', 0.0),
                        retail_info=retail_info_entry,
                    )
                    console.print(f"\n[bold cyan]Profit Information:[/bold cyan]")
                    console.print(f"  Input Cost: ${input_cost:.2f}")
                    console.print(f"  Profit/Hour: ${profit_per_hour:,.2f}")
                    console.print(f"  Profit/Day: ${profit_per_hour * 24:,.2f}\n")
            else:
                # Show ordered list of all retail items by PPH
                config = ProfitConfig(
                    quality=args.quality,
                    abundance=getattr(args, 'abundance', 90.0),
                    admin_overhead=getattr(args, 'admin_overhead', 0.0),
                    is_contract=False,
                    has_robots=getattr(args, 'robots', False),
                    sales_speed_bonus=getattr(args, 'sales_speed_bonus', 0.0) / 100.0,
                )

                retail_profits = []
                for res in resources:
                    if not res.retail_info:
                        continue

                    retail_info_entry = None
                    for ri in res.retail_info:
                        if ri.get("quality") == args.quality:
                            retail_info_entry = ri.copy()
                            break
                    
                    if not retail_info_entry:
                        continue

                    api_data = retail_info_api.get(res.id, {})
                    if not api_data:
                        continue

                    # Skip restaurant items
                    if api_data.get('retailData', [{}])[0].get('amountSoldRestaurant', 0) > 0:
                        continue

                    retail_info_entry.update({
                        "saturation": api_data.get("saturation", retail_info_entry.get("saturation", 1.0)),
                        "averagePrice": api_data.get("averagePrice", retail_info_entry.get("averagePrice", 0)),
                    })

                    retail_price = retail_info_entry.get("averagePrice", 0)
                    if retail_price <= 0:
                        continue

                    input_cost = market.get_price(res.id, args.quality)
                    building_level = getattr(args, 'building_level', 1)

                    retail_profit = res.calculate_retail_profit(
                        market=market,
                        retail_data=retail_info_entry | api_data,
                        quality=args.quality,
                        building_level=building_level,
                        sales_speed_bonus=config.sales_speed_bonus,
                        admin_overhead=config.admin_overhead,
                        input_cost_per_unit=input_cost,
                    )
                    retail_profits.append(retail_profit)

                # Sort by PPH
                retail_profits.sort(key=lambda x: x["profit_per_hour"], reverse=True)

                # Display table
                display_retail_table(retail_profits, config, building_level=getattr(args, 'building_level', 1))
        except Exception as exc:
            console.print(f"[bold red]Error: {exc}[/bold red]")
            raise
        return

    # Handle debug commands
    if args.command == "debug":
        if hasattr(args, "debug_unassigned") and args.debug_unassigned:
            # Load data files
            buildings = Building.load_all(get_data_path("buildings.json"))
            resource_to_building = build_resource_to_building_map(buildings)
            
            # Fetch API data
            api = SimcoAPI(realm=0)
            try:
                resources_data = api.get_resources()
                raw_resources = resources_data.get("resources", [])
                
                unassigned = [
                    res.get("name")
                    for res in raw_resources
                    if res.get("name", "").lower() not in resource_to_building
                ]
                unassigned.sort()
                console.print("\n[bold red]Resources not assigned to any building:[/bold red]")
                for name in unassigned:
                    console.print(f" - {name}")
            except Exception as exc:
                console.print(f"[bold red]Error fetching data: {exc}[/bold red]")
                raise
        else:
            # No debug option specified, show help
            console.print("[yellow]No debug option specified. Use -u/--unassigned[/yellow]")
        return

    # For all other commands (profit, roi, lifecycle), we need full data
    # Load data files
    abundance_resources = load_json_list(get_data_path("abundance_resources.json"))
    seasonal_resources = load_json_list(get_data_path("seasonal_resources.json"))
    buildings = Building.load_all(get_data_path("buildings.json"))
    resource_to_building = build_resource_to_building_map(buildings)

    # Fetch API data
    api = SimcoAPI(realm=0)

    try:
        resources_data = api.get_resources()
        raw_resources = resources_data.get("resources", [])

        vwaps_data = api.get_market_vwaps()

        # Try to fetch retail info (optional, may fail)
        temp_retail_data = api.get_retail_info()
        retail_info_api = {}
        for r in temp_retail_data:
            if isinstance(r, dict) and "dbLetter" in r:
                retail_info_api[r["dbLetter"]] = r
        # Build MarketData from API response
        market = MarketData.from_api_response(
            vwaps_data=vwaps_data,
            resources_data=raw_resources,
        )

        # Create Resource objects
        resources = [
            Resource.from_api_data(
                data,
                abundance_resources=[r.lower() for r in abundance_resources],
                seasonal_resources=[r.lower() for r in seasonal_resources],
            )
            for data in raw_resources
        ]

        # Link resources to buildings
        resource_by_name = {r.name.lower(): r for r in resources}
        for building in buildings:
            building.link_resources(resource_by_name)

        # Set building names on resources
        for res in resources:
            building_name = resource_to_building.get(res.name.lower())
            if building_name:
                res.building_name = building_name

        # Filter resources
        filtered_resources = resources

        # Exclude seasonal if requested (available across all commands via parent parser)
        if hasattr(args, "exclude_seasonal") and args.exclude_seasonal:
            filtered_resources = [r for r in filtered_resources if not r.is_seasonal]

        # Filter by building
        if hasattr(args, "building") and args.building:
            filtered_resources = [
                r
                for r in filtered_resources
                if r.building_name
                and any(term.lower() in r.building_name.lower() for term in args.building)
            ]

        # Filter by search terms (only for profit command)
        if args.command == "profit" and hasattr(args, "search") and args.search:
            filtered_resources = [
                r
                for r in filtered_resources
                if any(term.lower() in r.name.lower() for term in args.search)
            ]

        # Create lookup for filtered resources (used by analyze and genetic commands)
        filtered_resource_by_name = {r.name.lower(): r for r in filtered_resources}

        # Calculate profits
        config = ProfitConfig(
            quality=args.quality,
            abundance=args.abundance,
            admin_overhead=args.admin_overhead,
            is_contract=args.contract,
            has_robots=args.robots,
            sales_speed_bonus=args.sales_speed_bonus / 100.0,  # Convert percentage to decimal
        )

        # Process temp_retail_data: it's already in the right format (dbLetter -> retail_info_dict)
        # Just pass it as temp parameter (raw retail data dict)
        profits = calculate_all_profits(filtered_resources, market, args.quality, config, retail_data=retail_info_api)

        # Handle scenario command (custom quality/price comparisons)
        if args.command == "scenario":
            qualities = args.scenario_quality or []
            prices = args.scenario_price or []
            if len(qualities) != len(prices):
                console.print("[bold red]Error: provide the same number of -q/--quality and -p/--price values.[/bold red]")
                return

            building_level = getattr(args, "building_level", 1)
            retail_only = getattr(args, "retail_only", False)
            production_only = getattr(args, "production_only", False)

            rows: list[dict] = []
            scenario_pairs = list(zip(qualities, prices))

            for name in [n.lower() for n in args.resource]:
                res = resource_by_name.get(name)
                if not res:
                    console.print(f"[yellow]Warning: resource '{name}' not found.[/yellow]")
                    continue

                # Production scenarios
                if not retail_only:
                    for quality, price in scenario_pairs:
                        prod = res.calculate_profit(
                            selling_price=price,
                            market=market,
                            quality=quality,
                            abundance=args.abundance,
                            admin_overhead=args.admin_overhead,
                            is_contract=args.contract,
                            has_robots=args.robots,
                        )
                        rows.append(
                            {
                                "mode": "Production",
                                "resource": res.name,
                                "quality": quality,
                                "price": price,
                                "profit_per_hour": prod["profit_per_hour"] * building_level,
                                "revenue_per_hour": prod["revenue_per_hour"] * building_level,
                                "costs_per_hour": prod["costs_per_hour"] * building_level,
                                "fee_per_hour": prod["market_fee_per_hour"] * building_level,
                                "transport_per_hour": prod["transport_costs_per_hour"] * building_level,
                                "units_per_hour": res.get_effective_production(args.abundance) * building_level,
                            }
                        )

                # Retail scenarios
                if not production_only and res.retail_info:
                    for quality, price in scenario_pairs:
                        retail_info_entry = None
                        for ri in res.retail_info:
                            if ri.get("quality") == quality:
                                retail_info_entry = ri.copy()
                                break

                        if not retail_info_entry:
                            continue

                        api_data = retail_info_api.get(res.id, {})
                        retail_info_entry.update({
                            "saturation": api_data.get("saturation", retail_info_entry.get("saturation", 1.0)),
                            "averagePrice": price,
                        })

                        input_cost = market.get_price(res.id, quality)

                        units_per_hour = calculate_retail_units_per_hour(
                            retail_info=retail_info_entry,
                            price=price,
                            quality=quality,
                            building_level=building_level,
                            sales_speed_bonus=getattr(args, "sales_speed_bonus", 0.0),
                            acceleration_multiplier=getattr(args, "acceleration_multiplier", 1.0) if hasattr(args, "acceleration_multiplier") else 1.0,
                            weather_multiplier=getattr(args, "weather_multiplier", 1.0) if hasattr(args, "weather_multiplier") else 1.0,
                        )

                        sales_wages = retail_info_entry.get("salesWages", 0)
                        wages_per_hour = sales_wages * building_level * (1.0 + args.admin_overhead / 100.0)
                        revenue_per_hour = price * units_per_hour
                        costs_per_hour = wages_per_hour + (input_cost * units_per_hour)
                        profit_per_hour = revenue_per_hour - costs_per_hour

                        rows.append(
                            {
                                "mode": "Retail",
                                "resource": res.name,
                                "quality": quality,
                                "price": price,
                                "profit_per_hour": profit_per_hour,
                                "revenue_per_hour": revenue_per_hour,
                                "costs_per_hour": costs_per_hour,
                                "fee_per_hour": 0.0,
                                "transport_per_hour": 0.0,
                                "units_per_hour": units_per_hour,
                            }
                        )

            rows.sort(key=lambda x: x["profit_per_hour"], reverse=True)
            display_scenario_table(rows, building_level)
            return

        # Handle lifecycle command
        if args.command == "lifecycle":
            # Lifecycle Analysis
            abundance_res_objects = [r for r in filtered_resources if r.is_abundance]
            
            lifecycle_results = []
            for res in abundance_res_objects:
                # Find the building
                if not res.building_name:
                    continue
                # We need the building object
                building = next((b for b in buildings if b.name == res.building_name), None)
                if not building:
                    continue

                res_results = calculate_lifecycle_roi(
                    building=building,
                    resource=res,
                    profit_config=config,
                    market=market,
                    quality=args.quality,
                    start_abundance=args.abundance / 100.0,
                    max_level=args.max_level,
                    base_build_time=args.build_time,
                )
                lifecycle_results.extend(res_results)
            
            # Sort by Net Profit
            lifecycle_results.sort(key=lambda x: x["net_profit"], reverse=True)
            
            display_lifecycle_table(lifecycle_results, args.abundance)
            return

        # Handle ROI command
        if args.command == "roi":
            if hasattr(args, "building") and args.building:
                # Generate level-based ROI for filtered buildings
                res_profit_map = {p["name"].lower(): p for p in profits}
                all_roi_data = []
                for building in buildings:
                    best_profit = -float("inf")
                    best_p_data = None
                    for res_name in building.produces:
                        res_name_lower = res_name.lower()
                        if res_name_lower in res_profit_map:
                            p_data = res_profit_map[res_name_lower]
                            if p_data["profit_per_hour"] > best_profit:
                                best_profit = p_data["profit_per_hour"]
                                best_p_data = p_data
                    
                    if best_p_data:
                        all_roi_data.extend(
                            calculate_level_roi(
                                building,
                                best_p_data,
                                market,
                                max_level=args.max_level,
                                step_mode=args.step_roi,
                            )
                        )
                display_roi_table(all_roi_data)
            else:
                roi_data = calculate_building_roi(buildings, profits, market)
                display_roi_table(roi_data)
            return

        # Handle profit command (default display)
        if args.command == "profit":
            building_level = getattr(args, "building_level", 1)
            display_profits_table(
                profits, 
                market.transport_price, 
                config,
                building_level=building_level,
                search_terms=args.search if hasattr(args, "search") else None, 
                building_terms=args.building if hasattr(args, "building") else None
            )

        # Handle compare command
        if args.command == "compare":
            # Filter resources by search terms
            search_filtered = [
                r
                for r in filtered_resources
                if any(term.lower() in r.name.lower() for term in args.search)
            ]

            if not search_filtered:
                console.print(
                    f"[bold red]No resources found matching search terms: {', '.join(args.search)}[/bold red]"
                )
                return

            # Calculate comparisons for each resource
            comparisons = []
            for res in search_filtered:
                market_price = market.get_price(res.id, args.quality)
                if market_price == 0:
                    console.print(
                        f"[yellow]Warning: No market price found for {res.name} at Quality {config.quality}[/yellow]"
                    )
                    continue

                comparison = compare_market_vs_contract(
                    resource=res,
                    market_price=market_price,
                    contract_price=args.contract_price,
                    market=market,
                    quality=args.quality,
                    config=config,
                )
                comparisons.append(comparison)

            if comparisons:
                display_compare_table(comparisons, market.transport_price, config)
            else:
                console.print(
                    f"[bold red]No valid comparisons could be made. Check that resources have market prices at Quality {config.quality}[/bold red]"
                )

        # Handle genetic algorithm command
        if args.command == "genetic":
            # Create simulation config
            sim_config = SimulationConfig(
                slots=args.slots,
                budget=args.budget,
                population_size=args.population_size,
                generations=args.generations,
                mutation_rate=args.mutation_rate,
                crossover_rate=args.crossover_rate,
                max_level=args.max_level,
                elitism=args.elitism,
                tournament_size=args.tournament_size,
                budget_penalty_factor=args.budget_penalty_factor,
            )

            # Create and run genetic algorithm
            ga = GeneticAlgorithm(
                config=sim_config,
                buildings=buildings,
                resources=filtered_resources,
                market=market,
                quality=args.quality,
                abundance=args.abundance,
                admin_overhead=args.admin_overhead,
                has_robots=args.robots,
            )

            console.print("\n[bold blue]Starting Genetic Algorithm Optimization...[/bold blue]\n")

            # Run with progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Evolving population...",
                    total=sim_config.generations,
                )

                def update_progress(gen: int, best: float, avg: float) -> None:
                    progress.update(
                        task,
                        completed=gen,
                        description=f"[cyan]Gen {gen}/{sim_config.generations} | Best: ${best:,.0f} | Avg: ${avg:,.0f}",
                    )

                best_individual, fitness_history = ga.run(progress_callback=update_progress)

            # Display results
            display_genetic_results(
                best_individual,
                fitness_history,
                sim_config,
                ga,
            )

        # Handle analyze command
        if args.command == "analyze":
            # Fetch company data
            company_data = api.get_company(args.user_id)
            company = company_data.get("company", {})

            if not company:
                console.print(
                    f"[bold red]No company data found for user ID {args.user_id}[/bold red]"
                )
                return

            # Get buildings from company data
            company_buildings = company.get("buildings", {})

            if not company_buildings:
                console.print(
                    "[bold red]No buildings found in company data[/bold red]"
                )
                return

            # Prompt user for building levels
            building_levels = prompt_building_levels(company_buildings, buildings)

            # Create building ID to Building object lookup
            id_to_building = {b.id: b for b in buildings}

            # Create building resources lookup using filtered resources
            # This respects the -e flag to exclude seasonal resources
            building_resources: dict[str, list[Resource]] = {}
            for building in buildings:
                res_list = []
                for res_name in building.produces:
                    res = filtered_resource_by_name.get(res_name.lower())
                    if res:
                        res_list.append(res)
                if res_list:
                    building_resources[building.name] = res_list

            # Calculate building stats
            building_stats = []
            buildings_with_levels: list[tuple[Building, int]] = []

            for building_key, level in building_levels.items():
                # Handle both single and multiple buildings of same type
                # building_key might be "E" or "E_0", "E_1" etc.
                if "_" in building_key:
                    building_id = building_key.rsplit("_", 1)[0]
                else:
                    building_id = building_key

                building = id_to_building.get(building_id)
                if not building:
                    console.print(
                        f"[yellow]Warning: Unknown building ID '{building_id}'[/yellow]"
                    )
                    continue

                building_res = building_resources.get(building.name, [])

                stat = calculate_company_building_stats(
                    building=building,
                    level=level,
                    resources=building_res,
                    market=market,
                    quality=args.quality,
                    config=config,
                )
                building_stats.append(stat)
                buildings_with_levels.append((building, level))

            # Display company analysis
            display_company_analysis(company_data, building_stats, config)

            # Calculate and display upgrade recommendations
            recommendations = calculate_upgrade_recommendations(
                buildings_with_levels=buildings_with_levels,
                building_resources=building_resources,
                market=market,
                quality=args.quality,
                config=config,
            )

            display_upgrade_recommendations(recommendations, config, top_n=args.top_n)

    except Exception as exc:
        console.print(f"[bold red]Error fetching data: {exc}[/bold red]")
        raise


if __name__ == "__main__":
    main()
