"""Command-line interface for simtools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from simtools.api import SimcoAPI
from simtools.calculator import (
    ProfitConfig,
    calculate_all_profits,
    calculate_building_roi,
    calculate_level_roi,
    calculate_lifecycle_roi,
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


def save_json(data, filename: str) -> None:
    """Save data to a JSON file in the workspace root.

    Args:
        data: Data to save.
        filename: Name of the output file.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    console.log(f"Data saved to [cyan]{filename}[/cyan]")


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
    search_terms: list[str] | None = None,
    building_terms: list[str] | None = None,
) -> None:
    """Display the profits table.

    Args:
        profits: List of profit dictionaries.
        transport_price: Price per transport unit.
        config: Profit calculation configuration.
        search_terms: Search terms used for filtering (for header).
        building_terms: Building terms used for filtering (for header).
    """
    # Build header title
    header_title = "Top 30 Most Profitable Resources"
    if search_terms or building_terms:
        parts = []
        if search_terms:
            parts.append(f"search: '{', '.join(search_terms)}'")
        if building_terms:
            parts.append(f"building: '{', '.join(building_terms)}'")
        header_title = f"Results for {' & '.join(parts)}"

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

        table.add_row(
            f"{p['name']}{abundance_mark}",
            f"[{profit_style}]${p['profit_per_hour']:,.2f}[/{profit_style}]",
            f"${p['revenue_per_hour']:,.2f}",
            f"${p['market_fee_per_hour']:,.2f}",
            f"${p['costs_per_hour']:,.2f}",
            f"${p['transport_costs_per_hour']:,.2f}{warn}",
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

    # Main parser
    parser = argparse.ArgumentParser(
        description="Simtools - Sim Companies calculation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Add common flags at top level for backwards compatibility
    # NOTE: These are intentionally duplicated from parent_parser to support
    # usage without subcommands (e.g., "simtools -a 85" defaults to profit)
    parser.add_argument(
        "-q", "--quality", type=int, default=0, help="Quality level (default: 0)"
    )
    parser.add_argument(
        "-a",
        "--abundance",
        type=float,
        default=90,
        help="Abundance percentage for mine/well resources (default: 90)",
    )
    parser.add_argument(
        "-c",
        "--contract",
        action="store_true",
        help="Direct contract mode (0%% market fee, 50%% transport)",
    )
    parser.add_argument(
        "-r",
        "--robots",
        action="store_true",
        help="Apply 3%% wage reduction",
    )
    parser.add_argument(
        "-o",
        "--overhead",
        type=float,
        default=0,
        dest="admin_overhead",
        help="Admin overhead percentage (default: 0)",
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
    parser.add_argument(
        "-e",
        "--no-seasonal",
        action="store_true",
        dest="exclude_seasonal",
        help="Exclude seasonal resources",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Make the subparser optional by setting required=False
    subparsers.required = False

    # profit subcommand - default/main functionality
    profit_parser = subparsers.add_parser(
        "profit",
        parents=[parent_parser],
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

    # roi subcommand
    roi_parser = subparsers.add_parser(
        "roi",
        parents=[parent_parser],
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
        parents=[parent_parser],
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
        parents=[parent_parser],
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

    # genetic subcommand
    genetic_parser = subparsers.add_parser(
        "genetic",
        parents=[parent_parser],
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

        # Save API data
        save_json(resources_data, "resources.json")
        save_json(vwaps_data, "vwaps.json")

        # Build price maps
        price_map: dict[int, float] = {}
        q0_price_map: dict[int, float] = {}

        if isinstance(vwaps_data, list):
            for entry in vwaps_data:
                if isinstance(entry, dict):
                    r_id = entry.get("resourceId")
                    quality = entry.get("quality")
                    vwap = entry.get("vwap")
                    if r_id is not None and vwap is not None:
                        if quality == args.quality:
                            price_map[int(r_id)] = vwap
                        if quality == 0:
                            q0_price_map[int(r_id)] = vwap

        # Build name to ID map
        name_to_id = {r.get("name", "").lower(): r.get("id") for r in raw_resources}

        # Get transport price
        transport_id = None
        for res in raw_resources:
            if res.get("name", "").lower() == "transport":
                transport_id = res.get("id")
                break

        if transport_id is None:
            for res in raw_resources:
                if "transport" in res.get("name", "").lower():
                    transport_id = res.get("id")
                    break

        transport_price = 0.0
        if transport_id is not None:
            if isinstance(vwaps_data, list):
                for entry in vwaps_data:
                    if (
                        isinstance(entry, dict)
                        and entry.get("resourceId") == transport_id
                        and entry.get("quality") == 0
                    ):
                        transport_price = entry.get("vwap", 0)
                        break
        else:
            console.print("[yellow]Warning: Could not find 'Transport' resource by name.[/yellow]")

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

        # Calculate profits
        config = ProfitConfig(
            quality=args.quality,
            abundance=args.abundance,
            admin_overhead=args.admin_overhead,
            is_contract=args.contract,
            has_robots=args.robots,
        )

        profits = calculate_all_profits(filtered_resources, price_map, transport_price, config)

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
                    current_prices=price_map,
                    q0_prices=q0_price_map,
                    transport_price=transport_price,
                    name_to_id=name_to_id,
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
                                q0_price_map,
                                name_to_id,
                                max_level=args.max_level,
                                step_mode=args.step_roi,
                            )
                        )
                display_roi_table(all_roi_data)
            else:
                roi_data = calculate_building_roi(buildings, profits, q0_price_map, name_to_id)
                display_roi_table(roi_data)
            return

        # Handle profit command (default display)
        if args.command == "profit":
            display_profits_table(
                profits, 
                transport_price, 
                config, 
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
                market_price = price_map.get(res.id, 0)
                if market_price == 0:
                    console.print(
                        f"[yellow]Warning: No market price found for {res.name} at Quality {config.quality}[/yellow]"
                    )
                    continue

                comparison = compare_market_vs_contract(
                    resource=res,
                    market_price=market_price,
                    contract_price=args.contract_price,
                    input_prices=price_map,
                    transport_price=transport_price,
                    config=config,
                )
                comparisons.append(comparison)

            if comparisons:
                display_compare_table(comparisons, transport_price, config)
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
                price_map=price_map,
                q0_price_map=q0_price_map,
                transport_price=transport_price,
                name_to_id=name_to_id,
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

    except Exception as exc:
        console.print(f"[bold red]Error fetching data: {exc}[/bold red]")
        raise


if __name__ == "__main__":
    main()
