import httpx
import json
import argparse
import os
import random
import math
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import box

console = Console()

class SimcoAPI:
    def __init__(self, realm: int = 0):
        self.base_url = f"https://api.simcotools.com/v1/realms/{realm}"
        self.headers = {"accept": "application/json"}

    def get_resources(self):
        url = f"{self.base_url}/resources"
        all_resources = []
        
        with console.status("[bold green]Fetching resources...", spinner="dots"):
            # Try disable_pagination=True first
            try:
                response = httpx.get(url, headers=self.headers, params={"disable_pagination": "True"})
                response.raise_for_status()
                data = response.json()
                
                # If disable_pagination worked, it might return a list or a dict with all resources
                if isinstance(data, list):
                    return {"resources": data}
                if isinstance(data, dict) and "resources" in data:
                    # Check if totalRecords matches len(resources)
                    metadata = data.get("metadata", {})
                    total_records = metadata.get("totalRecords")
                    if total_records is not None and len(data["resources"]) >= total_records:
                        console.log(f"Fetched all {len(data['resources'])} resources with disable_pagination.")
                        return data
            except Exception as e:
                console.log(f"[yellow]disable_pagination failed, falling back to manual pagination: {e}[/yellow]")

            # Fallback to manual pagination
            current_page = 1
            last_page = 1
            
            while current_page <= last_page:
                response = httpx.get(url, headers=self.headers, params={"page": current_page})
                response.raise_for_status()
                data = response.json()
                
                resources = data.get("resources", [])
                all_resources.extend(resources)
                
                metadata = data.get("metadata", {})
                current_page = metadata.get("currentPage", 1) + 1
                last_page = metadata.get("lastPage", 1)
                
            console.log(f"Fetched {len(all_resources)} resources via manual pagination.")
            return {"resources": all_resources}

    def get_market_vwaps(self):
        url = f"{self.base_url}/market/vwaps"
        with console.status("[bold green]Fetching market VWAPs...", spinner="dots"):
            response = httpx.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            # If the API returns a dict with a key like 'vwaps'
            if isinstance(data, dict) and "vwaps" in data:
                return data["vwaps"]
            return data

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    console.log(f"Data saved to [cyan]{filename}[/cyan]")

def simulate_prospecting(target_abundance, attempt_time, slots=1):
    """
    Simulates prospecting based on the Gaussian roll formula.
    Formula: min(1.0, max(0.1, random.gauss(mu=0.6, sigma=0.15)))
    """
    mu = 0.6
    sigma = 0.15
    
    # We want to find the probability P(X >= target_abundance)
    # The distribution is a truncated/clamped Gaussian.
    # For target_abundance <= 0.1, probability is 1.0.
    # For target_abundance > 1.0, probability is 0.0.
    
    if target_abundance <= 0.1:
        p_success_single = 1.0
    elif target_abundance > 1.0:
        p_success_single = 0.0
    else:
        # P(X >= target) where X ~ N(mu, sigma)
        # P(X >= target) = 1 - P(X < target) = 1 - Phi((target - mu) / sigma)
        # where Phi is the CDF of standard normal distribution.
        z = (target_abundance - mu) / sigma
        p_success_single = 1.0 - (0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
        
    if p_success_single <= 0:
        console.print(f"[bold red]Target abundance {target_abundance*100:.1f}% is impossible with the current distribution.[/bold red]")
        return

    # Probability of at least one success in a block of 'slots' attempts
    p_success_block = 1.0 - (1.0 - p_success_single)**slots

    expected_blocks = 1.0 / p_success_block
    expected_time = expected_blocks * attempt_time

    # Use a fixed width for both tables to make them look uniform
    table_width = 60

    table = Table(title="Prospecting Simulation Results", show_header=True, header_style="bold magenta", box=box.ROUNDED, width=table_width)
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Target Abundance", f"{target_abundance*100:.1f}%")
    table.add_row("Build Time per Attempt", f"{attempt_time:.1f} hours")
    table.add_row("Number of Slots", f"{slots}")
    table.add_row("Prob. Success (Single)", f"{p_success_single*100:.4f}%")
    table.add_row("Prob. Success (Block)", f"{p_success_block*100:.4f}%")
    table.add_row("Expected Blocks", f"{expected_blocks:.2f}")
    table.add_row("Expected Time", f"{expected_time:.2f} hours ({expected_time/24:.2f} days)")

    console.print(table)

    # Confidence Intervals
    # The number of blocks until success follows a Geometric distribution.
    # P(Success within n blocks) = 1 - (1 - p_block)^n
    # Solving for n: n = log(1 - confidence) / log(1 - p_block)
    
    conf_table = Table(title="Confidence Intervals (Time to Success)", show_header=True, header_style="bold blue", box=box.ROUNDED, width=table_width)
    conf_table.add_column("Confidence Level", style="cyan")
    conf_table.add_column("Required Blocks", justify="right", style="yellow")
    conf_table.add_column("Required Time", justify="right", style="green")

    if p_success_block < 1.0:
        for conf in [0.50, 0.80, 0.90, 0.95, 0.99]:
            n_blocks = math.ceil(math.log(1.0 - conf) / math.log(1.0 - p_success_block))
            n_time = n_blocks * attempt_time
            conf_table.add_row(f"{conf*100:.0f}%", f"{n_blocks}", f"{n_time:.1f}h ({n_time/24:.1f}d)")
    else:
        conf_table.add_row("100%", "1", f"{attempt_time:.1f}h")

    console.print(conf_table)

def main():
    parser = argparse.ArgumentParser(description="Simcotools calculation script")
    parser.add_argument("-Q", "--quality", type=int, default=0, help="Quality level to calculate for (default: 0)")
    parser.add_argument("-S", "--search", type=str, nargs="+", help="Search for specific resources by name (case-insensitive)")
    parser.add_argument("-B", "--building", type=str, nargs="+", help="Filter resources by building name")
    parser.add_argument("-A", "--abundance", type=float, default=90, help="Abundance percentage for mine/well resources (default: 90)")
    parser.add_argument("-O", "--admin-overhead", type=float, default=0, help="Administration overhead percentage to add to wages (default: 0)")
    parser.add_argument("-C", "--contract", action="store_true", help="Calculate values for direct contracts (0% market fee, 50% transportation cost)")
    parser.add_argument("-R", "--roi", action="store_true", help="Calculate and display ROI for buildings based on best performing resource")
    parser.add_argument("-D", "--debug-unassigned", action="store_true", help="List all resources that are not assigned to any building")
    parser.add_argument("-E", "--exclude-seasonal", action="store_true", help="Exclude seasonal resources from calculations")
    parser.add_argument("-P", "--prospect", action="store_true", help="Simulate prospecting to find target abundance")
    parser.add_argument("-T", "--time", type=float, default=12, help="Time in hours for one build attempt (default: 12)")
    parser.add_argument("-L", "--slots", type=int, default=1, help="Number of simultaneous building slots (default: 1)")
    args = parser.parse_args()

    if args.prospect:
        simulate_prospecting(args.abundance / 100, args.time, args.slots)
        return

    # Load abundance resources
    abundance_resources = []
    if os.path.exists("abundance_resources.json"):
        with open("abundance_resources.json", "r") as f:
            abundance_resources = [res.lower() for res in json.load(f)]

    # Load seasonal resources
    seasonal_resources = []
    if os.path.exists("seasonal_resources.json"):
        with open("seasonal_resources.json", "r") as f:
            seasonal_resources = [res.lower() for res in json.load(f)]

    # Load buildings
    buildings_data = []
    resource_to_building = {}
    if os.path.exists("buildings.json"):
        with open("buildings.json", "r") as f:
            buildings_data = json.load(f)
            for b in buildings_data:
                for res_name in b.get("produces", []):
                    resource_to_building[res_name.lower()] = b["name"]

    api = SimcoAPI(realm=0)
    
    try:
        # Fetch resources and VWAPs
        resources_data = api.get_resources()
        resources = resources_data.get("resources", [])

        if args.debug_unassigned:
            unassigned = [res.get("name") for res in resources if res.get("name", "").lower() not in resource_to_building]
            unassigned.sort()
            console.print("\n[bold red]Resources not assigned to any building:[/bold red]")
            for name in unassigned:
                console.print(f" - {name}")
            return

        vwaps_data = api.get_market_vwaps()

        # Save the API output
        save_json(resources_data, "resources.json")
        save_json(vwaps_data, "vwaps.json")
        
        # Build a price map: {resource_id: vwap} (Only for selected Quality)
        price_map = {}
        q0_price_map = {} # For building costs
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
        
        # Map resource names to IDs for building cost lookup
        name_to_id = {r.get("name", "").lower(): r.get("id") for r in resources_data.get("resources", [])}
        
        # Get Transport price for shipping costs
        # Find the resource ID for "Transport" first
        transport_id = None
        for res in resources_data.get("resources", []):
            res_name_lower = res.get("name", "").lower()
            if res_name_lower == "transport":
                transport_id = res.get("id")
                break
        
        # If not found by exact name, try "Transport units" or similar
        if transport_id is None:
            for res in resources_data.get("resources", []):
                if "transport" in res.get("name", "").lower():
                    transport_id = res.get("id")
                    break
        
        transport_price = 0
        if transport_id is not None:
            if isinstance(vwaps_data, list):
                for entry in vwaps_data:
                    if isinstance(entry, dict) and entry.get("resourceId") == transport_id and entry.get("quality") == 0:
                        transport_price = entry.get("vwap", 0)
                        break
        else:
            console.print("[yellow]Warning: Could not find 'Transport' resource by name.[/yellow]")
        
        profits = []
        resources = resources_data.get("resources", [])
        
        for res in resources:
            name = res.get("name", "")
            name_lower = name.lower()
            
            # Filter seasonal resources if requested
            if args.exclude_seasonal and name_lower in seasonal_resources:
                continue

            # Filter by building if provided
            if args.building:
                building_name = resource_to_building.get(name_lower)
                if not building_name or not any(term.lower() in building_name.lower() for term in args.building):
                    continue

            # Filter by search string if provided
            if args.search:
                if not any(term.lower() in name_lower for term in args.search):
                    continue

            res_id = res.get("id")
            produced_per_hour = res.get("producedAnHour", 0)
            
            # Apply abundance if applicable
            is_abundance_res = name_lower in abundance_resources
            if is_abundance_res:
                produced_per_hour *= (args.abundance / 100)

            wages = res.get("wages", 0)
            
            # Apply administration overhead to wages
            admin_overhead = wages * (args.admin_overhead / 100)
            total_wages = wages + admin_overhead

            inputs = res.get("inputs", {})
            transport_units_needed = res.get("transportation", 0)
            
            # 1. Revenue per hour (Selected Quality)
            selling_price = price_map.get(res_id, 0)
            if selling_price == 0:
                continue
                
            revenue_per_hour = selling_price * produced_per_hour
            
            # 2. Input costs per hour
            input_costs_per_hour = 0
            missing_input_price = False
            for input_id_str, input_info in inputs.items():
                input_id = int(input_id_str)
                qty_per_unit = input_info.get("quantity", 0)
                input_price = price_map.get(input_id, 0)
                
                if input_price == 0:
                    missing_input_price = True
                
                input_costs_per_hour += (input_price * qty_per_unit) * produced_per_hour

            # 3. Transportation costs (per unit and per hour)
            # Market fee is usually 4%, but transport is a direct cost to sell
            transport_cost_per_unit = transport_units_needed * transport_price
            if args.contract:
                transport_cost_per_unit *= 0.5
            transport_costs_per_hour = transport_cost_per_unit * produced_per_hour
            
            # 4. Market Fee (4% of revenue, 0% for contracts)
            market_fee_percentage = 0.04 if not args.contract else 0.0
            market_fee_per_hour = revenue_per_hour * market_fee_percentage

            # 5. Total profit per hour
            # Profit = Revenue - Fee - Wages - Admin Overhead - Input Costs - Transport Costs
            profit_per_hour = revenue_per_hour - market_fee_per_hour - total_wages - input_costs_per_hour - transport_costs_per_hour
            
            profits.append({
                "name": name,
                "profit_per_hour": profit_per_hour,
                "revenue_per_hour": revenue_per_hour,
                "market_fee_per_hour": market_fee_per_hour,
                "costs_per_hour": total_wages + input_costs_per_hour,
                "transport_costs_per_hour": transport_costs_per_hour,
                "missing_input_price": missing_input_price,
                "is_abundance_res": is_abundance_res
            })

        # Sort by profit per hour descending
        profits.sort(key=lambda x: x["profit_per_hour"], reverse=True)

        header_title = "Top 30 Most Profitable Resources"
        if args.search or args.building:
            parts = []
            if args.search:
                parts.append(f"search: '{', '.join(args.search)}'")
            if args.building:
                parts.append(f"building: '{', '.join(args.building)}'")
            header_title = f"Results for {' & '.join(parts)}"
        
        if args.contract:
            header_title += " (Direct Contract Mode)"
        
        console.print(f"\n[bold blue]{header_title}[/bold blue]")
        market_fee_display = "0%" if args.contract else "4%"
        console.print(f"Quality: [bold cyan]{args.quality}[/bold cyan] | Transport: [bold cyan]${transport_price:.3f}[/bold cyan] | Market Fee: [bold cyan]{market_fee_display}[/bold cyan] | Admin Overhead: [bold cyan]{args.admin_overhead}%[/bold cyan]")

        table = Table(show_header=True, header_style="bold white on blue", box=box.ROUNDED, border_style="bright_black")
        table.add_column("Resource", style="bold white", width=25)
        table.add_column("Profit/hr", justify="right")
        table.add_column("Revenue/hr", justify="right", style="white")
        table.add_column("Fee/hr", justify="right", style="red")
        table.add_column("Costs/hr", justify="right", style="yellow")
        table.add_column("Transp/hr", justify="right", style="magenta")
        
        display_count = 30 if not args.search else len(profits)
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
                f"${p['transport_costs_per_hour']:,.2f}{warn}"
            )
        
        console.print(table)
        
        if any(p["is_abundance_res"] for p in profits[:display_count]):
            console.print(f"\n[bold yellow](*)[/bold yellow] indicates abundance-based resource (applied {args.abundance}% abundance)")
        if any(p["missing_input_price"] for p in profits[:display_count]):
            console.print(f"[bold red](!)[/bold red] indicates one or more source materials had no Quality {args.quality} market price")

        # ROI Calculation
        if args.roi and buildings_data:
            roi_table = Table(title="Building ROI Analysis", show_header=True, header_style="bold green", box=box.ROUNDED)
            roi_table.add_column("Building", style="bold white")
            roi_table.add_column("Best Resource", style="cyan")
            roi_table.add_column("Building Cost", justify="right", style="magenta")
            roi_table.add_column("Daily Profit", justify="right", style="green")
            roi_table.add_column("ROI (Daily)", justify="right", style="bold yellow")
            roi_table.add_column("Break Even", justify="right", style="white")

            # Create a map of resource name -> profit info for quick lookup (case-insensitive keys)
            res_profit_map = {p["name"].lower(): p for p in profits}
            
            roi_data = []

            for b in buildings_data:
                b_name = b["name"]
                
                # Check if this building is relevant (has any resources in the current profits list)
                produces = b.get("produces", [])
                
                best_res_profit = -float('inf')
                best_res_name = None
                
                has_relevant_resource = False
                for res_name in produces:
                    res_name_lower = res_name.lower()
                    if res_name_lower in res_profit_map:
                        has_relevant_resource = True
                        p_data = res_profit_map[res_name_lower]
                        if p_data["profit_per_hour"] > best_res_profit:
                            best_res_profit = p_data["profit_per_hour"]
                            best_res_name = p_data["name"]
                
                if not has_relevant_resource:
                    continue

                # Calculate Building Cost (using Q0 prices)
                total_cost = 0
                cost_dict = b.get("cost", {})
                missing_cost_price = False
                for mat_name, amount in cost_dict.items():
                    mat_id = name_to_id.get(mat_name.lower())
                    if mat_id:
                        price = q0_price_map.get(mat_id, 0)
                        if price == 0:
                            missing_cost_price = True
                        total_cost += price * amount
                    else:
                        missing_cost_price = True
                
                daily_profit = best_res_profit * 24
                
                roi_daily = 0
                days_break_even = float('inf')
                
                if total_cost > 0:
                    roi_daily = (daily_profit / total_cost) * 100
                    if daily_profit > 0:
                        days_break_even = total_cost / daily_profit
                
                roi_data.append({
                    "building": b_name,
                    "resource": best_res_name,
                    "cost": total_cost,
                    "daily_profit": daily_profit,
                    "roi": roi_daily,
                    "break_even": days_break_even,
                    "missing_cost": missing_cost_price
                })

            # Sort by ROI Descending
            roi_data.sort(key=lambda x: x["roi"], reverse=True)
            
            for d in roi_data:
                break_even_str = "∞" if d["break_even"] == float('inf') else f"{d['break_even']:.1f} days"
                if d["daily_profit"] < 0:
                     break_even_str = "Never"

                warn = " (!)" if d["missing_cost"] else ""
                
                roi_table.add_row(
                    d["building"],
                    d["resource"],
                    f"${d['cost']:,.0f}{warn}",
                    f"${d['daily_profit']:,.0f}",
                    f"{d['roi']:.2f}%",
                    break_even_str
                )
            
            console.print("\n")
            console.print(roi_table)
            if any(d["missing_cost"] for d in roi_data):
                console.print("[yellow](!) Warning: Some building costs calculated with missing material prices (assumed $0).[/yellow]")

    except httpx.HTTPError as exc:
        console.print(f"[bold red]Error fetching data: {exc}[/bold red]")

if __name__ == "__main__":
    main()
