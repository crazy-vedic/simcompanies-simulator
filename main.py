import httpx
import json
import argparse
import os
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

def main():
    parser = argparse.ArgumentParser(description="Simcotools calculation script")
    parser.add_argument("-Q", "--quality", type=int, default=0, help="Quality level to calculate for (default: 0)")
    parser.add_argument("-S", "--search", type=str, nargs="+", help="Search for specific resources by name (case-insensitive)")
    parser.add_argument("-B", "--building", type=str, nargs="+", help="Filter resources by building name")
    parser.add_argument("-A", "--abundance", type=float, default=90, help="Abundance percentage for mine/well resources (default: 90)")
    parser.add_argument("-O", "--admin-overhead", type=float, default=0, help="Administration overhead percentage to add to wages (default: 0)")
    parser.add_argument("-C", "--contract", action="store_true", help="Calculate values for direct contracts (0% market fee, 50% transportation cost)")
    parser.add_argument("-D", "--debug-unassigned", action="store_true", help="List all resources that are not assigned to any building")
    args = parser.parse_args()

    # Load abundance resources
    abundance_resources = []
    if os.path.exists("abundance_resources.json"):
        with open("abundance_resources.json", "r") as f:
            abundance_resources = json.load(f)

    # Load buildings
    buildings_data = []
    resource_to_building = {}
    if os.path.exists("buildings.json"):
        with open("buildings.json", "r") as f:
            buildings_data = json.load(f)
            for b in buildings_data:
                for res_name in b.get("produces", []):
                    resource_to_building[res_name] = b["name"]

    api = SimcoAPI(realm=0)
    
    try:
        # Fetch resources and VWAPs
        resources_data = api.get_resources()
        resources = resources_data.get("resources", [])

        if args.debug_unassigned:
            unassigned = [res.get("name") for res in resources if res.get("name") not in resource_to_building]
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
        if isinstance(vwaps_data, list):
            for entry in vwaps_data:
                if isinstance(entry, dict):
                    r_id = entry.get("resourceId")
                    quality = entry.get("quality")
                    vwap = entry.get("vwap")
                    if r_id is not None and vwap is not None and quality == args.quality:
                        price_map[int(r_id)] = vwap
        
        # Get Transport price for shipping costs
        # Find the resource ID for "Transport" first
        transport_id = None
        for res in resources_data.get("resources", []):
            if res.get("name") == "Transport":
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
            name = res.get("name")
            
            # Filter by building if provided
            if args.building:
                building_name = resource_to_building.get(name)
                if not building_name or not any(term.lower() in building_name.lower() for term in args.building):
                    continue

            # Filter by search string if provided
            if args.search:
                if not any(term.lower() in name.lower() for term in args.search):
                    continue

            res_id = res.get("id")
            produced_per_hour = res.get("producedAnHour", 0)
            
            # Apply abundance if applicable
            is_abundance_res = name in abundance_resources
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

        # Show building cost if filtering by building
        if args.building and buildings_data:
            for b_term in args.building:
                for b in buildings_data:
                    if b_term.lower() in b["name"].lower():
                        cost_table = Table(title=f"Construction Cost: [bold cyan]{b['name']}[/bold cyan]", show_header=True, header_style="bold magenta", box=box.SIMPLE)
                        cost_table.add_column("Resource")
                        cost_table.add_column("Amount", justify="right")
                        for cost_res, cost_amt in b.get("cost", {}).items():
                            cost_table.add_row(cost_res, str(cost_amt))
                        console.print(cost_table)

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

    except httpx.HTTPError as exc:
        console.print(f"[bold red]Error fetching data: {exc}[/bold red]")

if __name__ == "__main__":
    main()
