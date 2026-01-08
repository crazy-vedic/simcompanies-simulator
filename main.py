import httpx
import json
import argparse
import os

class SimcoAPI:
    def __init__(self, realm: int = 0):
        self.base_url = f"https://api.simcotools.com/v1/realms/{realm}"
        self.headers = {"accept": "application/json"}

    def get_resources(self):
        url = f"{self.base_url}/resources"
        all_resources = []
        
        # Try disable_pagination=True first
        print(f"Fetching resources from {url} (attempting disable_pagination=True)...")
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
                    print(f"Successfully fetched all {len(data['resources'])} resources with disable_pagination.")
                    return data
        except Exception as e:
            print(f"disable_pagination failed or returned unexpected format: {e}")

        # Fallback to manual pagination
        print("Falling back to manual pagination...")
        current_page = 1
        last_page = 1
        
        while current_page <= last_page:
            print(f"  Fetching page {current_page}...")
            response = httpx.get(url, headers=self.headers, params={"page": current_page})
            response.raise_for_status()
            data = response.json()
            
            resources = data.get("resources", [])
            all_resources.extend(resources)
            
            metadata = data.get("metadata", {})
            current_page = metadata.get("currentPage", 1) + 1
            last_page = metadata.get("lastPage", 1)
            
        print(f"Successfully fetched {len(all_resources)} resources via manual pagination.")
        return {"resources": all_resources}

    def get_market_vwaps(self):
        url = f"{self.base_url}/market/vwaps"
        print(f"Fetching market VWAPs from {url}...")
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
    print(f"Data saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Simcotools calculation script")
    parser.add_argument("-Q", "--quality", type=int, default=0, help="Quality level to calculate for (default: 0)")
    parser.add_argument("-S", "--search", type=str, help="Search for a specific resource by name (case-insensitive)")
    parser.add_argument("-A", "--abundance", type=float, default=90, help="Abundance percentage for mine/well resources (default: 90)")
    parser.add_argument("-O", "--admin-overhead", type=float, default=0, help="Administration overhead percentage to add to wages (default: 0)")
    args = parser.parse_args()

    # Load abundance resources
    abundance_resources = []
    if os.path.exists("abundance_resources.json"):
        with open("abundance_resources.json", "r") as f:
            abundance_resources = json.load(f)

    api = SimcoAPI(realm=0)
    
    try:
        # Fetch resources and VWAPs
        resources_data = api.get_resources()
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
            print("Warning: Could not find 'Transport' resource by name.")
        
        profits = []
        resources = resources_data.get("resources", [])
        
        for res in resources:
            name = res.get("name")
            
            # Filter by search string if provided
            if args.search and args.search.lower() not in name.lower():
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
            # Market fee is usually 3%, but transport is a direct cost to sell
            transport_cost_per_unit = transport_units_needed * transport_price
            transport_costs_per_hour = transport_cost_per_unit * produced_per_hour
            
            # 4. Market Fee (4% of revenue)
            market_fee_per_hour = revenue_per_hour * 0.04

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

        header_title = f"Top 30 Most Profitable Resources" if not args.search else f"Search results for '{args.search}'"
        print(f"\n{header_title} per Hour (Quality {args.quality} Only):")
        print(f"Using Transport price: ${transport_price:.3f} | Market Fee: 4% | Admin Overhead: {args.admin_overhead}%")
        print("-" * 105)
        print(f"{'Resource':<25} | {'Profit/hr':>12} | {'Revenue/hr':>12} | {'Fee/hr':>10} | {'Costs/hr':>12} | {'Transp/hr':>10}")
        print("-" * 105)
        
        display_count = 30 if not args.search else len(profits)
        for p in profits[:display_count]:
            warn = " (!)" if p["missing_input_price"] else ""
            abundance_mark = " (*)" if p["is_abundance_res"] else ""
            print(f"{p['name'] + abundance_mark:<25} | ${p['profit_per_hour']:>11.2f} | ${p['revenue_per_hour']:>11.2f} | ${p['market_fee_per_hour']:>9.2f} | ${p['costs_per_hour']:>11.2f} | ${p['transport_costs_per_hour']:>9.2f}{warn}")
        
        if any(p["is_abundance_res"] for p in profits[:display_count]):
            print(f"\n(*) indicates abundance-based resource (applied {args.abundance}% abundance)")
        print(f"(!) indicates one or more source materials had no Quality {args.quality} market price")

    except httpx.HTTPError as exc:
        print(f"Error fetching data: {exc}")

if __name__ == "__main__":
    main()
