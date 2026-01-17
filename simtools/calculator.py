"""Profit and ROI calculation logic for simtools."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simtools.models.building import Building
    from simtools.models.resource import Resource
    from simtools.models.market import MarketData


@dataclass
class ProfitConfig:
    """Configuration for profit calculations."""

    quality: int = 0
    abundance: float = 90.0
    admin_overhead: float = 0.0
    is_contract: bool = False
    has_robots: bool = False
    sales_speed_bonus: float = 0.0


def find_best_resource_profit(
    resources: list[Resource],
    market: MarketData,
    quality: int,
    config: ProfitConfig,
) -> tuple[Resource | None, dict | None]:
    """Find the most profitable resource from a list and return its profit data.

    This is a centralized helper that avoids duplicating the logic for finding
    the best profitable resource for a building across multiple functions.

    Args:
        resources: List of Resource instances to evaluate.
        market: MarketData instance containing prices and transport info.
        quality: Quality level for prices.
        config: Profit calculation configuration.

    Returns:
        Tuple of (best_resource, profit_data) or (None, None) if no valid resources.
    """
    best_resource = None
    best_profit = -float("inf")
    best_profit_data = None

    for res in resources:
        selling_price = market.get_price(res.id, quality)
        if selling_price <= 0:
            continue

        profit_data = res.calculate_profit(
            selling_price=selling_price,
            market=market,
            quality=quality,
            abundance=config.abundance,
            admin_overhead=config.admin_overhead,
            is_contract=config.is_contract,
            has_robots=config.has_robots,
        )

        if profit_data["profit_per_hour"] > best_profit:
            best_profit = profit_data["profit_per_hour"]
            best_resource = res
            best_profit_data = profit_data

    return best_resource, best_profit_data


def calculate_total_investment(
    building: Building,
    level: int,
    market: MarketData,
) -> tuple[float, bool]:
    """Calculate the total investment cost to build and upgrade to a given level.

    This is a centralized helper that computes the total cost including:
    - Base construction cost (level 1)
    - Upgrade costs from level 1 to target level

    The upgrade cost formula: step from level k to k+1 costs k * base_cost.
    Total upgrade cost = base_cost * level * (level - 1) / 2

    Args:
        building: Building instance.
        level: Target level (1 or higher).
        market: MarketData instance containing Q0 prices and name_to_id mapping.

    Returns:
        Tuple of (total_investment, missing_cost_flag).
    """
    base_cost, missing_cost = building.calculate_construction_cost(market)

    if level <= 1:
        return base_cost, missing_cost

    # Total upgrade cost = base_cost * level * (level - 1) / 2
    upgrade_cost = base_cost * level * (level - 1) / 2
    return base_cost + upgrade_cost, missing_cost


def calculate_all_profits(
    resources: list[Resource],
    market: MarketData,
    quality: int,
    config: ProfitConfig,
    retail_data: dict
) -> list[dict]:
    """Calculate profits for all resources.

    Args:
        resources: List of Resource instances to calculate profits for.
        market: MarketData instance containing prices and transport info.
        quality: Quality level for prices.
        config: Profit calculation configuration.
        retail_data: Raw retail info data dict keyed by dbLetter (resource ID).

    Returns:
        List of profit dictionaries sorted by profit_per_hour descending.
        Includes both market sales and retail sales where applicable.
    """
    profits = []

    for res in resources:
        selling_price = market.get_price(res.id, quality)
        if selling_price > 0:
            # Calculate market profit (existing behavior)
            # Input costs use market exchange prices (for resources that have inputs)
            profit_data = res.calculate_profit(
                selling_price=selling_price,
                market=market,
                quality=quality,
                abundance=config.abundance,
                admin_overhead=config.admin_overhead,
                is_contract=config.is_contract,
                has_robots=config.has_robots,
            )
            profits.append(profit_data)

        # Add retail profit if resource has retail info
        if res.retail_info:
            # Process retail data: merge resource retailInfo with API retail info
            # temp is keyed by dbLetter (resource ID), so use res.id to lookup
            retail_info_from_api = retail_data.get(res.id, {})
            
            # Find retail info entry for the requested quality
            retail_info_entry = None
            for ri in res.retail_info:
                if ri.get("quality") == quality:
                    retail_info_entry = ri.copy()  # Start with base retail info
                    # Merge/override with API data if available
                    if retail_info_from_api:
                        # Update with API data (saturation, averagePrice, etc.)
                        retail_info_entry.update({
                            "saturation": retail_info_from_api.get("saturation", retail_info_entry.get("saturation", 1.0)),
                            "averagePrice": retail_info_from_api.get("averagePrice", retail_info_entry.get("averagePrice", 0)),
                        })
                    break
            
            if retail_info_from_api:
                # For retail, input cost is the market exchange price (cost to buy the product)
                # This is the price you pay to acquire the product to sell it
                input_cost_per_unit = selling_price if selling_price > 0 else 0.0
                if retail_info_from_api.get('retailData')[0].get('amountSoldRestaurant',0)>0: continue
                retail_profit = res.calculate_retail_profit(
                    market=market,
                    retail_data=retail_info_from_api | retail_info_entry,
                    quality=quality,
                    building_level=1,
                    sales_speed_bonus=config.sales_speed_bonus,
                    admin_overhead=config.admin_overhead,
                    input_cost_per_unit=input_cost_per_unit,
                )
                
                # Add retail entry (consistent with market entries - add regardless of profitability)
                retail_profit["is_retail"] = True
                profits.append(retail_profit)

    # Sort by profit descending
    profits.sort(key=lambda x: x["profit_per_hour"], reverse=True)
    return profits


def calculate_building_roi(
    buildings: list[Building],
    profits: list[dict],
    market: MarketData,
) -> list[dict]:
    """Calculate ROI for buildings based on their best performing resource.

    For retail buildings, looks for retail profit entries (with "(Retail)" suffix).
    For production buildings, uses standard market profit entries.

    Args:
        buildings: List of Building instances.
        profits: List of profit dictionaries from calculate_all_profits.
        market: MarketData instance containing Q0 prices and name_to_id mapping.

    Returns:
        List of ROI dictionaries sorted by ROI descending.
    """
    # Create a lookup map for profits by resource name
    res_profit_map = {p["name"].lower(): p for p in profits}

    roi_data = []

    for building in buildings:
        # Find the best resource for this building
        best_profit = -float("inf")
        best_name = None
        has_relevant_resource = False

        for res_name in building.produces:
            # For retail buildings, look for the retail version
            if building.retail:
                lookup_name = f"{res_name} (Retail)".lower()
            else:
                lookup_name = res_name.lower()
            
            if lookup_name in res_profit_map:
                has_relevant_resource = True
                p_data = res_profit_map[lookup_name]
                if p_data["profit_per_hour"] > best_profit:
                    best_profit = p_data["profit_per_hour"]
                    best_name = p_data["name"]

        if not has_relevant_resource:
            continue

        # Calculate building cost
        total_cost, missing_cost = building.calculate_construction_cost(market)

        daily_profit = best_profit * 24

        roi_daily = 0.0
        days_break_even = float("inf")

        if total_cost > 0:
            roi_daily = (daily_profit / total_cost) * 100
            if daily_profit > 0:
                days_break_even = total_cost / daily_profit

        roi_data.append(
            {
                "building": building.name,
                "resource": best_name,
                "cost": total_cost,
                "daily_profit": daily_profit,
                "roi": roi_daily,
                "break_even": days_break_even,
                "missing_cost": missing_cost,
            }
        )

    # Sort by ROI descending
    roi_data.sort(key=lambda x: x["roi"], reverse=True)
    return roi_data


def calculate_level_roi(
    building: Building,
    best_profit_data: dict,
    market: MarketData,
    max_level: int = 20,
    step_mode: bool = False,
) -> list[dict]:
    """Calculate ROI for a single building across multiple levels.

    Args:
        building: The Building instance.
        best_profit_data: Profit data for the best resource at level 1.
        market: MarketData instance containing Q0 prices and name_to_id mapping.
        max_level: Maximum level to calculate up to.
        step_mode: If True, calculate ROI for each step L -> L+1 based on 
                   that step's cost and the additional profit gained.

    Returns:
        List of ROI dictionaries for each level/step.
    """
    base_cost, missing_cost = building.calculate_construction_cost(market)
    base_profit = best_profit_data["profit_per_hour"]

    # Ensure building is treated as level 1 for starting point
    original_level = building.level
    building.level = 1

    level_roi_data = []

    for level in range(1, max_level + 1):
        if step_mode:
            # ROI for the step L -> L+1
            # Cost = L * base_cost
            # Gained Profit = (L+1 - L) * base_profit = base_profit
            if level == 1:
                # Level 1 is the initial construction
                cost = base_cost
                gained_daily_profit = base_profit * 24
                display_label = "Initial"
            else:
                # Upgrade step from level-1 to level
                # Step cost from k to k+1 is k * base_cost. 
                # So to reach 'level' from 'level-1', cost is (level-1) * base_cost
                cost = (level - 1) * base_cost
                gained_daily_profit = base_profit * 24  # Every level adds exactly 1x base profit
                display_label = f"Lv{level-1}→{level}"
        else:
            # Total investment to reach this level
            upgrade_cost, _ = building.calculate_upgrade_cost(market, level)
            cost = base_cost + upgrade_cost
            gained_daily_profit = base_profit * 24 * level
            display_label = str(level)

        roi_daily = 0.0
        days_break_even = float("inf")

        if cost > 0:
            roi_daily = (gained_daily_profit / cost) * 100
            if gained_daily_profit > 0:
                days_break_even = cost / gained_daily_profit

        level_roi_data.append(
            {
                "building": building.name,
                "level": display_label,
                "resource": best_profit_data["name"],
                "cost": cost,
                "daily_profit": gained_daily_profit,
                "roi": roi_daily,
                "break_even": days_break_even,
                "missing_cost": missing_cost,
            }
        )

    building.level = original_level
    return level_roi_data


def calculate_lifecycle_roi(
    building: Building,
    resource: Resource,
    profit_config: ProfitConfig,
    market: MarketData,
    quality: int,
    start_abundance: float,
    end_abundance: float = 0.85,
    decay_rate: float = 0.00032,
    max_level: int = 20,
    base_build_time: float = 0.0,
) -> list[dict]:
    """Calculate lifecycle ROI simulating decay from start_abundance to end_abundance.

    Accounts for:
      - Build time delay (abundance decays during construction/upgrades).
      - Daily profit decay as abundance drops.
      - Scrapping returns: 100% of cost for L1-L2, 50% for L3+.
      - Net Profit = Sum(Daily Profits) - Unrecoverable Cost.

    Args:
        building: Building instance.
        resource: Resource instance.
        profit_config: ProfitConfig for parameters like overhead, robots.
        market: MarketData instance containing prices and transport info.
        quality: Quality level for prices.
        start_abundance: Starting abundance (0.0 to 1.0).
        end_abundance: Ending abundance target (default 0.85).
        decay_rate: Daily abundance decay rate (default 0.00032).
        max_level: Max level to simulate.
        base_build_time: Base construction time in hours (Lv 1).

    Returns:
        List of result dictionaries sorted by net profit.
    """
    if start_abundance <= end_abundance:
        return []

    # Calculate base profit metrics at 100% abundance and 0%
    selling_price = market.get_price(resource.id, quality)
    p_100 = resource.calculate_profit(
        selling_price=selling_price,
        market=market,
        quality=quality,
        abundance=100.0,
        admin_overhead=profit_config.admin_overhead,
        is_contract=profit_config.is_contract,
        has_robots=profit_config.has_robots,
    )
    p_0 = resource.calculate_profit(
        selling_price=selling_price,
        market=market,
        quality=quality,
        abundance=0.0,
        admin_overhead=profit_config.admin_overhead,
        is_contract=profit_config.is_contract,
        has_robots=profit_config.has_robots,
    )
    
    hourly_fixed_cost = -p_0["profit_per_hour"]  # Wages
    hourly_variable_profit_at_100 = p_100["profit_per_hour"] + hourly_fixed_cost
    
    # Base construction cost (Level 1)
    base_cost, missing_cost = building.calculate_construction_cost(market)
    
    results = []
    
    for level in range(1, max_level + 1):
        # Calculate total investment
        if level == 1:
            total_investment = base_cost
        else:
            upgrade_steps_cost = (level * (level - 1) // 2) * base_cost
            total_investment = base_cost + upgrade_steps_cost
            
        # Calculate recoverable cost
        cost_l2 = 2 * base_cost 
        if level <= 2:
            recoverable = total_investment
        else:
            recoverable = cost_l2 + (total_investment - cost_l2) * 0.5
        unrecoverable_cost = total_investment - recoverable
        
        # Calculate Build Time & Abundance Decay during build
        # Time(L) = base_time (L1) + sum_{k=1}^{L-1} k*base_time
        if level == 1:
            total_build_time = base_build_time
        else:
            upgrade_steps_time = (level * (level - 1) // 2) * base_build_time
            total_build_time = base_build_time + upgrade_steps_time
            
        build_days = total_build_time / 24.0
        
        # Effective starting abundance after build time
        effective_start_abundance = start_abundance * ((1.0 - decay_rate) ** build_days)
        
        # If abundance dropped below target during build, skipped
        if effective_start_abundance <= end_abundance:
            continue
            
        # Calculate days to reach end_abundance from effective start
        if decay_rate > 0:
            total_days = math.log(end_abundance / effective_start_abundance) / math.log(1.0 - decay_rate)
        else:
            total_days = float("inf")
            
        num_days = int(total_days)
        
        # Sum of abundance fractions over the lifecycle
        abundance_sum_fraction = effective_start_abundance * (1.0 - (1.0 - decay_rate) ** num_days) / decay_rate
        
        # Operational Profit
        total_variable = hourly_variable_profit_at_100 * abundance_sum_fraction
        total_fixed = hourly_fixed_cost * num_days
        total_operational_profit = level * 24 * (total_variable - total_fixed)
        
        net_profit = total_operational_profit - unrecoverable_cost
        
        results.append({
            "level": level,
            "days": num_days,
            "build_time_hours": total_build_time,
            "investment": total_investment,
            "unrecoverable": unrecoverable_cost,
            "operational_profit": total_operational_profit,
            "net_profit": net_profit,
            "resource": resource.name,
            "missing_cost": missing_cost or p_100["missing_input_price"]
        })
        
    results.sort(key=lambda x: x["net_profit"], reverse=True)
    return results


def compare_market_vs_contract(
    resource: Resource,
    market_price: float,
    contract_price: float,
    market: MarketData,
    quality: int,
    config: ProfitConfig,
) -> dict:
    """Compare selling on market vs selling via contract with custom price.

    Args:
        resource: Resource instance to compare.
        market_price: Market price from VWAP.
        contract_price: User-defined contract price per unit.
        market: MarketData instance containing prices and transport info.
        quality: Quality level for input prices.
        config: Profit calculation configuration.

    Returns:
        Dictionary with comparison data including:
            - market: Market mode profit breakdown
            - contract: Contract mode profit breakdown
            - diff_per_unit: Net difference per unit (contract - market)
            - diff_per_hour: Net difference per hour
            - missing_input_price: True if any input price was missing
            - is_abundance_res: True if resource is abundance-based
    """
    # Calculate market mode (4% fee, 100% transport)
    market_data = resource.calculate_profit(
        selling_price=market_price,
        market=market,
        quality=quality,
        abundance=config.abundance,
        admin_overhead=config.admin_overhead,
        is_contract=False,
        has_robots=config.has_robots,
    )

    # Calculate contract mode (0% fee, 50% transport) with custom price
    contract_data = resource.calculate_profit(
        selling_price=contract_price,
        market=market,
        quality=quality,
        abundance=config.abundance,
        admin_overhead=config.admin_overhead,
        is_contract=True,
        has_robots=config.has_robots,
    )

    # Calculate per-unit values
    produced_per_hour = resource.get_effective_production(config.abundance)
    
    market_net_per_unit = market_data["profit_per_hour"] / produced_per_hour if produced_per_hour > 0 else 0
    contract_net_per_unit = contract_data["profit_per_hour"] / produced_per_hour if produced_per_hour > 0 else 0
    
    diff_per_unit = contract_net_per_unit - market_net_per_unit
    diff_per_hour = contract_data["profit_per_hour"] - market_data["profit_per_hour"]

    return {
        "name": resource.name,
        "market": {
            "price": market_price,
            "fee_per_unit": market_data["market_fee_per_hour"] / produced_per_hour if produced_per_hour > 0 else 0,
            "transport_per_unit": market_data["transport_costs_per_hour"] / produced_per_hour if produced_per_hour > 0 else 0,
            "net_per_unit": market_net_per_unit,
            "profit_per_hour": market_data["profit_per_hour"],
        },
        "contract": {
            "price": contract_price,
            "fee_per_unit": 0.0,  # Always 0 for contracts
            "transport_per_unit": contract_data["transport_costs_per_hour"] / produced_per_hour if produced_per_hour > 0 else 0,
            "net_per_unit": contract_net_per_unit,
            "profit_per_hour": contract_data["profit_per_hour"],
        },
        "diff_per_unit": diff_per_unit,
        "diff_per_hour": diff_per_hour,
        "missing_input_price": market_data["missing_input_price"],
        "is_abundance_res": market_data["is_abundance_res"],
    }


def simulate_prospecting(
    target_abundance: float,
    attempt_time: float,
    slots: int = 1,
) -> dict:
    """Simulate prospecting to find target abundance.

    Uses the Gaussian roll formula:
        min(1.0, max(0.1, random.gauss(mu=0.6, sigma=0.15)))

    Args:
        target_abundance: Target abundance as a decimal (0.0 to 1.0).
        attempt_time: Time in hours for one build attempt.
        slots: Number of simultaneous building slots.

    Returns:
        Dictionary with simulation results.
    """
    mu = 0.6
    sigma = 0.15

    # Calculate probability of success
    if target_abundance <= 0.1:
        p_success_single = 1.0
    elif target_abundance > 1.0:
        p_success_single = 0.0
    else:
        # P(X >= target) = 1 - Phi((target - mu) / sigma)
        z = (target_abundance - mu) / sigma
        p_success_single = 1.0 - (0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

    if p_success_single <= 0:
        return {
            "target_abundance": target_abundance,
            "impossible": True,
            "p_success_single": 0.0,
        }

    # Probability of at least one success in a block of 'slots' attempts
    p_success_block = 1.0 - (1.0 - p_success_single) ** slots

    expected_blocks = 1.0 / p_success_block
    expected_time = expected_blocks * attempt_time

    # Calculate decay time to 85% abundance
    decay_target = 0.85
    decay_rate = 0.00032
    days_to_85 = None
    if target_abundance > decay_target:
        days_to_85 = math.log(decay_target / target_abundance) / math.log(1.0 - decay_rate)

    # Calculate confidence intervals
    confidence_intervals = []
    if p_success_block < 1.0:
        for conf in [0.50, 0.80, 0.90, 0.95, 0.99]:
            n_blocks = math.ceil(math.log(1.0 - conf) / math.log(1.0 - p_success_block))
            n_time = n_blocks * attempt_time
            confidence_intervals.append(
                {
                    "confidence": conf,
                    "blocks": n_blocks,
                    "time_hours": n_time,
                    "time_days": n_time / 24,
                }
            )
    else:
        confidence_intervals.append(
            {
                "confidence": 1.0,
                "blocks": 1,
                "time_hours": attempt_time,
                "time_days": attempt_time / 24,
            }
        )

    return {
        "target_abundance": target_abundance,
        "attempt_time": attempt_time,
        "slots": slots,
        "impossible": False,
        "p_success_single": p_success_single,
        "p_success_block": p_success_block,
        "expected_blocks": expected_blocks,
        "expected_time": expected_time,
        "days_to_85": days_to_85,
        "confidence_intervals": confidence_intervals,
    }


def calculate_company_building_stats(
    building: Building,
    level: int,
    resources: list[Resource],
    market: MarketData,
    quality: int,
    config: ProfitConfig,
) -> dict:
    """Calculate statistics for a company building at a given level.

    Args:
        building: Building instance.
        level: Current level of the building.
        resources: List of Resource instances the building can produce.
        market: MarketData instance containing prices and transport info.
        quality: Quality level for prices.
        config: Profit calculation configuration.

    Returns:
        Dictionary with building statistics including best resource profit and ROI.
    """
    # Find best resource for this building using centralized helper
    best_resource, best_profit_data = find_best_resource_profit(
        resources, market, quality, config
    )

    if best_profit_data is None:
        return {
            "building_name": building.name,
            "building_id": building.id,
            "level": level,
            "best_resource": None,
            "hourly_profit": 0,
            "daily_profit": 0,
            "building_value": 0,
            "roi_daily": 0,
            "break_even_days": float("inf"),
            "missing_cost": True,
        }

    # Scale profit by level
    best_profit = best_profit_data["profit_per_hour"]
    hourly_profit = best_profit * level
    daily_profit = hourly_profit * 24

    # Calculate building value using centralized helper
    building_value, missing_cost = calculate_total_investment(
        building, level, market
    )

    # Calculate ROI
    roi_daily = 0.0
    break_even_days = float("inf")
    if building_value > 0:
        roi_daily = (daily_profit / building_value) * 100
        if daily_profit > 0:
            break_even_days = building_value / daily_profit

    return {
        "building_name": building.name,
        "building_id": building.id,
        "level": level,
        "best_resource": best_resource.name if best_resource else None,
        "hourly_profit": hourly_profit,
        "daily_profit": daily_profit,
        "building_value": building_value,
        "roi_daily": roi_daily,
        "break_even_days": break_even_days,
        "missing_cost": missing_cost,
    }


def calculate_upgrade_recommendations(
    buildings_with_levels: list[tuple[Building, int]],
    building_resources: dict[str, list[Resource]],
    market: MarketData,
    quality: int,
    config: ProfitConfig,
) -> list[dict]:
    """Calculate upgrade recommendations for buildings based on marginal ROI.

    The marginal ROI is the ROI of upgrading from current level to next level,
    calculated as: (additional daily profit) / (upgrade cost) * 100.

    Args:
        buildings_with_levels: List of (Building, current_level) tuples.
        building_resources: Map of building name to list of Resource instances.
        market: MarketData instance containing prices and transport info.
        quality: Quality level for prices.
        config: Profit calculation configuration.

    Returns:
        List of upgrade recommendations sorted by marginal ROI descending.
    """
    recommendations = []

    for building, current_level in buildings_with_levels:
        resources = building_resources.get(building.name, [])
        if not resources:
            continue

        # Find best profit per hour at level 1 (base profit) using centralized helper
        best_resource, best_profit_data = find_best_resource_profit(
            resources, market, quality, config
        )

        if best_profit_data is None or best_resource is None:
            continue

        best_base_profit = best_profit_data["profit_per_hour"]

        # Calculate base construction cost
        base_cost, missing_cost = building.calculate_construction_cost(market)

        if base_cost <= 0:
            continue

        # Marginal upgrade from current_level to current_level + 1
        # Upgrade cost = current_level * base_cost
        upgrade_cost = current_level * base_cost

        # Additional profit gained = base_profit per hour (one level adds 1x base)
        additional_daily_profit = best_base_profit * 24

        # Marginal ROI
        marginal_roi = 0.0
        marginal_break_even = float("inf")
        if upgrade_cost > 0:
            marginal_roi = (additional_daily_profit / upgrade_cost) * 100
            if additional_daily_profit > 0:
                marginal_break_even = upgrade_cost / additional_daily_profit

        recommendations.append(
            {
                "building_name": building.name,
                "building_id": building.id,
                "current_level": current_level,
                "next_level": current_level + 1,
                "best_resource": best_resource.name,
                "upgrade_cost": upgrade_cost,
                "additional_daily_profit": additional_daily_profit,
                "marginal_roi": marginal_roi,
                "marginal_break_even": marginal_break_even,
                "missing_cost": missing_cost,
            }
        )

    # Sort by marginal ROI descending
    recommendations.sort(key=lambda x: x["marginal_roi"], reverse=True)
    return recommendations


def calculate_retail_units_per_hour(
    retail_info: dict,
    price: float,
    quality: int,
    building_level: int,
    sales_speed_bonus: float = 0.0,
    acceleration_multiplier: float = 1.0,
    weather_multiplier: float = 1.0,
) -> float:
    """Calculate units sold per hour for retail sales.

    This implements the retail calculation formula from Sim Companies:
    - rIr: Core demand calculation based on price, quality, saturation
    - uoe: Time per 100 units with building level and bonuses
    - ufe: Convert time to units per hour

    Args:
        retail_info: Retail info dict with keys:
            - buildingLevelsNeededPerUnitPerHour
            - modeledProductionCostPerUnit
            - modeledStoreWages
            - modeledUnitsSoldAnHour
        price: Retail price per unit.
        quality: Quality level (0-12).
        building_level: Building level.
        sales_speed_bonus: Sales speed bonus percentage (default: 0.0).
        acceleration_multiplier: Acceleration multiplier (default: 1.0).
        weather_multiplier: Weather multiplier (default: 1.0).
        saturation: Market saturation (0-2, default: 0.0).

    Returns:
        Units sold per hour, or NaN if calculation is invalid.
    """
    PROFIT_PER_BUILDING_LEVEL = 320
    RETAIL_MODELING_QUALITY_WEIGHT = 0.3

    # Extract modeled constants
    building_levels_needed = retail_info.get("buildingLevelsNeededPerUnitPerHour", None)
    production_cost = retail_info.get("modeledProductionCostPerUnit", None)
    store_wages = retail_info.get("modeledStoreWages", None)
    modeled_units = retail_info.get("modeledUnitsSoldAnHour", None)
    if not (building_levels_needed and production_cost and store_wages and modeled_units):
        return float("nan")

    # rIr calculation
    # Saturation factor
    saturation = retail_info.get("saturation", 0)
    p = max(min(2 - saturation, 2), 0)
    h = max(0.9, p / 2 + 0.5)
    f = quality / 12.0

    # Profit pressure calculation
    g = (
        PROFIT_PER_BUILDING_LEVEL
        * (building_levels_needed * modeled_units + 1)
        * (p / 2 * (1 + f * RETAIL_MODELING_QUALITY_WEIGHT))
        + store_wages
    )

    # Effective units
    v = modeled_units * h
    if modeled_units == 0:
        return float("nan")
    # Cost-adjusted demand base
    b = production_cost + (g + store_wages) / v

    # Demand curve (using price - b, not price - production_cost)
    alpha = (store_wages + g) / ((b - production_cost) ** 2)
    demand_curve = g - ((price - b) ** 2) * alpha

    # Check for invalid demand curve
    if demand_curve <= 0:
        return float("nan")

    # Time demand calculation (aIr)
    # Note: The original code passes units=100 as a constant to aIr, not acceleration_multiplier
    # This is a fixed constant in the formula, separate from the acceleration_multiplier
    # which is applied later in uoe
    UNITS_CONSTANT = 100
    d = (
        (UNITS_CONSTANT * ((price - production_cost) * 3600)) - store_wages
    ) / (demand_curve + store_wages)

    # Check for invalid time
    if d <= 0:
        return float("nan")
    
    # uoe calculation: seconds per 100 units
    seconds_per_100_units = (
        (d / building_level / acceleration_multiplier)
        * (1 - sales_speed_bonus / 100.0)
        / weather_multiplier
    )

    # ufe calculation: convert to units per hour
    if seconds_per_100_units <= 0:
        return float("nan")

    return 100 * 3600 / seconds_per_100_units
