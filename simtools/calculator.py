"""Profit and ROI calculation logic for simtools."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simtools.models.building import Building
    from simtools.models.resource import Resource


@dataclass
class ProfitConfig:
    """Configuration for profit calculations."""

    quality: int = 0
    abundance: float = 90.0
    admin_overhead: float = 0.0
    is_contract: bool = False
    has_robots: bool = False


def calculate_all_profits(
    resources: list[Resource],
    price_map: dict[int, float],
    transport_price: float,
    config: ProfitConfig,
) -> list[dict]:
    """Calculate profits for all resources.

    Args:
        resources: List of Resource instances to calculate profits for.
        price_map: Map of resource ID to price at the target quality.
        transport_price: Price per transport unit.
        config: Profit calculation configuration.

    Returns:
        List of profit dictionaries sorted by profit_per_hour descending.
    """
    profits = []

    for res in resources:
        selling_price = price_map.get(res.id, 0)
        if selling_price == 0:
            continue

        profit_data = res.calculate_profit(
            selling_price=selling_price,
            input_prices=price_map,
            transport_price=transport_price,
            abundance=config.abundance,
            admin_overhead=config.admin_overhead,
            is_contract=config.is_contract,
            has_robots=config.has_robots,
        )
        profits.append(profit_data)

    # Sort by profit descending
    profits.sort(key=lambda x: x["profit_per_hour"], reverse=True)
    return profits


def calculate_building_roi(
    buildings: list[Building],
    profits: list[dict],
    q0_price_map: dict[int, float],
    name_to_id: dict[str, int],
) -> list[dict]:
    """Calculate ROI for buildings based on their best performing resource.

    Args:
        buildings: List of Building instances.
        profits: List of profit dictionaries from calculate_all_profits.
        q0_price_map: Map of resource ID to Q0 price for building costs.
        name_to_id: Map of resource name (lowercase) to resource ID.

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
            res_name_lower = res_name.lower()
            if res_name_lower in res_profit_map:
                has_relevant_resource = True
                p_data = res_profit_map[res_name_lower]
                if p_data["profit_per_hour"] > best_profit:
                    best_profit = p_data["profit_per_hour"]
                    best_name = p_data["name"]

        if not has_relevant_resource:
            continue

        # Calculate building cost
        total_cost, missing_cost = building.calculate_construction_cost(
            q0_price_map, name_to_id
        )

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
    q0_price_map: dict[int, float],
    name_to_id: dict[str, int],
    max_level: int = 20,
    step_mode: bool = False,
) -> list[dict]:
    """Calculate ROI for a single building across multiple levels.

    Args:
        building: The Building instance.
        best_profit_data: Profit data for the best resource at level 1.
        q0_price_map: Map of resource ID to Q0 price for building costs.
        name_to_id: Map of resource name (lowercase) to resource ID.
        max_level: Maximum level to calculate up to.
        step_mode: If True, calculate ROI for each step L -> L+1 based on 
                   that step's cost and the additional profit gained.

    Returns:
        List of ROI dictionaries for each level/step.
    """
    base_cost, missing_cost = building.calculate_construction_cost(q0_price_map, name_to_id)
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
            upgrade_cost, _ = building.calculate_upgrade_cost(q0_price_map, level, name_to_id)
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
    current_prices: dict[int, float],
    q0_prices: dict[int, float],
    transport_price: float,
    name_to_id: dict[str, int],
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
        current_prices: Price map for revenue and inputs.
        q0_prices: Price map for construction costs.
        transport_price: Price per transport unit.
        name_to_id: Resource name to ID map.
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
    selling_price = current_prices.get(resource.id, 0)
    p_100 = resource.calculate_profit(
        selling_price=selling_price,
        input_prices=current_prices,
        transport_price=transport_price,
        abundance=100.0,
        admin_overhead=profit_config.admin_overhead,
        is_contract=profit_config.is_contract,
        has_robots=profit_config.has_robots,
    )
    p_0 = resource.calculate_profit(
        selling_price=selling_price,
        input_prices=current_prices,
        transport_price=transport_price,
        abundance=0.0,
        admin_overhead=profit_config.admin_overhead,
        is_contract=profit_config.is_contract,
        has_robots=profit_config.has_robots,
    )
    
    hourly_fixed_cost = -p_0["profit_per_hour"]  # Wages
    hourly_variable_profit_at_100 = p_100["profit_per_hour"] + hourly_fixed_cost
    
    # Base construction cost (Level 1)
    base_cost, missing_cost = building.calculate_construction_cost(q0_prices, name_to_id)
    
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
    input_prices: dict[int, float],
    transport_price: float,
    config: ProfitConfig,
) -> dict:
    """Compare selling on market vs selling via contract with custom price.

    Args:
        resource: Resource instance to compare.
        market_price: Market price from VWAP.
        contract_price: User-defined contract price per unit.
        input_prices: Map of resource ID to price for input materials.
        transport_price: Price per transport unit.
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
        input_prices=input_prices,
        transport_price=transport_price,
        abundance=config.abundance,
        admin_overhead=config.admin_overhead,
        is_contract=False,
        has_robots=config.has_robots,
    )

    # Calculate contract mode (0% fee, 50% transport) with custom price
    contract_data = resource.calculate_profit(
        selling_price=contract_price,
        input_prices=input_prices,
        transport_price=transport_price,
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
    price_map: dict[int, float],
    transport_price: float,
    config: ProfitConfig,
    q0_price_map: dict[int, float],
    name_to_id: dict[str, int],
) -> dict:
    """Calculate statistics for a company building at a given level.

    Args:
        building: Building instance.
        level: Current level of the building.
        resources: List of Resource instances the building can produce.
        price_map: Map of resource ID to price at target quality.
        transport_price: Price per transport unit.
        config: Profit calculation configuration.
        q0_price_map: Map of resource ID to Q0 price for building costs.
        name_to_id: Map of resource name (lowercase) to resource ID.

    Returns:
        Dictionary with building statistics including best resource profit and ROI.
    """
    # Find best resource for this building
    best_profit = -float("inf")
    best_resource_name = None
    best_profit_data = None

    for res in resources:
        selling_price = price_map.get(res.id, 0)
        if selling_price <= 0:
            continue

        profit_data = res.calculate_profit(
            selling_price=selling_price,
            input_prices=price_map,
            transport_price=transport_price,
            abundance=config.abundance,
            admin_overhead=config.admin_overhead,
            is_contract=config.is_contract,
            has_robots=config.has_robots,
        )

        if profit_data["profit_per_hour"] > best_profit:
            best_profit = profit_data["profit_per_hour"]
            best_resource_name = res.name
            best_profit_data = profit_data

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
    hourly_profit = best_profit * level
    daily_profit = hourly_profit * 24

    # Calculate building value (total investment to reach this level)
    base_cost, missing_cost = building.calculate_construction_cost(
        q0_price_map, name_to_id
    )
    if level <= 1:
        building_value = base_cost
    else:
        # Total upgrade cost = base_cost * level * (level - 1) / 2
        upgrade_cost = base_cost * level * (level - 1) / 2
        building_value = base_cost + upgrade_cost

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
        "best_resource": best_resource_name,
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
    price_map: dict[int, float],
    transport_price: float,
    config: ProfitConfig,
    q0_price_map: dict[int, float],
    name_to_id: dict[str, int],
) -> list[dict]:
    """Calculate upgrade recommendations for buildings based on marginal ROI.

    The marginal ROI is the ROI of upgrading from current level to next level,
    calculated as: (additional daily profit) / (upgrade cost) * 100.

    Args:
        buildings_with_levels: List of (Building, current_level) tuples.
        building_resources: Map of building name to list of Resource instances.
        price_map: Map of resource ID to price at target quality.
        transport_price: Price per transport unit.
        config: Profit calculation configuration.
        q0_price_map: Map of resource ID to Q0 price for building costs.
        name_to_id: Map of resource name (lowercase) to resource ID.

    Returns:
        List of upgrade recommendations sorted by marginal ROI descending.
    """
    recommendations = []

    for building, current_level in buildings_with_levels:
        resources = building_resources.get(building.name, [])
        if not resources:
            continue

        # Find best profit per hour at level 1 (base profit)
        best_base_profit = -float("inf")
        best_resource_name = None

        for res in resources:
            selling_price = price_map.get(res.id, 0)
            if selling_price <= 0:
                continue

            profit_data = res.calculate_profit(
                selling_price=selling_price,
                input_prices=price_map,
                transport_price=transport_price,
                abundance=config.abundance,
                admin_overhead=config.admin_overhead,
                is_contract=config.is_contract,
                has_robots=config.has_robots,
            )

            if profit_data["profit_per_hour"] > best_base_profit:
                best_base_profit = profit_data["profit_per_hour"]
                best_resource_name = res.name

        if best_base_profit <= -float("inf") or best_resource_name is None:
            continue

        # Calculate base construction cost
        base_cost, missing_cost = building.calculate_construction_cost(
            q0_price_map, name_to_id
        )

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
                "best_resource": best_resource_name,
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
