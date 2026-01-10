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

