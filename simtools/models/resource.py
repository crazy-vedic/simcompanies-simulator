"""Resource model for Sim Companies resources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simtools.models.market import MarketData


@dataclass
class ResourceInput:
    """Represents an input material required for production."""

    id: int
    name: str
    quantity: float


@dataclass
class Resource:
    """Represents a producible resource in Sim Companies.

    Resources are the items that can be produced by buildings. Each resource
    has production characteristics (rate, wages, inputs) and market properties
    (transportation costs, retail info).
    """

    id: int
    name: str
    produced_per_hour: float
    wages: float
    transportation: float
    inputs: dict[int, ResourceInput] = field(default_factory=dict)
    is_research: bool = False
    speed_modifier: float = 0
    retail_info: list[dict] | None = None

    # These are set based on external data (abundance_resources.json, seasonal_resources.json)
    is_abundance: bool = False
    is_seasonal: bool = False

    # Set after building association
    building_name: str | None = None

    @classmethod
    def from_api_data(
        cls,
        data: dict,
        abundance_resources: list[str] | None = None,
        seasonal_resources: list[str] | None = None,
    ) -> "Resource":
        """Create a Resource instance from API response data.

        Args:
            data: Raw resource data from the Simcotools API.
            abundance_resources: List of resource names that use abundance calculation.
            seasonal_resources: List of resource names that are seasonal.

        Returns:
            A new Resource instance.
        """
        abundance_resources = abundance_resources or []
        seasonal_resources = seasonal_resources or []

        name = data.get("name", "")
        name_lower = name.lower()

        # Parse inputs
        inputs = {}
        for input_id_str, input_info in data.get("inputs", {}).items():
            input_id = int(input_id_str)
            inputs[input_id] = ResourceInput(
                id=input_id,
                name=input_info.get("name", ""),
                quantity=input_info.get("quantity", 0),
            )

        return cls(
            id=data.get("id", 0),
            name=name,
            produced_per_hour=data.get("producedAnHour", 0),
            wages=data.get("wages", 0),
            transportation=data.get("transportation", 0),
            inputs=inputs,
            is_research=data.get("isResearch", False),
            speed_modifier=data.get("speedModifier", 0),
            retail_info=data.get("retailInfo"),
            is_abundance=name_lower in [r.lower() for r in abundance_resources],
            is_seasonal=name_lower in [r.lower() for r in seasonal_resources],
        )

    def get_effective_production(self, abundance: float = 100.0) -> float:
        """Get the effective production rate, accounting for abundance if applicable.

        Args:
            abundance: Abundance percentage (0-100) for mine/well resources.

        Returns:
            Effective production rate per hour.
        """
        rate = self.produced_per_hour
        if self.is_abundance:
            rate *= abundance / 100.0
        return rate

    def calculate_profit(
        self,
        selling_price: float,
        market: MarketData,
        quality: int = 0,
        abundance: float = 100.0,
        admin_overhead: float = 0.0,
        is_contract: bool = False,
        has_robots: bool = False,
    ) -> dict:
        """Calculate profit metrics for this resource.

        Args:
            selling_price: Price per unit at the target quality.
            market: MarketData instance containing prices and transport info.
            quality: Quality level for input prices (default: 0).
            abundance: Abundance percentage for mine/well resources.
            admin_overhead: Administrative overhead percentage to add to wages.
            is_contract: If True, use contract mode (0% fee, 50% transport).
            has_robots: If True, apply 3% wage reduction for robots.

        Returns:
            Dictionary with profit breakdown:
                - profit_per_hour: Net profit per hour
                - revenue_per_hour: Gross revenue per hour
                - market_fee_per_hour: Market fee per hour
                - costs_per_hour: Wages + admin + input costs per hour
                - transport_costs_per_hour: Transportation costs per hour
                - missing_input_price: True if any input price was missing
        """
        produced_per_hour = self.get_effective_production(abundance)

        # Wages with optional robots reduction (3%) and admin overhead
        base_wages = self.wages
        if has_robots:
            base_wages *= 0.97

        admin_cost = base_wages * (admin_overhead / 100.0)
        total_wages = base_wages + admin_cost

        # Revenue
        revenue_per_hour = selling_price * produced_per_hour

        # Market fee (4% normally, 0% for contracts)
        market_fee_pct = 0.0 if is_contract else 0.04
        market_fee_per_hour = revenue_per_hour * market_fee_pct

        # Input costs
        input_costs_per_hour = 0.0
        missing_input_price = False
        for input_id, input_info in self.inputs.items():
            price = market.get_price(input_id, quality)
            if price == 0:
                missing_input_price = True
            input_costs_per_hour += price * input_info.quantity * produced_per_hour

        # Transportation costs (50% reduction for contracts)
        transport_cost_per_unit = self.transportation * market.transport_price
        if is_contract:
            transport_cost_per_unit *= 0.5
        transport_costs_per_hour = transport_cost_per_unit * produced_per_hour

        # Total profit
        profit_per_hour = (
            revenue_per_hour
            - market_fee_per_hour
            - total_wages
            - input_costs_per_hour
            - transport_costs_per_hour
        )

        return {
            "name": self.name,
            "profit_per_hour": profit_per_hour,
            "revenue_per_hour": revenue_per_hour,
            "market_fee_per_hour": market_fee_per_hour,
            "costs_per_hour": total_wages + input_costs_per_hour,
            "transport_costs_per_hour": transport_costs_per_hour,
            "missing_input_price": missing_input_price,
            "is_abundance_res": self.is_abundance,
        }

    def calculate_retail_profit(
        self,
        market: MarketData,
        quality: int = 0,
        building_level: int = 1,
        sales_speed_bonus: float = 0.0,
        admin_overhead: float = 0.0,
        input_cost_per_unit: float = 0.0,
    ) -> dict:
        """Calculate retail profit metrics for this resource.

        Retail is a "virtual production" where:
        - Input: The resource being sold
        - Output: Money (profit from retail sales)
        - No market fees (money goes directly to you)
        - No transport costs on sales

        Args:
            market: MarketData instance containing prices and transport info.
            quality: Quality level for retail data lookup (default: 0).
            building_level: Level of the retail building (default: 1).
            sales_speed_bonus: Sales speed bonus as decimal (e.g., 0.01 for 1%).
            admin_overhead: Administrative overhead percentage to add to wages.
            input_cost_per_unit: Cost to acquire one unit of the resource.

        Returns:
            Dictionary with retail profit breakdown:
                - profit_per_hour: Net profit per hour
                - revenue_per_hour: Gross revenue per hour (retail_price * units_sold)
                - wages_per_hour: Total wages per hour
                - units_sold_per_hour: Units sold per hour
                - revenue_less_wages_per_unit: Revenue less wages per unit
                - retail_price: Retail selling price per unit
                - missing_input_price: True if input cost is missing
        """
        if not self.retail_info:
            return {
                "name": f"{self.name} (Retail)",
                "profit_per_hour": 0.0,
                "revenue_per_hour": 0.0,
                "wages_per_hour": 0.0,
                "units_sold_per_hour": 0.0,
                "revenue_less_wages_per_unit": 0.0,
                "retail_price": 0.0,
                "missing_input_price": True,
                "is_abundance_res": False,
                "market_fee_per_hour": 0.0,
                "costs_per_hour": 0.0,
                "transport_costs_per_hour": 0.0,
            }

        # Find retail data for quality
        retail_data = next(
            (r for r in self.retail_info if r.get("quality") == quality), None
        )
        if not retail_data:
            return {
                "name": f"{self.name} (Retail)",
                "profit_per_hour": 0.0,
                "revenue_per_hour": 0.0,
                "wages_per_hour": 0.0,
                "units_sold_per_hour": 0.0,
                "revenue_less_wages_per_unit": 0.0,
                "retail_price": 0.0,
                "missing_input_price": True,
                "is_abundance_res": False,
                "market_fee_per_hour": 0.0,
                "costs_per_hour": 0.0,
                "transport_costs_per_hour": 0.0,
            }

        # Extract retail data
        building_levels_per_unit = retail_data.get(
            "buildingLevelsNeededPerUnitPerHour", 0
        )
        sales_wages = retail_data.get("salesWages", 0)
        retail_price = retail_data.get("averagePrice", 0)

        if retail_price <= 0:
            return {
                "name": f"{self.name} (Retail)",
                "profit_per_hour": 0.0,
                "revenue_per_hour": 0.0,
                "wages_per_hour": 0.0,
                "units_sold_per_hour": 0.0,
                "revenue_less_wages_per_unit": 0.0,
                "retail_price": retail_price,
                "missing_input_price": True,
                "is_abundance_res": False,
                "market_fee_per_hour": 0.0,
                "costs_per_hour": 0.0,
                "transport_costs_per_hour": 0.0,
            }

        # Calculate units sold per hour
        # Use buildingLevelsNeededPerUnitPerHour which represents how many building
        # levels are needed to sell one unit per hour. The inverse gives units per
        # building level per hour.
        if building_levels_per_unit > 0:
            units_sold_per_hour = (
                (1.0 / building_levels_per_unit)
                * building_level
                * (1.0 + sales_speed_bonus)
            )
        else:
            return {
                "name": f"{self.name} (Retail)",
                "profit_per_hour": 0.0,
                "revenue_per_hour": 0.0,
                "wages_per_hour": 0.0,
                "units_sold_per_hour": 0.0,
                "revenue_less_wages_per_unit": 0.0,
                "retail_price": retail_price,
                "missing_input_price": True,
                "is_abundance_res": False,
                "market_fee_per_hour": 0.0,
                "costs_per_hour": 0.0,
                "transport_costs_per_hour": 0.0,
            }

        # Calculate wages per hour
        # Always use salesWages with admin overhead (API doesn't know user's overhead)
        wages_per_hour = sales_wages * building_level * (1.0 + admin_overhead / 100.0)

        # Calculate revenue less wages per unit
        if units_sold_per_hour > 0:
            revenue_less_wages_per_unit = retail_price - (
                wages_per_hour / units_sold_per_hour
            )
        else:
            revenue_less_wages_per_unit = 0.0

        # Calculate revenue per hour
        revenue_per_hour = retail_price * units_sold_per_hour

        # Calculate profit per hour
        # profit = (retail_price * units_sold) - wages - (input_cost_per_unit * units_sold)
        profit_per_hour = (
            revenue_per_hour - wages_per_hour - (input_cost_per_unit * units_sold_per_hour)
        )

        return {
            "name": f"{self.name} (Retail)",
            "profit_per_hour": profit_per_hour,
            "revenue_per_hour": revenue_per_hour,
            "wages_per_hour": wages_per_hour,
            "units_sold_per_hour": units_sold_per_hour,
            "revenue_less_wages_per_unit": revenue_less_wages_per_unit,
            "retail_price": retail_price,
            "missing_input_price": input_cost_per_unit == 0.0,
            "is_abundance_res": False,  # Retail entries are not abundance-based
            # Add fields expected by display functions
            "market_fee_per_hour": 0.0,  # No market fees for retail
            "costs_per_hour": wages_per_hour + (input_cost_per_unit * units_sold_per_hour),
            "transport_costs_per_hour": 0.0,  # No transport costs for retail sales
        }

