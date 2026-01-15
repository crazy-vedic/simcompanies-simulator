"""Resource model for Sim Companies resources."""

from dataclasses import dataclass, field


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
        market: "MarketData",
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
        from simtools.models.market import MarketData
        
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

