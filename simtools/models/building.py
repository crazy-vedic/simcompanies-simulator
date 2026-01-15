"""Building model for Sim Companies buildings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simtools.models.resource import Resource


@dataclass
class Building:
    """Represents a building in Sim Companies.

    Buildings are the production facilities that create resources. Each building
    has construction costs and a list of resources it can produce.
    """

    name: str
    id: str = ""
    cost: dict[str, int] = field(default_factory=dict)
    produces: list[str] = field(default_factory=list)
    level: int = 1

    # Resolved resources (populated after linking with API data)
    _resources: list[Resource] = field(default_factory=list, repr=False)

    @property
    def production_multiplier(self) -> float:
        """Get the production multiplier for this building level.

        Returns:
            Production multiplier (equal to level).
        """
        return float(self.level)

    @property
    def wage_multiplier(self) -> float:
        """Get the wage multiplier for this building level.

        Returns:
            Wage multiplier (equal to level).
        """
        return float(self.level)

    @classmethod
    def from_dict(cls, data: dict) -> "Building":
        """Create a Building instance from a dictionary.

        Args:
            data: Building data from buildings.json.

        Returns:
            A new Building instance.
        """
        return cls(
            name=data.get("name", ""),
            id=data.get("id", ""),
            cost=data.get("cost", {}),
            produces=data.get("produces", []),
            level=data.get("level", 1),
        )

    @classmethod
    def load_all(cls, filepath: str | Path | None = None) -> list["Building"]:
        """Load all buildings from the JSON file.

        Args:
            filepath: Path to buildings.json. If None, uses the default data location.

        Returns:
            List of Building instances.
        """
        if filepath is None:
            # Default to simtools/data/buildings.json
            filepath = Path(__file__).parent.parent / "data" / "buildings.json"

        filepath = Path(filepath)
        if not filepath.exists():
            return []

        with open(filepath, "r") as f:
            data = json.load(f)

        return [cls.from_dict(b) for b in data]

    def get_resources(self) -> list[Resource]:
        """Get the resolved Resource objects for this building.

        Returns:
            List of Resource instances that this building produces.
        """
        return self._resources

    def link_resources(self, resources: dict[str, Resource]) -> None:
        """Link this building to its Resource objects.

        Args:
            resources: Dictionary mapping resource names (lowercase) to Resource instances.
        """
        self._resources = []
        for res_name in self.produces:
            resource = resources.get(res_name.lower())
            if resource:
                resource.building_name = self.name
                self._resources.append(resource)

    def calculate_construction_cost(
        self, market: "MarketData"
    ) -> tuple[float, bool]:
        """Calculate the total construction cost for this building.

        Args:
            market: MarketData instance containing Q0 prices and name_to_id mapping.

        Returns:
            Tuple of (total_cost, missing_price_flag).
        """
        from simtools.models.market import MarketData
        
        total_cost = 0.0
        missing_price = False

        for material_name, amount in self.cost.items():
            mat_id = market.name_to_id.get(material_name.lower())
            if mat_id:
                price = market.get_price(mat_id, quality=0)
            else:
                price = 0
                missing_price = True

            if price == 0:
                missing_price = True
            total_cost += price * amount

        return total_cost, missing_price

    def calculate_upgrade_cost(
        self,
        market: "MarketData",
        target_level: int,
    ) -> tuple[float, bool]:
        """Calculate the cost to upgrade this building to a target level.

        The cost to upgrade from level L to L+1 is L * base_cost.
        Total cost is the sum of costs for each step.

        Args:
            market: MarketData instance containing Q0 prices and name_to_id mapping.
            target_level: Level to upgrade to.

        Returns:
            Tuple of (total_upgrade_cost, missing_price_flag).
        """
        from simtools.models.market import MarketData
        
        if target_level <= self.level:
            return 0.0, False

        base_cost, missing_price = self.calculate_construction_cost(market)
        
        total_upgrade_cost = 0.0
        # Calculate cost for each step: current -> current+1, ..., target-1 -> target
        # Step cost from k to k+1 is k * base_cost
        for k in range(self.level, target_level):
            total_upgrade_cost += k * base_cost

        return total_upgrade_cost, missing_price

    def produces_resource(self, resource_name: str) -> bool:
        """Check if this building produces a specific resource.

        Args:
            resource_name: Name of the resource to check.

        Returns:
            True if this building produces the resource.
        """
        return resource_name.lower() in [p.lower() for p in self.produces]


def build_resource_to_building_map(buildings: list[Building]) -> dict[str, str]:
    """Build a mapping from resource names to building names.

    Args:
        buildings: List of Building instances.

    Returns:
        Dictionary mapping resource name (lowercase) to building name.
    """
    mapping = {}
    for building in buildings:
        for res_name in building.produces:
            mapping[res_name.lower()] = building.name
    return mapping

