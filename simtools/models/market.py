"""Market data model for centralized pricing information."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketData:
    """Centralized market price data.
    
    This class encapsulates all pricing information needed for calculations,
    including:
    - VWAP prices at different qualities for all resources
    - Transport pricing
    - Resource name to ID mapping
    
    Attributes:
        vwaps: Nested dict mapping resource_id -> {quality -> price}
        transport_price: Price per transport unit (Q0)
        name_to_id: Map of resource name (lowercase) to resource ID
    """
    
    vwaps: dict[int, dict[int, float]]
    transport_price: float
    name_to_id: dict[str, int]
    
    def get_price(self, resource_id: int, quality: int = 0) -> float:
        """Get price for a resource at a specific quality.
        
        Args:
            resource_id: The resource ID.
            quality: Quality level (default: 0).
            
        Returns:
            Price at the specified quality, or 0.0 if not found.
        """
        return self.vwaps.get(resource_id, {}).get(quality, 0.0)
    
    def get_price_map(self, quality: int = 0) -> dict[int, float]:
        """Get a flat price map for a specific quality level.
        
        This is a convenience method for legacy code that expects
        a simple dict[int, float] mapping.
        
        Args:
            quality: Quality level to extract prices for.
            
        Returns:
            Dictionary mapping resource_id -> price at specified quality.
        """
        price_map = {}
        for resource_id, qualities in self.vwaps.items():
            if quality in qualities:
                price_map[resource_id] = qualities[quality]
        return price_map
    
    @classmethod
    def from_api_response(
        cls,
        vwaps_data: list[dict],
        resources_data: list[dict],
    ) -> "MarketData":
        """Build MarketData from API response.
        
        Args:
            vwaps_data: List of VWAP entries with resourceId, quality, and vwap values.
            resources_data: List of resource data from API.
            
        Returns:
            A new MarketData instance.
        """
        # Build nested vwaps structure: resource_id -> {quality -> price}
        vwaps: dict[int, dict[int, float]] = {}
        
        for entry in vwaps_data:
            if not isinstance(entry, dict):
                continue
                
            r_id = entry.get("resourceId")
            quality = entry.get("quality")
            vwap = entry.get("vwap")
            
            if r_id is not None and quality is not None and vwap is not None:
                r_id = int(r_id)
                quality = int(quality)
                
                if r_id not in vwaps:
                    vwaps[r_id] = {}
                vwaps[r_id][quality] = float(vwap)
        
        # Build name to ID mapping
        name_to_id = {}
        for res in resources_data:
            if isinstance(res, dict):
                name = res.get("name", "")
                res_id = res.get("id")
                if name and res_id is not None:
                    name_to_id[name.lower()] = int(res_id)
        
        # Extract transport price
        transport_id = name_to_id.get("transport")
        transport_price = 0.0
        
        if transport_id is not None:
            transport_price = vwaps.get(transport_id, {}).get(0, 0.0)
        
        return cls(
            vwaps=vwaps,
            transport_price=transport_price,
            name_to_id=name_to_id,
        )
