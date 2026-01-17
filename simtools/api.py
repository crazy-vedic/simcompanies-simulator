"""Simcotools API client for fetching game data."""

import httpx
import json
import time
from pathlib import Path
from rich.console import Console

console = Console()

# Cache duration: 1 hour in seconds
CACHE_DURATION = 3600


def _get_cache_path(filename: str) -> Path:
    """Get path to cache file.

    Args:
        filename: Base filename for cache.

    Returns:
        Path to cache file.
    """
    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{filename}.cache"


def _load_cache(filename: str) -> tuple[dict | list | None, float]:
    """Load data from cache file if valid.

    Args:
        filename: Cache filename.

    Returns:
        Tuple of (cached_data, timestamp) or (None, 0.0) if cache invalid/missing.
    """
    cache_path = _get_cache_path(filename)
    if not cache_path.exists():
        return None, 0.0

    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)
            return cache_data.get("data"), cache_data.get("timestamp", 0.0)
    except Exception:
        return None, 0.0


def _save_cache(filename: str, data: dict | list) -> None:
    """Save data to cache file.

    Args:
        filename: Cache filename.
        data: Data to cache.
    """
    cache_path = _get_cache_path(filename)
    try:
        with open(cache_path, "w") as f:
            json.dump({"data": data, "timestamp": time.time()}, f)
    except Exception:
        pass  # Silently fail if cache can't be written


def _is_cache_valid(timestamp: float) -> bool:
    """Check if cache timestamp is still valid.

    Args:
        timestamp: Cache timestamp.

    Returns:
        True if cache is valid (less than CACHE_DURATION old).
    """
    if timestamp == 0.0:
        return False
    return (time.time() - timestamp) < CACHE_DURATION


class SimcoAPI:
    """Client for the Simcotools API.

    Provides methods to fetch resource data and market prices from the
    Simcotools API for a specific realm.
    """

    def __init__(self, realm: int = 0):
        """Initialize the API client.

        Args:
            realm: The realm number to fetch data for (default: 0).
        """
        self.base_url = f"https://api.simcotools.com/v1/realms/{realm}"
        self.headers = {"accept": "application/json"}

    def get_resources(self) -> dict:
        """Fetch all resources from the API.

        Uses 1-hour file cache to avoid unnecessary network requests.

        Returns:
            Dictionary with 'resources' key containing list of resource data.
        """
        cache_key = f"resources_realm_{self.base_url.split('/')[-1]}"
        cached_data, timestamp = _load_cache(cache_key)
        
        if _is_cache_valid(timestamp) and cached_data is not None:
            console.log("[cyan]Using cached resources data[/cyan]")
            return cached_data  # type: ignore
        
        url = f"{self.base_url}/resources"
        all_resources = []

        with console.status("[bold green]Fetching resources...", spinner="dots"):
            # Try disable_pagination=True first
            try:
                response = httpx.get(
                    url, headers=self.headers, params={"disable_pagination": "True"}
                )
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
                        console.log(
                            f"Fetched all {len(data['resources'])} resources with disable_pagination."
                        )
                        return data
            except Exception as e:
                console.log(
                    f"[yellow]disable_pagination failed, falling back to manual pagination: {e}[/yellow]"
                )

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
            result = {"resources": all_resources}
            _save_cache(cache_key, result)
            return result

    def get_market_vwaps(self) -> list:
        """Fetch market Volume Weighted Average Prices.

        Uses 1-hour file cache to avoid unnecessary network requests.

        Returns:
            List of VWAP entries with resourceId, quality, and vwap values.
        """
        cache_key = f"vwaps_realm_{self.base_url.split('/')[-1]}"
        cached_data, timestamp = _load_cache(cache_key)
        
        if _is_cache_valid(timestamp) and cached_data is not None:
            console.log("[cyan]Using cached VWAPs data[/cyan]")
            return cached_data  # type: ignore
        
        url = f"{self.base_url}/market/vwaps"
        with console.status("[bold green]Fetching market VWAPs...", spinner="dots"):
            response = httpx.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            # If the API returns a dict with a key like 'vwaps'
            if isinstance(data, dict) and "vwaps" in data:
                result = data["vwaps"]
            else:
                result = data
            _save_cache(cache_key, result)
            return result  # type: ignore

    def get_company(self, user_id: int) -> dict:
        """Fetch company data for a specific user.

        Args:
            user_id: The user ID to fetch company data for.

        Returns:
            Dictionary with company data including buildings.
        """
        url = f"{self.base_url}/companies/{user_id}"
        with console.status(
            f"[bold green]Fetching company data for user {user_id}...", spinner="dots"
        ):
            response = httpx.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()

    def get_retail_info(self) -> list:
        """Fetch retail information from the Sim Companies API.

        Uses 1-hour file cache to avoid unnecessary network requests.

        Returns:
            List of retail info entries.
        """
        cache_key = "retail_info"
        cached_data, timestamp = _load_cache(cache_key)
        
        if _is_cache_valid(timestamp) and cached_data is not None:
            console.log("[cyan]Using cached retail info data[/cyan]")
            return cached_data if isinstance(cached_data, list) else []
        
        url = "https://www.simcompanies.com/api/v4/0/resources-retail-info/"
        with console.status("[bold green]Fetching retail info...", spinner="dots"):
            try:
                response = httpx.get(url, headers=self.headers, timeout=10.0)
                response.raise_for_status()
                result = response.json()
                _save_cache(cache_key, result)
                return result
            except Exception as e:
                console.log(f"[yellow]Failed to fetch retail info: {e}[/yellow]")
                # Try to load from saved file if network fails
                fallback_path = Path("data") / "retail-info.json"
                if fallback_path.exists():
                    try:
                        with open(fallback_path, "r") as f:
                            data = json.load(f)
                            # Convert back to list format if it's a dict
                            if isinstance(data, dict):
                                return list(data.values())
                            return data
                    except Exception:
                        pass
                return []

