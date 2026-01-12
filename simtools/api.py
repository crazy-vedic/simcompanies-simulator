"""Simcotools API client for fetching game data."""

import httpx
from rich.console import Console

console = Console()


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

        Returns:
            Dictionary with 'resources' key containing list of resource data.
        """
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
            return {"resources": all_resources}

    def get_market_vwaps(self) -> list:
        """Fetch market Volume Weighted Average Prices.

        Returns:
            List of VWAP entries with resourceId, quality, and vwap values.
        """
        url = f"{self.base_url}/market/vwaps"
        with console.status("[bold green]Fetching market VWAPs...", spinner="dots"):
            response = httpx.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            # If the API returns a dict with a key like 'vwaps'
            if isinstance(data, dict) and "vwaps" in data:
                return data["vwaps"]
            return data

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

