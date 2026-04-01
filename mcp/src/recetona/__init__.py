"""RecetONA package."""

from .config import Settings, get_settings
from .models import FetchPayload, RecipeAnswer, SearchResult

try:
    from .services.runtime import RecetonaService
except ModuleNotFoundError:  # pragma: no cover - compatibilidad defensiva
    RecetonaService = None

__all__ = [
    "FetchPayload",
    "RecipeAnswer",
    "SearchResult",
    "Settings",
    "get_settings",
]

if RecetonaService is not None:
    __all__.append("RecetonaService")
