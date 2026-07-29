"""ASGI entry for uvicorn matching_service.main:app"""

from matching_service.presentation.api import app

__all__ = ["app"]
