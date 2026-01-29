"""API modules"""

from .routes import router as api_router
from .admin import router as admin_router
from .images import router as images_router
from .gemini import router as gemini_router

__all__ = ["api_router", "admin_router", "images_router", "gemini_router"]
