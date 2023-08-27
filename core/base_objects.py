from enum import Enum
from pydantic import BaseModel


class ConceptType(Enum):
    """Concept types for teaching the Diffusion model"""

    OBJECT = "object"
    STYLE = "style"
