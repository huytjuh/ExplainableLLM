from S01_client import GeminiFlashLiteClient
from mcp_overview import mcp_learning_map
from model_families import model_family_map
from vector_search import InMemoryVectorStore


def test_roadmap_packages_are_importable():
    families = model_family_map()
    assert any(family.name == "Transformers" for family in families)
    assert any(component.name == "Tools" for component in mcp_learning_map())
    assert GeminiFlashLiteClient.__name__ == "GeminiFlashLiteClient"
    assert InMemoryVectorStore.__name__ == "InMemoryVectorStore"
