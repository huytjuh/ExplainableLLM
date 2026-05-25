"""Index documents into a small in-memory vector store.

The indexing flow mirrors a typical RAG tutorial:

1. Load documents from files.
2. Split documents into chunks.
3. Embed each chunk.
4. Store chunk vectors with metadata.
5. Search the vector store at query time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, str]=field(default_factory=dict)

@dataclass(frozen=True)
class TextChunk:
    id: str
    text: str
    metadata: dict[str, str]=field(default_factory=dict)

@dataclass(frozen=True)
class Embedding:
    embedding: list[float]

@dataclass(frozen=True)
class VectorStore:
    vectorstore: list[Embedding]=[]


""" 
Embeddings:
- HashingEmbeddingModel
- OpenAIEmbeddingModel

Vector Stores:
- InMemoryVectorStore
- PineconeVectorStore
- ChromaVectorStore
- FAISSVectorStore
"""

class RAGIndexing:
    """Orchestrate loading, chunking, embedding, indexing, and searching."""

    def __init__(self):
        

    def index_documents(self, documents: list[Document]) -> int:
        pass

    def index_chunks(self, chunks: list[TextChunk]) -> int:
        pass

    def index_embeddings(self, embeddings: list[Embedding]) -> int:
        pass

    def vector_store(self) -> VectorStore:
        pass

    def search(self, query: str) -> list[SearchResult]:
        pass


@dataclass(frozen=True)
class VectorRecord:
    id: str
    text: str
    vector: list[float]
    metadata: dict[str, str]=field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    record: VectorRecord
    score: float


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: list[VectorRecord]=[]

    def add(self, record_id: str, text: str, vector: list[float], metadata: dict[str, str] | None=None) -> None:
        self._records.append(VectorRecord(record_id, text, vector, metadata or {}))

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int=3,
        metadata_filter: dict[str, str] | None=None,
    ) -> list[SearchResult]:
        results: list[SearchResult]=[]
        for record in self._records:
            if metadata_filter and any(record.metadata.get(key) != value for key, value in metadata_filter.items()):
                continue
            results.append(SearchResult(record=record, score=cosine_similarity(query_vector, record.vector)))
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class TextDocumentLoader:
    """Load plain-text files into Document objects."""

    def load_file(self, path: str | Path) -> Document:
        return load_text_file(path)

    def load_directory(self, directory: str | Path, *, pattern: str="*.txt") -> list[Document]:
        folder = Path(directory)
        return [self.load_file(path) for path in sorted(folder.glob(pattern))]


class WordChunker:
    """Split a document into overlapping word chunks."""

    def __init__(self, *, chunk_size: int=180, overlap: int=30) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[TextChunk]:
        chunks = []
        for index, text in enumerate(
            chunk_text(document.text, chunk_size=self.chunk_size, overlap=self.overlap)
        ):
            chunks.append(
                TextChunk(
                    id=f"{document.id}-chunk-{index}",
                    text=text,
                    metadata={**document.metadata, "document_id": document.id, "chunk_index": str(index)},
                )
            )
        return chunks


class RAGIndexer:
    """Orchestrate loading, chunking, embedding, indexing, and searching."""

    def __init__(
        self,
        *,
        loader: TextDocumentLoader | None=None,
        chunker: WordChunker | None=None,
        embedding_model: HashingEmbeddingModel | None=None,
        vector_store: InMemoryVectorStore | None=None,
    ) -> None:
        self.loader = loader or TextDocumentLoader()
        self.chunker = chunker or WordChunker()
        self.embedding_model = embedding_model or HashingEmbeddingModel()
        self.vector_store = vector_store or InMemoryVectorStore()

    def index_file(self, path: str | Path) -> int:
        document = self.loader.load_file(path)
        return self.index_documents([document])

    def index_directory(self, directory: str | Path, *, pattern: str="*.txt") -> int:
        documents = self.loader.load_directory(directory, pattern=pattern)
        return self.index_documents(documents)

    def index_documents(self, documents: list[Document]) -> int:
        indexed_chunks = 0
        for document in documents:
            for chunk in self.chunker.split(document):
                self.vector_store.add(
                    chunk.id,
                    chunk.text,
                    self.embedding_model.embed(chunk.text),
                    chunk.metadata,
                )
                indexed_chunks += 1
        return indexed_chunks

    def search(
        self,
        query: str,
        *,
        top_k: int=3,
        metadata_filter: dict[str, str] | None=None,
    ) -> list[SearchResult]:
        query_vector = self.embedding_model.embed(query)
        return self.vector_store.search(query_vector, top_k=top_k, metadata_filter=metadata_filter)

