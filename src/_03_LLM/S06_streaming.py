@dataclass(frozen=True)
class StreamChunk:
    text: str
    provider: str
    model: str
    raw_chunk: Any | None=field(default=None, repr=False)
