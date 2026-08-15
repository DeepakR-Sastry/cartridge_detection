"""
Detections -> ranked candidates for a set of queried products.

Sits above the recognizer and knows nothing about which model produced the
detections. Currently exact vocabulary match after uppercasing, which scored
97.1% find rate / 9.4% wrong box rate on the 29-photo eval set. Deliberately
plain: a height filter was measured and rejected (correct and wrong box heights
overlap too much to separate).
"""

import re
from dataclasses import dataclass

from .recognize import Detection

TOP_K = 3
PRODUCT = re.compile(r"^(\d{2,4})(XL)?$")


@dataclass(frozen=True)
class Product:
    number: str
    xl: bool

    @classmethod
    def parse(cls, s: str) -> "Product | None":
        m = PRODUCT.match(s.strip().upper())
        return cls(m.group(1), bool(m.group(2))) if m else None

    def __str__(self) -> str:
        return self.number + ("XL" if self.xl else "")


@dataclass(frozen=True)
class Candidate:
    product: Product
    box: tuple[int, int, int, int]
    confidence: float


def match(
    detections: list[Detection],
    queries: list[Product],
    top_k: int = TOP_K,
) -> dict[Product, list[Candidate]]:
    """
    Returns one entry per query that got at least one candidate, ranked by
    confidence. A query with no candidates is absent from the result, which is
    how "not found" is expressed. Absence is an answer, not a failure.
    """
    wanted = set(queries)
    found: dict[Product, list[Candidate]] = {}

    for d in detections:
        p = Product.parse(d.text)
        if p is None or p not in wanted:
            continue
        found.setdefault(p, []).append(
            Candidate(product=p, box=d.box, confidence=d.confidence)
        )

    for cands in found.values():
        cands.sort(key=lambda c: -c.confidence)
        del cands[top_k:]

    return found