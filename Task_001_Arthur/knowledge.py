"""Persistent ChromaDB з локальними детермінованими embeddings."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import chromadb

DOCUMENTS = [
    (
        "reserve-a",
        "criticality",
        "Для критичності A аварійний резерв не скорочують без підтвердженої заміни та інженерного погодження.",
    ),
    (
        "reserve-b",
        "criticality",
        "Для критичності B страховий запас дорівнює прогнозу на lead time плюс одна одиниця резерву.",
    ),
    (
        "reserve-c",
        "criticality",
        "Матеріали критичності C без руху 24 місяці є кандидатами на продаж або списання після технічної перевірки.",
    ),
    (
        "transfer",
        "transfer",
        "Перед зовнішнім продажем надлишку потрібно перевірити дефіцит на інших підприємствах групи та сумісність матеріалу.",
    ),
    (
        "purchase",
        "purchase",
        "Відкрите замовлення на закупівлю входить до майбутнього доступного запасу. При надлишку спочатку розглядають зупинку або зменшення замовлення.",
    ),
    (
        "repair-plan",
        "demand",
        "Затверджений ремонтний план має пріоритет над історичним середнім. Незатверджена заявка позначається як невизначеність.",
    ),
    (
        "emergency",
        "demand",
        "Аварійний попит не вважають нульовим лише через відсутність списань; потрібна оцінка критичності та наслідків простою.",
    ),
    (
        "write-off",
        "disposal",
        "Списання потребує технічного висновку, причини непридатності, кількості, вартості та людського погодження.",
    ),
    (
        "sale",
        "disposal",
        "Продаж надлишку можливий після внутрішнього балансування, перевірки ринку та виключення ремонтного попиту.",
    ),
    (
        "data-quality",
        "quality",
        "Невідомі одиниці виміру, дублікати кодів, негативні залишки та відсутні дати руху знижують впевненість; агент має просити перевірку.",
    ),
]


def _embed(text: str, size: int = 192) -> list[float]:
    words = re.findall(r"[\w'-]+", text.casefold(), flags=re.UNICODE)
    vector = [0.0] * size
    for word in words:
        digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % size
        vector[bucket] += 1 if digest[4] & 1 else -1
    norm = math.sqrt(sum(item * item for item in vector)) or 1.0
    return [item / norm for item in vector]


class KnowledgeBase:
    """Ідемпотентна обгортка над Chroma collection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection("warehouse_policies", metadata={"hnsw:space": "cosine"})
        self.collection.upsert(
            ids=[row[0] for row in DOCUMENTS],
            metadatas=[{"topic": row[1]} for row in DOCUMENTS],
            documents=[row[2] for row in DOCUMENTS],
            embeddings=[_embed(row[2]) for row in DOCUMENTS],
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[_embed(query)],
            n_results=min(top_k, self.count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": doc_id,
                "topic": meta["topic"],
                "text": text,
                "score": round(max(0.0, 1 - float(distance)), 4),
            }
            for doc_id, meta, text, distance in zip(
                result["ids"][0],
                result["metadatas"][0],
                result["documents"][0],
                result["distances"][0],
                strict=True,
            )
        ]
