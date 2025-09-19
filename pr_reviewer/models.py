from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class Triplet:
    id: str
    title: str
    before: str
    comment: str
    after: str
    meta: Dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "before": self.before,
            "comment": self.comment,
            "after": self.after,
            "meta": self.meta,
        }