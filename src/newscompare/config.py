"""Load and validate config from YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Invalid or missing configuration."""


@dataclass
class FeedConfig:
    url: str
    name: str | None = None
    max_articles: int = 20

    @property
    def source_id(self) -> str:
        return self.name or self.url


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "gemma3:4b"
    max_tokens: int = 2048
    claim_extract_prompt: str = ""


@dataclass
class GroupingConfig:
    hours_window: int = 24
    title_similarity_threshold: float = 0.5


@dataclass
class CompareConfig:
    # 0.74–0.78: lower finds more agreements (same fact, different wording); higher reduces false matches
    claim_match_threshold: float = 0.74


@dataclass
class Config:
    feeds: list[FeedConfig] = field(default_factory=list)
    database: str = "./newscompare.db"
    fetch_timeout_seconds: int = 10
    # Domains for which we skip fetching full article body (paywall/blocking). Use feed summary only.
    skip_enrich_domains: list[str] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding_model: str = "all-MiniLM-L6-v2"
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    compare: CompareConfig = field(default_factory=CompareConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        config_path = path or os.environ.get("NEWSCOMPARE_CONFIG", "config.yaml")
        p = Path(config_path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}")
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        feeds_raw = data.get("feeds") or []
        feeds: list[FeedConfig] = []
        for item in feeds_raw:
            if isinstance(item, str):
                feeds.append(FeedConfig(url=item))
            else:
                feeds.append(
                    FeedConfig(
                        url=item["url"],
                        name=item.get("name"),
                        max_articles=item.get("max_articles", 20),
                    )
                )
        llm_data = data.get("llm") or {}
        llm = LLMConfig(
            provider=llm_data.get("provider", "ollama"),
            model=llm_data.get("model", "gemma3:4b"),
            max_tokens=llm_data.get("max_tokens", 2048),
            claim_extract_prompt=llm_data.get("claim_extract_prompt", "").strip(),
        )
        grp = data.get("grouping") or {}
        grouping = GroupingConfig(
            hours_window=grp.get("hours_window", 24),
            title_similarity_threshold=grp.get("title_similarity_threshold", 0.5),
        )
        cmp_data = data.get("compare") or {}
        compare = CompareConfig(claim_match_threshold=cmp_data.get("claim_match_threshold", 0.74))
        return cls(
            feeds=feeds,
            database=data.get("database", "./newscompare.db"),
            fetch_timeout_seconds=data.get("fetch_timeout_seconds", 10),
            skip_enrich_domains=data.get("skip_enrich_domains") or [],
            llm=llm,
            embedding_model=data.get("embedding_model", "all-MiniLM-L6-v2"),
            grouping=grouping,
            compare=compare,
        )
