"""Huey + Redis queue configuration.

This is the single source of truth for the task queue. Both the worker
process and the dashboard client import from here.

Start the worker:
    huey_consumer notgpt.orchestration.queue.huey -w 2 -k process

Environment variables:
    REDIS_HOST      default: localhost
    REDIS_PORT      default: 6379
    REDIS_DB        default: 0
"""

from __future__ import annotations

import os

from huey import RedisHuey

huey = RedisHuey(
    name="cerebros_pipeline",
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    immediate=os.getenv("HUEY_IMMEDIATE", "").lower() in ("1", "true"),
    results=True,
    store_none=False,
)
