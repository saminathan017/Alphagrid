"""
core/events.py
Lightweight async event bus for decoupled communication between
trading system components (data → strategy → risk → execution).
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Awaitable
from loguru import logger


class EventType(Enum):
    # Data events
    MARKET_DATA      = auto()
    NEWS_ARTICLE     = auto()
    SENTIMENT_UPDATE = auto()

    # Model events
    PREDICTION       = auto()
    SIGNAL_GENERATED = auto()

    # Order lifecycle
    ORDER_SUBMITTED  = auto()
    ORDER_FILLED     = auto()
    ORDER_CANCELLED  = auto()
    ORDER_REJECTED   = auto()

    # Portfolio events
    POSITION_OPENED  = auto()
    POSITION_CLOSED  = auto()
    POSITION_UPDATED = auto()

    # Risk events
    RISK_BREACH      = auto()
    STOP_TRIGGERED   = auto()
    DRAWDOWN_ALERT   = auto()

    # System events
    SYSTEM_START     = auto()
    SYSTEM_STOP      = auto()
    HEARTBEAT        = auto()
    ERROR            = auto()


@dataclass
class Event:
    """Base event container."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __repr__(self) -> str:
        return f"Event({self.event_type.name}, source={self.source}, ts={self.timestamp:%H:%M:%S})"


# Type alias
Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """
    Async pub/sub event bus.
    Components subscribe to event types and receive async callbacks.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = {}
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register an async handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__qualname__} to {event_type.name}")

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """Remove a handler registration."""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)

    async def publish(self, event: Event) -> None:
        """Enqueue an event for async dispatch."""
        await self._queue.put(event)

    def publish_sync(self, event: Event) -> None:
        """Synchronous publish (fire-and-forget into event loop)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.publish(event))
            else:
                loop.run_until_complete(self.publish(event))
        except RuntimeError:
            asyncio.run(self.publish(event))

    async def _dispatch(self, event: Event) -> None:
        """Dispatch a single event to all registered handlers."""
        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            return
        tasks = [asyncio.create_task(h(event)) for h in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for handler, result in zip(handlers, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Handler {handler.__qualname__} raised exception "
                    f"for {event.event_type.name}: {result}"
                )

    async def run(self) -> None:
        """Main event processing loop."""
        self._running = True
        logger.info("EventBus started.")
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(event)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"EventBus error: {e}")

    def stop(self) -> None:
        self._running = False
        logger.info("EventBus stopped.")


# Global singleton event bus
event_bus = EventBus()
