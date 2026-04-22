"""
WebSocketManager – real-time message streaming via FastAPI WebSockets.

Manages client connections, session mapping, and typed message dispatch
for progress updates and results.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time AEGIS updates.

    Thread-safe with asyncio locks for concurrent access.
    """

    def __init__(self) -> None:
        # session_id -> WebSocket
        self._connections: Dict[str, WebSocket] = {}
        # Reverse map: WebSocket -> session_id
        self._ws_to_session: Dict[WebSocket, str] = {}
        self._lock = asyncio.Lock()
        self._connect_count: int = 0
        self._message_count: int = 0

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept a WebSocket connection and register it under *session_id*.

        If a connection already exists for *session_id*, it is disconnected
        first to prevent duplicates.
        """
        async with self._lock:
            # Disconnect existing session if any
            if session_id in self._connections:
                old_ws = self._connections[session_id]
                logger.info(
                    "Replacing existing connection for session %s", session_id
                )
                self._ws_to_session.pop(old_ws, None)
                try:
                    await old_ws.close(code=1008, reason="Replaced by new connection")
                except Exception as exc:
                    logger.debug("Error closing old connection: %s", exc)

            await websocket.accept()
            self._connections[session_id] = websocket
            self._ws_to_session[websocket] = session_id
            self._connect_count += 1

            logger.info(
                "WebSocket connected: session=%s (total_active=%d, total_connects=%d)",
                session_id,
                len(self._connections),
                self._connect_count,
            )

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection."""
        async with self._lock:
            session_id = self._ws_to_session.pop(websocket, None)
            if session_id and session_id in self._connections:
                if self._connections[session_id] is websocket:
                    del self._connections[session_id]
            logger.info(
                "WebSocket disconnected: session=%s (remaining=%d)",
                session_id,
                len(self._connections),
            )

    # ------------------------------------------------------------------
    # Message sending
    # ------------------------------------------------------------------
    async def send_message(
        self,
        session_id: str,
        message: Any,
        msg_type: str = "message",
    ) -> bool:
        """Send a typed message to a specific session.

        Returns True if the message was sent successfully.
        """
        async with self._lock:
            ws = self._connections.get(session_id)
            if ws is None:
                logger.debug("No connection for session %s", session_id)
                return False

        payload = self._build_payload(message, msg_type)
        try:
            await ws.send_json(payload)
            # Fix CRIT-05: increment under lock — non-atomic read-modify-write
            # on a shared integer is a race condition under concurrent sends.
            async with self._lock:
                self._message_count += 1
            return True
        except Exception as exc:
            logger.warning("Failed to send to session %s: %s", session_id, exc)
            # Connection may be stale – clean up
            await self.disconnect(ws)
            return False

    async def broadcast(self, message: Any, msg_type: str = "broadcast") -> int:
        """Send a message to all connected clients.

        Returns the number of clients that received the message.
        """
        async with self._lock:
            sessions = list(self._connections.items())

        payload = self._build_payload(message, msg_type)
        sent = 0
        for session_id, ws in sessions:
            try:
                await ws.send_json(payload)
                sent += 1
                self._message_count += 1
            except Exception as exc:
                logger.warning("Broadcast failed for session %s: %s", session_id, exc)
                await self.disconnect(ws)

        return sent

    # ------------------------------------------------------------------
    # Typed messages
    # ------------------------------------------------------------------
    async def send_progress(
        self,
        session_id: str,
        pipeline_id: str,
        progress: float,
        message: str,
    ) -> bool:
        """Send a progress update for a pipeline run.

        Parameters
        ----------
        session_id:
            Target WebSocket session.
        pipeline_id:
            Identifier for the running pipeline.
        progress:
            Progress percentage (0–100).
        message:
            Human-readable progress message.
        """
        payload = {
            "pipeline_id": pipeline_id,
            "progress": min(max(progress, 0.0), 100.0),
            "message": message,
        }
        return await self.send_message(session_id, payload, msg_type="progress")

    async def send_result(
        self,
        session_id: str,
        pipeline_id: str,
        result: Any,
    ) -> bool:
        """Send a pipeline result to a session.

        Parameters
        ----------
        session_id:
            Target WebSocket session.
        pipeline_id:
            Identifier for the completed pipeline.
        result:
            Result data (will be JSON-serialised).
        """
        payload = {
            "pipeline_id": pipeline_id,
            "result": result,
            "status": "completed",
        }
        return await self.send_message(session_id, payload, msg_type="result")

    async def send_error(
        self,
        session_id: str,
        pipeline_id: str,
        error: str,
    ) -> bool:
        """Send an error message to a session."""
        payload = {
            "pipeline_id": pipeline_id,
            "error": error,
            "status": "error",
        }
        return await self.send_message(session_id, payload, msg_type="error")

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------
    def get_active_connections(self) -> int:
        """Return the number of currently active WebSocket connections."""
        return len(self._connections)

    def get_session_ids(self) -> List[str]:
        """Return all active session IDs."""
        return list(self._connections.keys())

    def is_connected(self, session_id: str) -> bool:
        """Check if a session is currently connected."""
        return session_id in self._connections

    def get_stats(self) -> Dict[str, Any]:
        """Return connection statistics."""
        return {
            "active_connections": len(self._connections),
            "total_connections": self._connect_count,
            "total_messages_sent": self._message_count,
            "session_ids": list(self._connections.keys()),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _build_payload(message: Any, msg_type: str) -> Dict[str, Any]:
        """Build a standardised WebSocket message payload."""
        payload: Dict[str, Any] = {
            "type": msg_type,
            "data": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return payload


# ---------------------------------------------------------------------------
# Module-level singleton for easy import
# ---------------------------------------------------------------------------
_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Return the global WebSocketManager singleton."""
    global _manager
    if _manager is None:
        _manager = WebSocketManager()
    return _manager
