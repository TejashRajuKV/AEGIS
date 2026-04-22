"""
WebSocket API Routes — real-time streaming for AEGIS pipeline updates.

Endpoint
--------
/ws/{session_id}  – WebSocket connection for real-time message streaming.

The WebSocket supports:
* **Connect**: Client opens a connection with a session ID.
* **Receive**: Server broadcasts received messages to all connected sessions.
* **Disconnect**: Client closes the connection.
* **Typed messages**: Progress updates, results, and errors.
"""

import asyncio
import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Shared singleton — Bug 14/40 fix: import from websocket_manager so there is
# ONE manager shared by all parts of the system (routes, dependencies, etc.).
# ---------------------------------------------------------------------------
try:
    from app.services.websocket_manager import WebSocketManager, get_websocket_manager
    _HAS_WS_MANAGER = True
except ImportError as exc:
    WebSocketManager = None  # type: ignore[assignment, misc]
    get_websocket_manager = None  # type: ignore[assignment]
    _HAS_WS_MANAGER = False
    logger.warning("WebSocketManager import failed: %s", exc)


# Use the shared singleton from websocket_manager (not a new instance)
ws_manager: "WebSocketManager" = get_websocket_manager() if _HAS_WS_MANAGER else None  # type: ignore[assignment]


def _get_ws_manager() -> WebSocketManager:
    """Return the module-level WebSocketManager singleton.

    Raises
    ------
    RuntimeError
        If WebSocketManager is not available.
    """
    if ws_manager is None:
        raise RuntimeError(
            "WebSocketManager is not available. Ensure the websocket_manager "
            "module is properly installed."
        )
    return ws_manager


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
) -> None:
    """Handle a WebSocket connection for real-time streaming.

    Lifecycle:
    1. **Accept** the connection and register it under ``session_id``.
    2. **Receive loop**: read messages from the client and broadcast them
       to all other connected sessions.
    3. **Disconnect**: unregister the connection when the client closes it.

    Parameters
    ----------
    websocket : fastapi.WebSocket
        The incoming WebSocket connection.
    session_id : str
        Client-provided session identifier for connection tracking.
    """
    manager = _get_ws_manager()

    # --- Connect ---
    try:
        await manager.connect(websocket, session_id)
    except Exception as exc:
        logger.error(
            "WebSocket accept failed for session '%s': %s",
            session_id, exc,
        )
        await websocket.close(code=1011, reason="Server error during connect")
        return

    # Send welcome message
    await manager.send_message(
        session_id=session_id,
        message={
            "event": "connected",
            "session_id": session_id,
            "active_connections": manager.get_active_connections(),
        },
        msg_type="system",
    )

    logger.info(
        "WebSocket session '%s' connected. Active: %d",
        session_id,
        manager.get_active_connections(),
    )

    # --- Receive loop ---
    try:
        while True:
            # Receive raw data from client
            raw_data = await websocket.receive_text()

            # Attempt to parse as JSON
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                data = {"raw": raw_data, "parsed": False}

            logger.debug(
                "WebSocket received from '%s': %s",
                session_id,
                str(data)[:200],
            )

            # Broadcast the message to all other connected sessions
            broadcast_count = await manager.broadcast(
                message={
                    "event": "message",
                    "from_session": session_id,
                    "data": data,
                },
                msg_type="broadcast",
            )

            # Acknowledge receipt to the sender
            await manager.send_message(
                session_id=session_id,
                message={
                    "event": "ack",
                    "status": "received",
                    "broadcast_to": broadcast_count,
                },
                msg_type="system",
            )

    except WebSocketDisconnect:
        logger.info(
            "WebSocket session '%s' disconnected by client.",
            session_id,
        )
    except asyncio.CancelledError:
        logger.info(
            "WebSocket session '%s' cancelled.",
            session_id,
        )
    except Exception as exc:
        logger.error(
            "WebSocket error for session '%s': %s",
            session_id, exc,
        )
        # Try to send an error message before disconnecting
        try:
            await manager.send_error(
                session_id=session_id,
                pipeline_id="",
                error=f"Connection error: {str(exc)}",
            )
        except Exception:
            pass

    # --- Disconnect ---
    await manager.disconnect(websocket)
    logger.info(
        "WebSocket session '%s' cleaned up. Remaining: %d",
        session_id,
        manager.get_active_connections(),
    )


# ---------------------------------------------------------------------------
# HTTP helper endpoints (for diagnostics)
# ---------------------------------------------------------------------------

@router.get("/ws-stats")
async def get_websocket_stats() -> Dict[str, Any]:
    """Return WebSocket connection statistics.

    This is an HTTP endpoint (not WebSocket) for monitoring purposes.

    Returns
    -------
    dict
        Connection statistics including active count and session IDs.
    """
    manager = _get_ws_manager()
    return manager.get_stats()
