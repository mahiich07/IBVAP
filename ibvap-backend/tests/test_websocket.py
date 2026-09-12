import unittest
import asyncio
import json
from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from services.websocket_gateway import ws_gateway

class MockWebSocket:
    def __init__(self):
        self.accepted = False
        self.sent_messages = []
        self.is_closed = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, text: str):
        if self.is_closed:
            raise RuntimeError("Socket is closed")
        self.sent_messages.append(text)

class TestWebSocketGateway(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_connect_and_broadcast(self):
        mock_ws = MockWebSocket()
        await ws_gateway.connect(mock_ws)
        self.assertTrue(mock_ws.accepted)
        self.assertIn(mock_ws, ws_gateway.active_connections)

        # Broadcast telemetry
        test_payload = {"test": "alert_data", "severity": "CRITICAL"}
        await ws_gateway.broadcast("alert_telemetry", test_payload)

        self.assertEqual(len(mock_ws.sent_messages), 1)
        data = json.loads(mock_ws.sent_messages[0])
        self.assertEqual(data["event_type"], "alert_telemetry")
        self.assertEqual(data["data"]["severity"], "CRITICAL")

        # Disconnect
        await ws_gateway.disconnect(mock_ws)
        self.assertNotIn(mock_ws, ws_gateway.active_connections)

if __name__ == "__main__":
    unittest.main()
