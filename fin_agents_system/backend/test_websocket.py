#!/usr/bin/env python3
"""
Test script to verify real-time WebSocket updates during workflow execution.
"""
import asyncio
import json
import websockets
import requests

async def test_websocket_updates():
    """Test that WebSocket receives real-time agent updates."""
    
    # Connect to WebSocket
    async with websockets.connect("ws://localhost:8000/ws") as websocket:
        print("Connected to WebSocket server")
        
        # Send workflow request
        response = requests.post(
            "http://localhost:8000/run-workflow",
            data={
                "url": "https://finance.yahoo.com/quote/000300.SS",
                "time_range": "7d",
                "interval": "1d"
            }
        )
        print(f"Workflow request sent, response: {response.json()}")
        
        # Track received updates
        received_updates = []
        
        # Receive WebSocket messages for 30 seconds
        try:
            for _ in range(30):
                message = await websocket.recv()
                data = json.loads(message)
                received_updates.append(data)
                print(f"Received: {json.dumps(data, ensure_ascii=False)}")
                await asyncio.sleep(1)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except asyncio.TimeoutError:
            print("Timeout waiting for messages")
        
        # Verify all agent updates were received
        print("\n--- Summary of Received Updates ---")
        agent_updates = [u for u in received_updates if u.get("type") == "agent_update"]
        print(f"Total agent updates received: {len(agent_updates)}")
        
        for update in agent_updates:
            print(f"- {update['agent_name']}: {update['status']} - {update['message']}")
        
        # Check if we received updates from all agents
        expected_agents = ["数据工程师", "情感分析师", "技术分析师", "回测专家"]
        received_agent_names = [u["agent_name"] for u in agent_updates]
        
        print("\n--- Verification ---")
        for agent in expected_agents:
            if agent in received_agent_names:
                print(f"✓ Received updates from {agent}")
            else:
                print(f"✗ No updates received from {agent}")
        
        return agent_updates

if __name__ == "__main__":
    asyncio.run(test_websocket_updates())
