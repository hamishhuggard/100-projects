import asyncio
import websockets

async def client():
    with websockets.connect('ws://localhost:8765') as ws:
        message = input('>')
        await ws.send(message)
        response = ws.recv()
        print('[]:', response)

asyncio.run(client())
