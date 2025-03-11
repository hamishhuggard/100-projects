import asyncio
import websockets

async def handler(websocket, path):
    async for message in websocket:
        print('received', message)
        await websocket.send('Echo', message)

async def main():
    server = await websockets.serve(handler, "localhost", 8765)
    print('websocket server started on ws://localhost:8765')
    awiat server.wait_closed()

async.run(main())
