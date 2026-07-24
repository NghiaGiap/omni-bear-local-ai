import asyncio
import sys

from omnibear_config import OmniBearConfigProvider, format_omnibear_context


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


async def main() -> None:
    provider = OmniBearConfigProvider()
    snapshot = await provider.get_config()
    print(f"source={snapshot.source}")
    print(format_omnibear_context(snapshot))


if __name__ == "__main__":
    asyncio.run(main())
