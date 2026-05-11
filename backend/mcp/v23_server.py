"""woodblock_stack v23-MCP server entrypoint — D1 stub.

The real FastMCP wiring + ``asyncio.Semaphore(1)`` GPU lock land in D19
per ``/tmp/research-v23-mcp-build-sequence.md``. D1 only needs:

- ``app`` object with ``name == "woodblock_stack"`` (for shape tests)
- ``main()`` callable (for the ``woodblock-mcp`` console script)

Avoid importing FastMCP here until D19; D1 scaffold runs on stock
Python so the CI lint + shape rings stay GPU-free.
"""
from __future__ import annotations


class _MockApp:
    """Stand-in for ``FastMCP("woodblock_stack")`` until D19."""

    name = "woodblock_stack"

    def __repr__(self) -> str:
        return f"<v23_server.app name={self.name!r} (stub)>"


app = _MockApp()


def main() -> None:
    """Console-script entry. Real wiring at D19.

    Prints a one-line banner so ``which woodblock-mcp && woodblock-mcp``
    in CI produces visible evidence the package installed.
    """
    print("woodblock_stack v23-MCP — D1 stub. Real server wires in D19.")


if __name__ == "__main__":
    main()
