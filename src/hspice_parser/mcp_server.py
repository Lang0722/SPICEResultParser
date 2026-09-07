"""MCP server exposing hspice_parser to agents over stdio.

Install the extra first: pip install 'hspice_parser[mcp]'. Then register the
command `hsp-mcp` with your MCP client.
"""
import struct
from typing import List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        # mcp 2.x renamed FastMCP to MCPServer.
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError as exc:  # pragma: no cover
        raise ImportError("MCP support needs the optional dependency: pip install 'hspice_parser[mcp]'") from exc

try:
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:
    try:
        from mcp.server.fastmcp.exceptions import ToolError
    except ImportError:  # pragma: no cover
        ToolError = ValueError

from . import api

server = FastMCP("hspice-parser")

_REPORTED = (ValueError, OSError, ImportError)   # errors whose text the client should see


def _tool_errors(fn):
    """The mcp server forwards only ToolError text to the client; anything else becomes a bare
    'Error executing tool'. Re-raise the errors we mean the agent to read."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except _REPORTED as exc:
            raise ToolError(str(exc)) from exc
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__annotations__ = fn.__annotations__
    wrapper.__wrapped__ = fn
    return wrapper


@server.tool()
@_tool_errors
def list_traces(path: str) -> dict:
    """List trace names in an HSPICE .tr*/.sw*/.ac* result file (binary 9601/2001 or ASCII).

    Reads only the header, so it is cheap on any file size. Returns the x variable
    name, the trace names to pass to `extract`, and the sweep parameter names.
    """
    return api.list_traces(path)


@server.tool()
@_tool_errors
def extract(path: str, names: Optional[List[str]] = None, sweeps: Optional[List[int]] = None,
            output: str = "summary", downsample: Optional[int] = None, dest: Optional[str] = None,
            xrange: Optional[List[float]] = None, yrange: Optional[List[float]] = None) -> dict:
    """Extract selected traces from an HSPICE result file with memory bounded by the selection.

    names: trace names from list_traces (v_out), raw HSPICE names (v(out)), or globs (v_*);
           None selects everything. The x variable is always included.
    sweeps: sweep indices to keep; None keeps all.
    output: "summary" returns per-trace count/min/max/mean/first/last plus `downsample`
            evenly spaced x/y points; "csv" or "npz" write a file and return its path;
            "png" writes a plot image (one line per trace per sweep) and returns its path
            and pixel size. "arrays" is not available over MCP.
    downsample: evenly spaced points kept per sweep (summary points, or csv/npz/png rows);
            None keeps every point, except for "summary", which defaults to 200.
    dest: output file path for csv/npz/png; default is beside the input file.
    xrange: [min, max] on the x variable (TIME, FREQ, sweep variable); only rows inside
            the closed interval are kept, for every output.
    yrange: [min, max] vertical axis limits for the png plot; ignored by other outputs.
    """
    if output == "arrays":
        raise ValueError("output='arrays' is not available over MCP; use 'summary', 'csv', 'npz' or 'png'")
    if output not in ("summary", "csv", "npz", "png"):
        raise ValueError(f"output must be one of ('summary', 'csv', 'npz', 'png'), not {output!r}")
    xrange = api.validate_range("xrange", xrange)
    yrange = api.validate_range("yrange", yrange)
    if output == "summary":
        return api.extract(path, names, sweeps, "summary", 200 if downsample is None else downsample,
                           xrange=xrange)
    ts = api.extract(path, names, sweeps, "arrays", downsample, xrange=xrange)
    written = api.write_file(ts, output, dest, yrange)
    x = ts.header.x_name
    result = {
        "output": output, "path": written, "traces": ts.selected, "sweeps": len(ts.sweep_values),
        "sweep_indices": list(ts.sweep_indices),
        "points": [int(ts.data[x][i].size) for i in range(len(ts.sweep_values))],
        "truncated": ts.truncated,
    }
    if output == "png":
        result["size"] = list(_png_size(written))
    return result


def _png_size(path: str):
    """(width, height) from the PNG IHDR chunk."""
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def main() -> None:
    server.run()   # stdio transport
