"""
JARVIS — Self-Modifying Cognitive Router
Entry point.

Usage:
    python main.py                    # Start CLI
    python main.py --server           # Start API server
    python main.py --verbose          # CLI with detailed output
    python main.py --checkpoint DIR   # Custom checkpoint directory
"""

import argparse
import asyncio
import logging
import sys

from jarvis.core.router import RouterConfig
from jarvis.interface.cli import JarvisCLI
from jarvis.interface.server import JarvisServer


def parse_args():
    parser = argparse.ArgumentParser(description="JARVIS — Self-Modifying Cognitive Router")
    parser.add_argument("--server", action="store_true", help="Run as API server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8420, help="Server port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--checkpoint", default="./jarvis_state", help="Checkpoint directory")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--watty-url", default=None, help="Watty MCP server URL")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Router hidden dimension")
    parser.add_argument("--input-dim", type=int, default=384, help="Input embedding dimension")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = RouterConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
    )

    cli = JarvisCLI(
        config=config,
        api_key=args.api_key,
        watty_url=args.watty_url,
        checkpoint_dir=args.checkpoint,
        verbose=args.verbose,
    )

    # Try loading previous state
    try:
        cli.router.load_state(args.checkpoint)
    except Exception:
        pass  # Fresh start

    if args.server:
        server = JarvisServer(cli, host=args.host, port=args.port)
        server.run()
    else:
        asyncio.run(cli.run())


if __name__ == "__main__":
    main()
