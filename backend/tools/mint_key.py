"""Admin CLI to mint, list and revoke TraceSort API keys."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_frozen_db() -> None:
    """Point SWR_DB_PATH at the real DB when running as a PyInstaller exe."""
    if not getattr(sys, "frozen", False):
        return
    if os.getenv("SWR_DB_PATH"):
        return
    exe_dir = Path(sys.executable).resolve().parent
    for candidate in (exe_dir / "backend" / "waste.db", exe_dir / "waste.db"):
        if candidate.exists():
            os.environ["SWR_DB_PATH"] = str(candidate)
            return
    print("waste.db پیدا نشد.")
    print(f"exe را کنار پوشهٔ backend بگذار (اینجا: {exe_dir})")
    print("یا SWR_DB_PATH را به مسیر waste.db ست کن.")
    input("Enter برای بستن...")
    raise SystemExit(1)


_resolve_frozen_db()

import argparse
import subprocess
import sys
from pathlib import Path

# Now it's safe to import backend modules – the env var is already set.
from backend import keys                # noqa: E402
from backend.config import DB_PATH       # noqa: E402


def _clip(text: str) -> bool:
    """Copy text to the Windows clipboard. False if clip.exe is unavailable."""
    try:
        subprocess.run(["clip"], input=text, text=True, check=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _announce_db() -> bool:
    """Print the resolved DB path. False if the DB does not exist yet."""
    path = Path(DB_PATH).resolve()
    print(f"[db] {path}")
    if path.exists():
        return True
    print("[warn] No database found at that path.")
    print("[warn] A new empty one will be created and keys minted here")
    print("[warn] will NOT be accepted by the running server.")
    print("[warn] Run this from the project root, next to the 'backend' folder.")
    return False


def issue(label: str, hours: int) -> None:
    """Mint one key, print it once, copy it to the clipboard."""
    raw, expires_at = keys.issue_key(label, hours)
    print(f"label   : {label}")
    print(f"expires : {expires_at}")
    print(f"key     : {raw}")
    if _clip(raw):
        print("\n[ok] Copied to clipboard. Paste it into the admin panel.")
    else:
        print("\nCopy it now. The raw key is never stored and cannot be recovered.")


def show(active_only: bool) -> None:
    """Print key metadata (never the raw key)."""
    rows = keys.list_keys(active_only=active_only)
    if not rows:
        print("No keys found.")
        return
    header = (
        f"{'id':>4}  {'label':<16} {'preview':<14} "
        f"{'expires_at':<32} {'rev':>3}  last_used_at"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['id']:>4}  {r['label']:<16} {r['preview']:<14} "
            f"{r['expires_at']:<32} {r['revoked']:>3}  {r['last_used_at'] or '-'}"
        )


def revoke(key_id: int) -> int:
    """Revoke one key by id. Returns the process exit code."""
    if keys.revoke_key(key_id):
        print(f"Key {key_id} revoked.")
        return 0
    print(f"No key with id {key_id}.", file=sys.stderr)
    return 1


def _ask_hours(default: int = 8) -> int:
    """Read a positive whole number of hours, re-prompting on bad input."""
    while True:
        raw = input(f"Valid for how many hours [{default}]: ").strip()
        if not raw:
            return default
        try:
            hours = int(raw)
        except ValueError:
            print("  Enter a whole number, e.g. 10")
            continue
        if hours <= 0:
            print("  Must be greater than 0.")
            continue
        return hours


def interactive() -> int:
    """Fallback flow when the exe is double-clicked with no arguments."""
    print(f"Database: {DB_PATH}\n")
    label = input("Label (e.g. shift-a): ").strip() or "shift"
    hours = _ask_hours()
    _announce_db()
    issue(label, hours)
    input("\nPress Enter to close...")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Define the issue/list/revoke subcommands."""
    parser = argparse.ArgumentParser(
        prog="tracesort-keygen", description="Manage TraceSort API keys."
    )
    sub = parser.add_subparsers(dest="command")

    p_issue = sub.add_parser("issue", help="mint a new key")
    p_issue.add_argument("label", help="who the key is for, e.g. shift-a")
    p_issue.add_argument(
        "--hours", type=int, default=8, help="validity in hours (default: 8)"
    )

    p_list = sub.add_parser("list", help="list keys")
    p_list.add_argument(
        "--active", action="store_true", help="only non-revoked, non-expired keys"
    )

    p_revoke = sub.add_parser("revoke", help="revoke a key by id")
    p_revoke.add_argument("key_id", type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Ensures the schema exists before any operation."""
    # Print DB path for debugging / transparency
    _announce_db()

    args = build_parser().parse_args(argv)
    keys.init_schema()

    try:
        if args.command == "issue":
            if args.hours <= 0:
                print("--hours must be greater than 0.", file=sys.stderr)
                return 2
            issue(args.label, args.hours)
            return 0
        if args.command == "list":
            show(args.active)
            return 0
        if args.command == "revoke":
            return revoke(args.key_id)
        return interactive()
    except Exception as exc:           # noqa: BLE001 – keep the window open on error
        print(f"[error] {exc}")
        input("Press Enter to close...")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())