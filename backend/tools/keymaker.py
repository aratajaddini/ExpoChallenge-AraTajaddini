"""Desktop client for minting shift API keys via the admin endpoints."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

import requests

TIMEOUT = 10


class KeyMaker(tk.Tk):
    """Minimal Tk GUI wrapping POST/GET/DELETE /admin/keys."""

    def __init__(self) -> None:
        super().__init__()
        self.title("SWR Key Maker")
        self.geometry("620x440")
        self.resizable(False, False)
        self._build()

    def _build(self) -> None:
        """Lay out the form, action buttons and result areas."""
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)

        self.base = tk.StringVar(value="http://127.0.0.1:8000")
        self.admin = tk.StringVar()
        self.label = tk.StringVar(value="shift-morning")
        self.hours = tk.StringVar(value="8")

        for row, (text, var, hide) in enumerate([
            ("Server URL", self.base, False),
            ("Admin key", self.admin, True),
            ("Label", self.label, False),
            ("Hours", self.hours, False),
        ]):
            ttk.Label(frm, text=text).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(
                frm, textvariable=var, width=58, show="*" if hide else ""
            ).grid(row=row, column=1, sticky="w", pady=3)

        btns = ttk.Frame(self)
        btns.pack(fill="x", **pad)
        ttk.Button(btns, text="Mint key", command=self._mint).pack(side="left")
        ttk.Button(btns, text="Refresh list", command=self._list).pack(side="left", padx=6)
        ttk.Button(btns, text="Revoke selected", command=self._revoke).pack(side="left")

        self.result = tk.Text(self, height=4, wrap="none")
        self.result.pack(fill="x", **pad)
        ttk.Button(self, text="Copy key", command=self._copy).pack(anchor="w", padx=8)

        cols = ("id", "preview", "label", "expires_at", "revoked")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=9)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=110 if col != "expires_at" else 190)
        self.tree.pack(fill="both", expand=True, **pad)

    def _headers(self) -> dict[str, str]:
        """Auth header built from the admin key field."""
        return {"X-API-Key": self.admin.get().strip()}

    def _url(self, path: str = "") -> str:
        """Absolute admin endpoint URL."""
        return f"{self.base.get().rstrip('/')}/admin/keys{path}"

    def _run(self, fn) -> None:
        """Run a network call off the UI thread."""
        threading.Thread(target=fn, daemon=True).start()

    def _mint(self) -> None:
        def job() -> None:
            try:
                hours = int(self.hours.get())
                r = requests.post(
                    self._url(),
                    json={"label": self.label.get().strip(), "hours": hours},
                    headers=self._headers(),
                    timeout=TIMEOUT,
                )
                r.raise_for_status()
                data = r.json()
                raw = data.get("api_key") or data.get("key", "")
                text = f"{raw}\nexpires_at: {data.get('expires_at', '?')}"
                self.after(0, lambda: self._show(text))
                self.after(0, self._list)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Mint failed", str(exc)))

        self._run(job)

    def _list(self) -> None:
        def job() -> None:
            try:
                r = requests.get(self._url(), headers=self._headers(), timeout=TIMEOUT)
                r.raise_for_status()
                rows = r.json()
                rows = rows.get("keys", rows) if isinstance(rows, dict) else rows
                self.after(0, lambda: self._fill(rows))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("List failed", str(exc)))

        self._run(job)

    def _revoke(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Revoke", "Select a row first.")
            return
        key_id = self.tree.item(sel[0], "values")[0]
        if not messagebox.askyesno("Revoke", f"Revoke key id {key_id}?"):
            return

        def job() -> None:
            try:
                r = requests.delete(
                    self._url(f"/{key_id}"), headers=self._headers(), timeout=TIMEOUT
                )
                r.raise_for_status()
                self.after(0, self._list)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Revoke failed", str(exc)))

        self._run(job)

    def _show(self, text: str) -> None:
        """Replace the result box content."""
        self.result.delete("1.0", "end")
        self.result.insert("1.0", text)

    def _fill(self, rows: list[dict]) -> None:
        """Repopulate the key table."""
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            self.tree.insert("", "end", values=(
                row.get("id"), row.get("preview"), row.get("label"),
                row.get("expires_at"), row.get("revoked"),
            ))

    def _copy(self) -> None:
        """Copy the first result line (the raw key) to the clipboard."""
        raw = self.result.get("1.0", "1.end").strip()
        if raw:
            self.clipboard_clear()
            self.clipboard_append(raw)


if __name__ == "__main__":
    KeyMaker().mainloop()
