# -*- coding: utf-8 -*-
"""Desktop client for minting shift API keys via the admin endpoints."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
import os
import requests

TIMEOUT = 10

# Create a global session to prevent localhost traffic from going through system proxies.
HTTP = requests.Session()
HTTP.trust_env = False

class KeyMaker(tk.Tk):
    """Minimal Tk GUI wrapping POST/GET/DELETE /admin/keys."""

    def __init__(self) -> None:
        super().__init__()
        self.title("SWR Key Maker")
        self.geometry("620x440")
        self.resizable(False, False)

        # Admin key variable
        self.admin = tk.StringVar()

        self._build()
        # ✅ Ensured that the admin key is loaded from .env after building the UI
        self._load_admin_key()  # Auto-load from .env

    def _load_admin_key(self) -> None:
        """Try to find .env file and extract API_KEY."""
        # Find the exact path of the .env file relative to this keymaker.py file.
        # Assumes .env is in the project root folder (same directory as keymaker.py).
        base_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(base_dir, ".env")

        if os.path.exists(env_path):
            try:
                with open(env_path, "r") as f:
                    for line in f:
                        if line.startswith("API_KEY="):
                            # Remove API_KEY= and any surrounding quotes/spaces
                            key = line.strip().split("=", 1)[1].strip("'\" ")
                            self.admin.set(key)
                            return
            except Exception as e:
                print(f"Error reading .env: {e}")

    def _build(self) -> None:
        """Lay out the form, action buttons and result areas."""
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)

        self.base = tk.StringVar(value="http://127.0.0.1:8001")
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

        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=4, column=0, columnspan=2, pady=10, sticky="w")
        ttk.Button(btn_frm, text="Mint Key", command=self._on_mint).pack(side="left", padx=5)
        ttk.Button(btn_frm, text="Refresh List", command=self._on_refresh).pack(side="left", padx=5)
        ttk.Button(btn_frm, text="Revoke Selected", command=self._on_revoke).pack(side="left", padx=5)
        ttk.Button(btn_frm, text="Copy Key", command=self._on_copy).pack(side="left", padx=5)

        self.tree = ttk.Treeview(self, columns=("id", "preview", "label", "expires_at", "revoked"), show="headings", height=12)
        self.tree.heading("id", text="ID"); self.tree.heading("preview", text="Preview")
        self.tree.heading("label", text="Label"); self.tree.heading("expires_at", text="Expires At")
        self.tree.heading("revoked", text="Revoked")

        self.tree.column("id", width=40, anchor="center"); self.tree.column("preview", width=180, anchor="w")
        self.tree.column("label", width=120, anchor="w"); self.tree.column("expires_at", width=180, anchor="center")
        self.tree.column("revoked", width=70, anchor="center")

        sb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        sb.pack(side="right", fill="y", padx=(0, 8), pady=8)

        self.last_minted_key = ""

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.admin.get().strip()}

    def _url(self, path: str = "") -> str:
        return f"{self.base.get().rstrip('/')}/admin/keys{path}"

    def _on_mint(self) -> None:
        def task():
            try:
                try: hours = int(self.hours.get().strip())
                except ValueError:
                    messagebox.showerror("Error", "Hours must be an integer."); return

                r = HTTP.post(self._url(), json={"label": self.label.get().strip(), "hours": hours}, headers=self._headers(), timeout=TIMEOUT)

                if r.status_code in (200, 201):
                    data = r.json()
                    # Fix: Extract api_key correctly
                    self.last_minted_key = data.get("api_key") or data.get("key", "")
                    messagebox.showinfo("Success", f"Key minted successfully!\n\nUse 'Copy Key' button to copy.")
                    self._on_refresh()
                else:
                    messagebox.showerror("Server Error", f"Status: {r.status_code}\n{r.text}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed: {str(e)}")

        threading.Thread(target=task, daemon=True).start()

    def _on_refresh(self) -> None:
        def task():
            try:
                r = HTTP.get(self._url(), headers=self._headers(), timeout=TIMEOUT)
                if r.status_code == 200:
                    for item in self.tree.get_children(): self.tree.delete(item)
                    for k in r.json():
                        self.tree.insert("", "end", values=(k.get("id"), k.get("preview"), k.get("label"), k.get("expires_at") or "Never", "Yes" if k.get("revoked") else "No"))
                else:
                    messagebox.showerror("Server Error", f"Status: {r.status_code}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        threading.Thread(target=task, daemon=True).start()

    def _on_revoke(self) -> None:
        selected = self.tree.selection()
        if not selected: return
        key_id = self.tree.item(selected[0])["values"][0]
        def task():
            r = HTTP.delete(self._url(f"/{key_id}"), headers=self._headers(), timeout=TIMEOUT)
            if r.status_code in (200, 204):
                messagebox.showinfo("Success", "Revoked.")
                self._on_refresh()
        threading.Thread(target=task, daemon=True).start()

    def _on_copy(self) -> None:
        if self.last_minted_key:
            self.clipboard_clear()
            self.clipboard_append(self.last_minted_key)
            messagebox.showinfo("Copied", "Full key copied to clipboard.")
        else:
            messagebox.showwarning("Warning", "No newly minted key available.")

if __name__ == "__main__":
    KeyMaker().mainloop()