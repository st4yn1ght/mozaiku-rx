"""Pemuatan checkpoint Ultralytics (YOLO) secara aman.

File ``.pt`` Ultralytics berisi objek Python ter-pickle (kelas layer). Memuat
pickle sembarangan bisa menjalankan kode. Modul ini:

1. Memindai opcode pickle secara statis (tanpa mengeksekusi apa pun) dan
   menolak file yang mereferensikan fungsi/kelas di luar daftar izin.
2. Memuat dengan ``torch.load(weights_only=True)`` + daftar izin (PyTorch >= 2.5).
3. Bila PyTorch lama (mis. 2.4 untuk DirectML) tidak mendukungnya, memakai
   unpickler terbatas klasik (``find_class`` hanya mengizinkan kelas yang sama).
"""

from __future__ import annotations

import importlib
import pickle
import pickletools
import types
import zipfile
from pathlib import Path

ALLOWED_PREFIXES = ("ultralytics.nn.", "torch.nn.modules.")

# Global selain kelas layer yang wajar ada di checkpoint PyTorch.
ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"),
    ("builtins", "set"),
    ("__builtin__", "set"),
    ("builtins", "frozenset"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_parameter_with_state"),
    ("torch", "Size"),
    ("torch", "device"),
}

_STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
               "SHORT_BINSTRING", "BINSTRING", "STRING"}
_GET_OPS = {"BINGET", "LONG_BINGET", "GET"}
_PUT_OPS = {"BINPUT", "LONG_BINPUT", "PUT"}
_NO_PUSH_OPS = {"PROTO", "FRAME", "MEMOIZE", "STOP", "POP", "POP_MARK"} | _PUT_OPS
_FORBIDDEN_OPS = {"EXT1", "EXT2", "EXT4", "INST", "OBJ"}


class UnsafeCheckpointError(RuntimeError):
    pass


def _is_allowed(module: str, name: str) -> bool:
    if (module, name) in ALLOWED_GLOBALS:
        return True
    if module == "torch" and name.endswith("Storage"):
        return True
    return module.startswith(ALLOWED_PREFIXES)


def scan_pickle_globals(path: str | Path) -> set[tuple[str, str]]:
    """Kembalikan semua (module, name) yang direferensikan pickle di dalam checkpoint zip."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
            if len(names) != 1:
                raise UnsafeCheckpointError("struktur checkpoint tidak dikenali")
            data = zf.read(names[0])
    except zipfile.BadZipFile as exc:
        raise UnsafeCheckpointError("format checkpoint lama/tidak dikenal (bukan zip)") from exc

    found: set[tuple[str, str]] = set()
    stack: list[object] = []      # hanya melacak string & penanda "objek lain"
    memo: dict[int, object] = {}
    for op, arg, _pos in pickletools.genops(data):
        name = op.name
        if name in _FORBIDDEN_OPS:
            raise UnsafeCheckpointError(f"opcode pickle tidak diizinkan: {name}")
        if name in _STRING_OPS:
            stack.append(arg if isinstance(arg, str) else str(arg))
        elif name in _GET_OPS:
            stack.append(memo.get(int(arg)))
        elif name in _PUT_OPS:
            memo[int(arg)] = stack[-1] if stack else None
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1] if stack else None
        elif name == "STACK_GLOBAL":
            if len(stack) < 2 or not isinstance(stack[-1], str) or not isinstance(stack[-2], str):
                raise UnsafeCheckpointError("STACK_GLOBAL dengan argumen tidak dikenal")
            module, attr = stack[-2], stack[-1]
            del stack[-2:]
            found.add((module, attr))
            stack.append(None)
        elif name == "GLOBAL":
            module, _, attr = str(arg).partition(" ")
            found.add((module, attr))
            stack.append(None)
        elif name in _NO_PUSH_OPS:
            pass
        else:
            stack.append(None)
        if len(stack) > 64:
            del stack[:-32]
    bad = sorted(f"{m}.{n}" for m, n in found if not _is_allowed(m, n))
    if bad:
        raise UnsafeCheckpointError("checkpoint mereferensikan objek yang tidak diizinkan: " + ", ".join(bad))
    return found


def _resolve(module: str, name: str):
    if module == "__builtin__":
        module = "builtins"
    obj = getattr(importlib.import_module(module), name)
    return obj


def _allowed_classes(found: set[tuple[str, str]]) -> list:
    classes = []
    for module, name in found:
        if module.startswith(ALLOWED_PREFIXES):
            obj = _resolve(module, name)
            if not isinstance(obj, type):
                raise UnsafeCheckpointError(f"{module}.{name} bukan kelas")
            classes.append(obj)
        elif (module, name) in (("builtins", "set"), ("__builtin__", "set")):
            classes.append(set)
        elif (module, name) == ("builtins", "frozenset"):
            classes.append(frozenset)
    return classes


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):  # noqa: D401
        if module == "__builtin__":
            module = "builtins"
        if not _is_allowed(module, name):
            raise pickle.UnpicklingError(f"global tidak diizinkan: {module}.{name}")
        obj = super().find_class(module, name)
        if module.startswith(ALLOWED_PREFIXES) and not isinstance(obj, type):
            raise pickle.UnpicklingError(f"{module}.{name} bukan kelas")
        return obj


_restricted_pickle = types.ModuleType("restricted_pickle")
_restricted_pickle.Unpickler = _RestrictedUnpickler  # type: ignore[attr-defined]
_restricted_pickle.load = lambda f, **kw: _RestrictedUnpickler(f, **kw).load()  # type: ignore[attr-defined]
_restricted_pickle.__dict__["UnpicklingError"] = pickle.UnpicklingError


def load_ultralytics_checkpoint(path: str | Path) -> dict:
    """Muat checkpoint YOLO dengan aman dan kembalikan dict checkpoint."""
    import torch

    found = scan_pickle_globals(path)
    try:
        importlib.import_module("ultralytics.nn.tasks")
    except ImportError as exc:  # pragma: no cover - dependensi wajib
        raise RuntimeError("Paket 'ultralytics' belum terpasang (pip install ultralytics)") from exc
    classes = _allowed_classes(found)

    add_safe = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe is not None:
        try:
            add_safe(classes)
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            pass  # PyTorch lama: lanjut ke unpickler terbatas di bawah
        except RuntimeError as exc:
            if "weights_only" not in str(exc).lower() and "unsupported" not in str(exc).lower():
                raise
    # Unpickler terbatas: hanya kelas/fungsi yang sama dengan daftar izin di atas
    # yang bisa dibuat. File sudah lolos pemindaian statis.
    return torch.load(str(path), map_location="cpu", weights_only=False,
                      pickle_module=_restricted_pickle)
