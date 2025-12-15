from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import thunder
from thunder.dynamo.utils import CompilerType


class ScalarRecompileTracker:
    """Utility to record Thunder recompilations and persist offending GraphModules."""

    def __init__(self, log_dir: str | None, run_label: str):
        self.enabled = log_dir is not None
        self.log_dir = Path(log_dir) if log_dir else None
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_label = run_label
        self.prev_misses: dict[tuple[int, int], int] = {}

    def _summarize_inputs(self, input_summary: dict[str, Any]) -> dict[str, Any]:
        return input_summary

    def _save_graph_module(self, gm: torch.fx.GraphModule | None, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        event = event.copy()
        code_hash = None
        if gm is not None:
            code_hash = hashlib.sha256(gm.code.encode()).hexdigest()[:8]
            event["code_hash"] = code_hash

        base = (
            f"{self.run_label}_graph{event['graph_idx']}_sub{event['subgraph_idx']}_"
            f"miss{event['cache_misses']}"
        )
        if code_hash:
            base += f"_h{code_hash}"
        meta_path = self.log_dir / f"{base}.json"
        with open(meta_path, "w") as f:
            json.dump(event, f, indent=2)
        if gm is not None:
            gm_path = self.log_dir / f"{base}.pt"
            torch.save(gm, gm_path)

    def track_thunder_fx(
        self, compiled_obj, iter_idx: int, input_summary: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Track cache misses for thunderfx-compiled submodules."""
        if not self.enabled:
            return []
        backend = compiled_obj._backend
        events: list[dict[str, Any]] = []
        for g_idx, sinfo in enumerate(backend.subgraph_infos):
            for sub_idx, (gm, compiled_fn) in enumerate(sinfo.submodule_to_compiled_functions.items()):
                if compiled_fn.compiler != CompilerType.THUNDER:
                    continue
                fn = compiled_fn.compiled_fn
                miss = thunder.cache_misses(fn)
                key = (g_idx, sub_idx)
                prev = self.prev_misses.get(key, miss)
                self.prev_misses[key] = miss
                if miss <= 1 or miss == prev:
                    continue
                event = {
                    "iter_idx": iter_idx,
                    "graph_idx": g_idx,
                    "subgraph_idx": sub_idx,
                    "cache_misses": miss,
                    "input_summary": self._summarize_inputs(input_summary or {}),
                }
                events.append(event)
                self._save_graph_module(gm, event)
        return events

    def track_thunder_jit(self, jit_fn, iter_idx: int, input_summary: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Track cache misses for plain thunder.jit callables."""
        if not self.enabled:
            return []
        miss = thunder.cache_misses(jit_fn)
        prev = self.prev_misses.get(("jit", 0), miss)
        self.prev_misses[("jit", 0)] = miss
        if miss <= 1 or miss == prev:
            return []
        event = {
            "iter_idx": iter_idx,
            "graph_idx": 0,
            "subgraph_idx": 0,
            "cache_misses": miss,
            "input_summary": self._summarize_inputs(input_summary or {}),
        }
        events = [event]
        gm = None
        try:
            traces = thunder.last_traces(jit_fn)
            if traces:
                gm = traces[-1].graph_module
        except Exception:
            gm = None
        self._save_graph_module(gm, event)
        return events

