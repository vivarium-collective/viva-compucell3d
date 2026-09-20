"""build_core() — allocate_core() + this workspace's own Edges.

``allocate_core()`` auto-discovers Processes/Steps from *installed* pbg-*
wheels via bigraph-schema's dist-walker, but a workspace checkout is typically
an editable install the dist-walker can't see (and cc3d pins Python 3.10, so
this package often isn't installed as a wheel at all — it is imported from the
source tree). The dashboard runs each composite in a subprocess that calls this
``build_core()``, so we explicitly register the workspace's own Process/Step
classes here — otherwise composites addressing ``local:CompuCell3DProcess``
fail with "no link found at address". Registration is generic (walks the
package), so new processes are picked up automatically.
"""
from __future__ import annotations

import importlib
import pkgutil

from process_bigraph import Process, Step, allocate_core

import pbg_compucell3d


def build_core():
    core = allocate_core()
    for cls, dotted in _iter_workspace_edges(pbg_compucell3d):
        # Newer process-bigraph (the workbench / CI env) uses register_link +
        # link_registry; the older 0.0.42 pinned by the cc3d (pixi) env uses
        # register_process. Support both so build_core() works in either.
        if hasattr(core, "register_link"):
            core.register_link(dotted, cls)
            core.register_link(cls.__name__, cls)
        else:
            core.register_process(cls.__name__, cls)
    return core


def _iter_workspace_edges(package):
    """Yield (class, dotted_path) for each Step/Process defined in ``package``."""
    pkg_name = package.__name__
    seen = set()
    for _, modname, _ in pkgutil.walk_packages(package.__path__,
                                               prefix=f"{pkg_name}."):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for attr in dir(mod):
            obj = getattr(mod, attr, None)
            if not isinstance(obj, type) or obj in (Process, Step):
                continue
            if not issubclass(obj, (Process, Step)):
                continue
            if not getattr(obj, "__module__", "").startswith(pkg_name):
                continue
            if obj in seen:
                continue
            seen.add(obj)
            yield obj, obj.__module__ + "." + obj.__name__
