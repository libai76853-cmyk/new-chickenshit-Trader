Vendored from https://github.com/shiyu-coder/Kronos (MIT License, see LICENSE) — the `model/` package only
(`__init__.py`, `kronos.py`, `module.py`). Upstream commit in UPSTREAM_COMMIT.txt. Weights are downloaded from
Hugging Face (NeoQuasar/Kronos-*) at runtime into data/models/hf.

Local modification: `module.py` RotaryPositionalEmbedding rebuilds its cos/sin cache when the input tensor's
device differs from the cached tensors (upstream keeps the cache as plain attributes, so switching devices in
one process raised "Expected all tensors to be on the same device").

Local modification 2: `kronos.py` imports `.module` relatively instead of `model.module`.
