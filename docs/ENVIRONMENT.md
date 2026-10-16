# Validated local environment

`requirements.txt` remains the flexible installation entry point. For the
direct-dependency combination exercised during the final local validation, use
the pinned [`requirements-verified.txt`](../requirements-verified.txt):

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements-verified.txt
.venv\Scripts\python -m pip check
.venv\Scripts\python -m unittest discover -s tests -v
```

The verified file intentionally pins only direct dependencies. A formal
submission should additionally archive the resolved transitive environment
(for example, `pip freeze` in the experiment artifact or a platform-specific
lockfile), operating-system details, model SHA-256, the source revision and
the `source_tree_dirty` value written to `run-manifest.json`.
