# Backend contract tests

The suite deliberately uses Python's standard-library `unittest`, so it can
run in the existing application virtual environment without adding a test
framework or downloading model weights.

From `D:\\AIC\\sport-vision` run:

```powershell
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
```

It checks the integration catalogue, preflight decision modes, upload-ID
validation, and deterministic local-rule helpers. It does not run MediaPipe,
specialist models, a web server, or external network calls.
