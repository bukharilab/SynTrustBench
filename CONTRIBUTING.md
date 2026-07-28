# Contributing

Changes should keep the evidence assessment and executable protocol aligned.

## Development setup

```bash
python -m pip install -e ".[dev,evidence]"
pytest
python scripts/evidence/validate_release.py
```

## Pull requests

- Explain the scientific or usability reason for the change.
- Add or update tests for metric and schema changes.
- Document any changed threshold, resampling unit, threat model, or failure condition.
- Do not commit generated run directories, credentials, or restricted clinical data.
- Keep benchmark gates distinct from regulatory or safety certification.

Metric changes should include a small controlled example showing the expected directional
response. Changes to evidence coding should retain the primary-source locator and correction
history.
