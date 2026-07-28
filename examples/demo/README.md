# Demo submission

These files form a small simulated submission used by the test suite and README examples.
They contain no real patient records and should not be used to support clinical claims.

Run from the repository root:

```bash
syntrustbench evaluate \
  --real-train examples/demo/real_train.csv \
  --real-test examples/demo/real_test.csv \
  --synthetic examples/demo/synthetic.csv \
  --config configs/example.yaml \
  --output runs/demo
```

The generated `runs/demo/` directory is intentionally not tracked. This keeps the repository
focused on source files while making every example result reproducible.
