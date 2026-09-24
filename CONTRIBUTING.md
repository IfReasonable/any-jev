# Contributing

Use Python 3.10+ and `pip install -e '.[dev,vision]'`. Run:

```sh
ruff check .
ruff format --check .
pytest -q
python -m build
twine check dist/*
```

The scoring package must not import the decision package. Integration tests download a tiny model and may be selected with `pytest -m integration`.
