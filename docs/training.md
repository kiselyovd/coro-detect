# Training

```bash
make data
make train
```

### Framingham mirror upload (one-time, operator only)

```bash
hf repo create kiselyovd/framingham --repo-type dataset
hf upload kiselyovd/framingham data/raw/framingham.csv --repo-type dataset
```
