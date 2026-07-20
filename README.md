# csi
Classic Slip Inversion

For a manual, installation instructions and some examples, please visit: http://www.geologie.ens.fr/~jolivet/csi

Local documentation build

Install CSI and docs dependencies:

```bash
pip install -e .
pip install -e ".[docs]"
```

Build docs:

```bash
cd documentation
make html
```

Open generated pages in `documentation/_build/html`.

If you want to cite something: [![DOI](https://zenodo.org/badge/212800412.svg)](https://doi.org/10.5281/zenodo.14170821)

#EOF
