# Conditioning Generative Latent Optimization to solve Imaging Inverse Problems

Pytorch implementation of cGLO method described in
[Conditioning Generative Latent Optimization for Sparse-View CT Image Reconstruction](https://arxiv.org/abs/2307.16670)

--------------------

### Installation

The code should work with most versions of packages listed in
`requirements.txt`. From this repository root, and preferably inside a
dedicated [virtual environment](https://virtualenv.pypa.io/en/latest/),
depedencies can be installed alongside `cglo` with :

```bash
pip install -U -r requirements.txt --no-cache-dir
python setup.py install
```

Specific environment details can be found in the `Dockerfile`, which can be
built using :

```bash
docker build -t mydocker .
```

### References

If you find this repository useful, please consider citing

```bib
@misc{braure2024conditioning,
      title={Conditioning Generative Latent Optimization for Sparse-View CT Image Reconstruction},
      author={Thomas Braure and Delphine Lazaro and David Hateau and Vincent Brandon and Kévin Ginsburger},
      year={2024},
      eprint={2307.16670},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

