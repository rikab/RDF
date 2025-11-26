# Resummed Distribution Functions (v1.0.0)
### By Rikab Gambhir and Radha Mastandrea

[![GitHub Project](https://img.shields.io/badge/GitHub--blue?style=social&logo=GitHub)](https://github.com/rikab/RDF)

`Resummed Distribution Functions` (RDFs) are a framework for performing all-orders extensions of a fixed order calculation to be consistent with unitarity, and for performing statistical fits, as defined in ["Resummed Distribution Functions: Making Perturbation Theory Positive and Normalized" (arxiv:25XX.XXXXX)](https://arxiv.org/abs/25XX.XXXXX). This was used to perform all the analyses shown in this paper, and can be used to perform similar analyses.

![thumbnail](https://github.com/rikab/RDF/blob/main/thumbnail.png)

Pictured: Example of using the RDF to match to finite Taylor expansions of the exponential distribution. While any fixed-order expansion is never a true distribution, every RDF matching is.

This repo contains::

- "Analytic" RDFs, which can be used to perform matching to a known closed-form calculation.
- "Numeric" RDFs, which can be used to perform matching to a numeric fixed-order calculation (such as those performed by MadGraph or EERAD3), and seperately, for fitting the full RDF to data with nuisance parameters.

Files containing fixed-order calculations generated ussing EERAD3 up to $\alpha_s^3$ may be found in `numeric/data`. We also save our final RDF parameters for all the studies performed in [(arxiv:25XX.XXXXX)](https://arxiv.org/abs/25XX.XXXXX) in `numeric/output_JAX`.


## Dependencies

The primary dependencies are `jax` and `jaxlib`. 

To install jax and jaxlib, run the following commands:

```bash
pip install --upgrade pip
pip install --upgrade jax jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```


## Citation

If you use this repo, please cite the corresponding paper, ["Resummed Distribution Functions: Making Perturbation Theory Positive and Normalized" (arxiv:25XX.XXXXX)](https://arxiv.org/abs/25XX.XXXXX):

TODO: ADD WHEN PAPER IS OUT.


## Changelog

- v1.0.0: XX XXXXX 2025. Official Release.


Based on the work in ["Resummed Distribution Functions: Making Perturbation Theory Positive and Normalized" (arxiv:25XX.XXXXX)](https://arxiv.org/abs/25XX.XXXXX)

Bugs, Fixes, Ideas, or Questions? Contact us at gambhirb@ucmail.uc.edu and rmastand@uchicago.edu
