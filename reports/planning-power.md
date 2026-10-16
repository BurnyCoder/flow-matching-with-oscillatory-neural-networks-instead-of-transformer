# Planning power analysis

This reproduces the pre-training simulation in the protocol. It is not a post-hoc estimate from observed KID differences. Six pairs were retained despite low moderate-effect power because the 24-hour budget defines an exploratory study. Common Gaussian draws are shifted across effect scenarios; all 64 sign assignments are enumerated for every simulated study. The null rejection rate is below 0.05 because the exact test is discrete. An analytical cross-check added during implementation uses the fact that six nonzero differences yield a two-sided minimum p-value of 2/64; the next level is 4/64. At alpha 0.05, rejection therefore requires all six signs to agree. For independent N(d,1) differences the exact power is Phi(d)^6 + Phi(-d)^6, and null size is 0.03125. This cross-check does not change the pre-selected seed count or stopping rule.

```json
{
  "seed": 20261016,
  "trials": 20000,
  "pairs": 6,
  "alpha": 0.05,
  "model": "independent Gaussian paired differences",
  "common_random_numbers_across_effects": true,
  "results": [
    {
      "standardized_effect": 0,
      "power": 0.02985,
      "monte_carlo_se": 0.0012033074731754973,
      "exact_gaussian_power": 0.03125
    },
    {
      "standardized_effect": 0.5,
      "power": 0.1133,
      "monte_carlo_se": 0.0022412397239028226,
      "exact_gaussian_power": 0.11016053356617381
    },
    {
      "standardized_effect": 0.8,
      "power": 0.2406,
      "monte_carlo_se": 0.0030225125309913934,
      "exact_gaussian_power": 0.23977242018781017
    },
    {
      "standardized_effect": 1,
      "power": 0.35465,
      "monte_carlo_se": 0.0033828492243965,
      "exact_gaussian_power": 0.35470184724805504
    },
    {
      "standardized_effect": 1.5,
      "power": 0.6567,
      "monte_carlo_se": 0.0033574179811277595,
      "exact_gaussian_power": 0.6604323391819238
    },
    {
      "standardized_effect": 2,
      "power": 0.8685,
      "monte_carlo_se": 0.002389641709545596,
      "exact_gaussian_power": 0.8710312226999941
    }
  ]
}
```

Source: [SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
