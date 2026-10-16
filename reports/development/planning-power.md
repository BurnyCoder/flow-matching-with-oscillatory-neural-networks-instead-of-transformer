# Planning power analysis

This reproduces the pre-training simulation in the protocol. It is not a post-hoc estimate from observed KID differences. Six pairs were retained despite low moderate-effect power because the 24-hour budget defines an exploratory study. Common Gaussian draws are shifted across effect scenarios; all 64 sign assignments are enumerated for every simulated study. The null rejection rate is below 0.05 because the exact test is discrete.

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
      "monte_carlo_se": 0.0012033074731754973
    },
    {
      "standardized_effect": 0.5,
      "power": 0.1133,
      "monte_carlo_se": 0.0022412397239028226
    },
    {
      "standardized_effect": 0.8,
      "power": 0.2406,
      "monte_carlo_se": 0.0030225125309913934
    },
    {
      "standardized_effect": 1,
      "power": 0.35465,
      "monte_carlo_se": 0.0033828492243965
    },
    {
      "standardized_effect": 1.5,
      "power": 0.6567,
      "monte_carlo_se": 0.0033574179811277595
    },
    {
      "standardized_effect": 2,
      "power": 0.8685,
      "monte_carlo_se": 0.002389641709545596
    }
  ]
}
```

Source: [SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
