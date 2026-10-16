# Experiment ledger and glossary

Primary comparison incomplete; no claim about an oscillator advantage is supported.

All networks generate unconditional images in pixel space. They share independent conditional flow matching, AdamW, continuous time conditioning, 4×4 patches, and an EMA sampling checkpoint. MNIST is development data; CIFAR-10 is the final comparison domain. There is no curriculum or augmentation. Classification accuracy, text fill rate, and task-completion accuracy are not applicable; run completion is reported separately from image quality.

| Network | Feature computation | Training objective | Role |
| --- | --- | --- | --- |
| onn | Coupled 4D unit-vector rotations; four recurrent blocks | Velocity MSE | Proposed generator |
| dit | Four time-conditioned attention/MLP blocks | Velocity MSE | Parameter-matched transformer |
| conv | Convolution + SiLU + GroupNorm recurrence | Velocity MSE | Conventional recurrent control |
| no_coupling | Oscillators with coupling convolution removed | Velocity MSE | Trained removal; lower capacity |
| no_frequency | Oscillators with natural-frequency term removed | Velocity MSE | Trained removal; slightly lower capacity |

## Training outcomes

| Run | Status | Parameters | Updates | Train seconds | Images/s | Validation MSE |
| --- | --- | --- | --- | --- | --- | --- |
| checks/cifar10-metric-smoke | untrained | 1.29771e+06 | 0 | 0 | — | — |
| checks/mnist-smoke | step_limit | 1.28949e+06 | 100 | 33.9622 | 376.889 | 2.00869 |
| study/pilot/mnist-conv-s101 | completed | 1.28846e+06 | 5245 | 1200.05 | 559.445 | 0.215197 |
| study/pilot/mnist-dit-s101 | completed | 1.29064e+06 | 6778 | 1200.04 | 722.965 | 0.203531 |
| study/pilot/mnist-onn-s101 | completed | 1.28949e+06 | 3736 | 1200.25 | 398.425 | 0.231511 |

## Distribution and cost measurements

| Evaluation | Generated/real N | NFE | KID mean | KID subset SD | FID | Pixel SWD | ms/image |
| --- | --- | --- | --- | --- | --- | --- | --- |
| checks/cifar10-metric-smoke/evaluations/validation | 128/5000 | 32 | 0.609741 | 0.0059603 | 487.666 | — | 21.3627 |
| checks/mnist-smoke/validation-check | 128/5000 | 32 | — | — | — | 1.10101 | 20.9486 |
| study/pilot/mnist-conv-s101/evaluations/validation | 512/5000 | 32 | — | — | — | 0.063441 | 13.336 |
| study/pilot/mnist-dit-s101/evaluations/validation | 512/5000 | 32 | — | — | — | 0.0617331 | 9.42476 |
| study/pilot/mnist-onn-s101/evaluations/validation | 512/5000 | 32 | — | — | — | 0.0714237 | 27.8529 |

## Completeness

Recorded diagnostics: 1. Recorded failed attempts: 1. Completed primary pairs: 0/6. Missing values mean unmeasured, not zero. Individual records and PDFs are in `experiments/`; raw records remain in the run directories and published artifact manifests.

## Glossary

| Term | Meaning |
| --- | --- |
| Flow matching | Learn a velocity that transports Gaussian noise toward the image distribution. |
| Oscillator | A hidden vector whose direction changes under natural rotation and learned interactions. |
| Coupling | A learned convolution that lets hidden oscillators influence one another. |
| Flow time / internal time | Noise-to-image position / recurrent computation inside one velocity prediction. |
| NFE | Number of velocity-network evaluations used to generate each image. |
| KID | Unbiased squared kernel discrepancy between fixed Inception features; smaller is better and estimates may be negative. |
| KID subset SD | Variation across random metric subsets of one fixed pair of image populations, not across trained models. |
| FID | Gaussian feature-distribution distance; finite-sample biased, so sample count must accompany it. |
| Pixel SWD | Wasserstein distance averaged over random pixel-space projections; a MNIST diagnostic, not semantic accuracy. |
| Seed pair | Two independently initialized architectures using paired data/noise/time stream prefixes. |
| Confidence interval | A model-based repeated-study procedure; here an approximate t interval across six seed differences. |
| Sign-flip test | Enumerate all signs of paired differences; exact only under the specified exchangeability/symmetry null. |
| EMA | Exponential moving average of trainable weights, used for all reported samples. |
| Ablation / intervention | Retrain with a mechanism removed / perturb a trained model during inference. |
| Power | Probability of detecting an assumed effect under a planning model; six pairs have about 24% power at effect 0.8. |
