# Experiment ledger and glossary

The observed positive KID difference contradicts the proposed oscillator improvement in this regime.

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
| [checks/cifar10-metric-smoke](<experiments/training-checks--cifar10-metric-smoke.md>) | untrained | 1297712 | 0 | 0 | — | — |
| [checks/mnist-smoke](<experiments/training-checks--mnist-smoke.md>) | step_limit | 1289488 | 100 | 33.9622 | 376.889 | 2.00869 |
| [study/ablations/cifar10-no_coupling-s101](<experiments/training-study--ablations--cifar10-no_coupling-s101.md>) | completed | 707888 | 9405 | 2700.28 | 445.82 | 0.262754 |
| [study/ablations/cifar10-no_coupling-s102](<experiments/training-study--ablations--cifar10-no_coupling-s102.md>) | completed | 707888 | 10045 | 2700.04 | 476.2 | 0.266352 |
| [study/ablations/cifar10-no_frequency-s101](<experiments/training-study--ablations--cifar10-no_frequency-s101.md>) | completed | 1295664 | 10320 | 2700.08 | 489.23 | 0.211623 |
| [study/ablations/cifar10-no_frequency-s102](<experiments/training-study--ablations--cifar10-no_frequency-s102.md>) | completed | 1295664 | 10583 | 2700.13 | 501.688 | 0.216272 |
| [study/controls/untrained-conv](<experiments/training-study--controls--untrained-conv.md>) | untrained | 1296688 | 0 | 0 | — | — |
| [study/controls/untrained-dit](<experiments/training-study--controls--untrained-dit.md>) | untrained | 1298864 | 0 | 0 | — | — |
| [study/controls/untrained-onn](<experiments/training-study--controls--untrained-onn.md>) | untrained | 1297712 | 0 | 0 | — | — |
| [study/main/cifar10-conv-s101](<experiments/training-study--main--cifar10-conv-s101.md>) | completed | 1296688 | 12473 | 2700.18 | 591.272 | 0.203593 |
| [study/main/cifar10-conv-s102](<experiments/training-study--main--cifar10-conv-s102.md>) | completed | 1296688 | 12365 | 2700.18 | 586.153 | 0.210089 |
| [study/main/cifar10-conv-s103](<experiments/training-study--main--cifar10-conv-s103.md>) | completed | 1296688 | 12482 | 2700.07 | 591.725 | 0.212733 |
| [study/main/cifar10-conv-s104](<experiments/training-study--main--cifar10-conv-s104.md>) | completed | 1296688 | 12480 | 2700.04 | 591.637 | 0.210087 |
| [study/main/cifar10-conv-s105](<experiments/training-study--main--cifar10-conv-s105.md>) | completed | 1296688 | 12236 | 2700.17 | 580.04 | 0.213325 |
| [study/main/cifar10-conv-s106](<experiments/training-study--main--cifar10-conv-s106.md>) | completed | 1296688 | 11489 | 2700.03 | 544.658 | 0.221481 |
| [study/main/cifar10-dit-s101](<experiments/training-study--main--cifar10-dit-s101.md>) | completed | 1298864 | 16959 | 2700.17 | 803.93 | 0.192738 |
| [study/main/cifar10-dit-s102](<experiments/training-study--main--cifar10-dit-s102.md>) | completed | 1298864 | 17303 | 2700.06 | 820.271 | 0.197029 |
| [study/main/cifar10-dit-s103](<experiments/training-study--main--cifar10-dit-s103.md>) | completed | 1298864 | 17148 | 2700.12 | 812.907 | 0.201229 |
| [study/main/cifar10-dit-s104](<experiments/training-study--main--cifar10-dit-s104.md>) | completed | 1298864 | 16981 | 2700.03 | 805.017 | 0.198842 |
| [study/main/cifar10-dit-s105](<experiments/training-study--main--cifar10-dit-s105.md>) | completed | 1298864 | 15926 | 2700.13 | 754.973 | 0.201822 |
| [study/main/cifar10-dit-s106](<experiments/training-study--main--cifar10-dit-s106.md>) | completed | 1298864 | 15345 | 2700.11 | 727.438 | 0.208719 |
| [study/main/cifar10-onn-s101](<experiments/training-study--main--cifar10-onn-s101.md>) | completed | 1297712 | 8589 | 2700.05 | 407.175 | 0.215884 |
| [study/main/cifar10-onn-s102](<experiments/training-study--main--cifar10-onn-s102.md>) | completed | 1297712 | 8588 | 2700.09 | 407.122 | 0.220631 |
| [study/main/cifar10-onn-s103](<experiments/training-study--main--cifar10-onn-s103.md>) | completed | 1297712 | 8632 | 2700.27 | 409.179 | 0.224278 |
| [study/main/cifar10-onn-s104](<experiments/training-study--main--cifar10-onn-s104.md>) | completed | 1297712 | 8536 | 2700.11 | 404.654 | 0.221115 |
| [study/main/cifar10-onn-s105](<experiments/training-study--main--cifar10-onn-s105.md>) | completed | 1297712 | 8248 | 2700.25 | 390.981 | 0.226086 |
| [study/main/cifar10-onn-s106](<experiments/training-study--main--cifar10-onn-s106.md>) | completed | 1297712 | 7642 | 2700.3 | 362.248 | 0.234817 |
| [study/pilot/mnist-conv-s101](<experiments/training-study--pilot--mnist-conv-s101.md>) | completed | 1288464 | 5245 | 1200.05 | 559.445 | 0.215197 |
| [study/pilot/mnist-dit-s101](<experiments/training-study--pilot--mnist-dit-s101.md>) | completed | 1290640 | 6778 | 1200.04 | 722.965 | 0.203531 |
| [study/pilot/mnist-onn-s101](<experiments/training-study--pilot--mnist-onn-s101.md>) | completed | 1289488 | 3736 | 1200.25 | 398.425 | 0.231511 |

## Distribution and cost measurements

| Evaluation | Generated/real N | NFE | KID mean | KID subset SD | FID | Pixel SWD | ms/image |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [checks/cifar10-metric-smoke/evaluations/validation](<experiments/evaluation-checks--cifar10-metric-smoke--evaluations--validation.md>) | 128/5000 | 32 | 0.609741 | 0.0059603 | 487.666 | — | 21.3627 |
| [checks/mnist-smoke/validation-check](<experiments/evaluation-checks--mnist-smoke--validation-check.md>) | 128/5000 | 32 | — | — | — | 1.10101 | 20.9486 |
| [study/ablations/cifar10-no_coupling-s101/evaluations/test-nfe32](<experiments/evaluation-study--ablations--cifar10-no_coupling-s101--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.26963 | 0.00267395 | 269.031 | — | 19.4271 |
| [study/ablations/cifar10-no_coupling-s102/evaluations/test-nfe32](<experiments/evaluation-study--ablations--cifar10-no_coupling-s102--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.267981 | 0.00290885 | 266.878 | — | 22.0612 |
| [study/ablations/cifar10-no_frequency-s101/evaluations/test-nfe32](<experiments/evaluation-study--ablations--cifar10-no_frequency-s101--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.169792 | 0.0025135 | 163.113 | — | 16.5046 |
| [study/ablations/cifar10-no_frequency-s102/evaluations/test-nfe32](<experiments/evaluation-study--ablations--cifar10-no_frequency-s102--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.160263 | 0.0024495 | 156.719 | — | 16.0385 |
| [study/controls/untrained-conv/evaluations/test](<experiments/evaluation-study--controls--untrained-conv--evaluations--test.md>) | 5000/10000 | 32 | 0.604058 | 0.00279767 | 456.73 | — | 19.9959 |
| [study/controls/untrained-dit/evaluations/test](<experiments/evaluation-study--controls--untrained-dit--evaluations--test.md>) | 5000/10000 | 32 | 0.601732 | 0.00281394 | 455.442 | — | 10.97 |
| [study/controls/untrained-onn/evaluations/test](<experiments/evaluation-study--controls--untrained-onn--evaluations--test.md>) | 5000/10000 | 32 | 0.599606 | 0.00259296 | 455.38 | — | 28.3087 |
| [study/main/cifar10-conv-s101/evaluations/test-nfe16](<experiments/evaluation-study--main--cifar10-conv-s101--evaluations--test-nfe16.md>) | 5000/10000 | 16 | 0.129412 | 0.0022568 | 127.666 | — | 7.59598 |
| [study/main/cifar10-conv-s101/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-conv-s101--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.132659 | 0.00224372 | 132.002 | — | 13.9111 |
| [study/main/cifar10-conv-s101/evaluations/test-nfe64](<experiments/evaluation-study--main--cifar10-conv-s101--evaluations--test-nfe64.md>) | 5000/10000 | 64 | 0.135467 | 0.00227747 | 135.195 | — | 28.4834 |
| [study/main/cifar10-conv-s101/evaluations/test-nfe8](<experiments/evaluation-study--main--cifar10-conv-s101--evaluations--test-nfe8.md>) | 5000/10000 | 8 | 0.127097 | 0.00227732 | 122.565 | — | 3.69969 |
| [study/main/cifar10-conv-s102/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-conv-s102--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.137173 | 0.0022348 | 136.479 | — | 14.257 |
| [study/main/cifar10-conv-s103/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-conv-s103--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.131977 | 0.00215866 | 130.711 | — | 17.6869 |
| [study/main/cifar10-conv-s104/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-conv-s104--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.131518 | 0.00212369 | 130.908 | — | 14.792 |
| [study/main/cifar10-conv-s105/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-conv-s105--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.13401 | 0.00216785 | 133.047 | — | 17.3853 |
| [study/main/cifar10-conv-s106/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-conv-s106--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.141894 | 0.00228545 | 141.231 | — | 16.7635 |
| [study/main/cifar10-dit-s101/evaluations/test-nfe16](<experiments/evaluation-study--main--cifar10-dit-s101--evaluations--test-nfe16.md>) | 5000/10000 | 16 | 0.0857987 | 0.00165292 | 98.6546 | — | 5.21638 |
| [study/main/cifar10-dit-s101/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-dit-s101--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.0909215 | 0.00172645 | 104.111 | — | 10.1586 |
| [study/main/cifar10-dit-s101/evaluations/test-nfe64](<experiments/evaluation-study--main--cifar10-dit-s101--evaluations--test-nfe64.md>) | 5000/10000 | 64 | 0.0952455 | 0.00181027 | 108.063 | — | 21.3938 |
| [study/main/cifar10-dit-s101/evaluations/test-nfe8](<experiments/evaluation-study--main--cifar10-dit-s101--evaluations--test-nfe8.md>) | 5000/10000 | 8 | 0.0838022 | 0.00163955 | 95.6179 | — | 2.4516 |
| [study/main/cifar10-dit-s102/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-dit-s102--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.0899517 | 0.00169648 | 103.125 | — | 9.77963 |
| [study/main/cifar10-dit-s103/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-dit-s103--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.0920323 | 0.00184124 | 105.047 | — | 10.1741 |
| [study/main/cifar10-dit-s104/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-dit-s104--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.0926866 | 0.00191923 | 105.141 | — | 13.4588 |
| [study/main/cifar10-dit-s105/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-dit-s105--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.0905512 | 0.00181148 | 104.204 | — | 12.1196 |
| [study/main/cifar10-dit-s106/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-dit-s106--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.0945936 | 0.00188419 | 106.249 | — | 9.76628 |
| [study/main/cifar10-onn-s101/evaluations/identity](<experiments/evaluation-study--main--cifar10-onn-s101--evaluations--identity.md>) | 5000/10000 | 32 | 0.188511 | 0.00274027 | 176.705 | — | 23.2444 |
| [study/main/cifar10-onn-s101/evaluations/shuffle](<experiments/evaluation-study--main--cifar10-onn-s101--evaluations--shuffle.md>) | 5000/10000 | 32 | 0.263509 | 0.00299769 | 240.955 | — | 23.6636 |
| [study/main/cifar10-onn-s101/evaluations/test-nfe16](<experiments/evaluation-study--main--cifar10-onn-s101--evaluations--test-nfe16.md>) | 5000/10000 | 16 | 0.178735 | 0.00259012 | 168.147 | — | 11.3092 |
| [study/main/cifar10-onn-s101/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-onn-s101--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.188511 | 0.00274027 | 176.705 | — | 21.099 |
| [study/main/cifar10-onn-s101/evaluations/test-nfe64](<experiments/evaluation-study--main--cifar10-onn-s101--evaluations--test-nfe64.md>) | 5000/10000 | 64 | 0.19384 | 0.00281791 | 181.258 | — | 46.2968 |
| [study/main/cifar10-onn-s101/evaluations/test-nfe8](<experiments/evaluation-study--main--cifar10-onn-s101--evaluations--test-nfe8.md>) | 5000/10000 | 8 | 0.174378 | 0.00246556 | 162.823 | — | 6.22954 |
| [study/main/cifar10-onn-s102/evaluations/identity](<experiments/evaluation-study--main--cifar10-onn-s102--evaluations--identity.md>) | 5000/10000 | 32 | 0.179221 | 0.00275456 | 170.893 | — | 26.7271 |
| [study/main/cifar10-onn-s102/evaluations/shuffle](<experiments/evaluation-study--main--cifar10-onn-s102--evaluations--shuffle.md>) | 5000/10000 | 32 | 0.257773 | 0.00364871 | 233.55 | — | 27.1896 |
| [study/main/cifar10-onn-s102/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-onn-s102--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.179221 | 0.00275456 | 170.893 | — | 21.6993 |
| [study/main/cifar10-onn-s103/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-onn-s103--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.181497 | 0.00273838 | 171.989 | — | 24.4268 |
| [study/main/cifar10-onn-s104/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-onn-s104--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.176938 | 0.00259902 | 167.294 | — | 24.6136 |
| [study/main/cifar10-onn-s105/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-onn-s105--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.186201 | 0.00282784 | 176.039 | — | 27.2132 |
| [study/main/cifar10-onn-s106/evaluations/test-nfe32](<experiments/evaluation-study--main--cifar10-onn-s106--evaluations--test-nfe32.md>) | 5000/10000 | 32 | 0.201672 | 0.0028731 | 187.592 | — | 21.3573 |
| [study/pilot/mnist-conv-s101/evaluations/validation](<experiments/evaluation-study--pilot--mnist-conv-s101--evaluations--validation.md>) | 512/5000 | 32 | — | — | — | 0.063441 | 13.336 |
| [study/pilot/mnist-dit-s101/evaluations/validation](<experiments/evaluation-study--pilot--mnist-dit-s101--evaluations--validation.md>) | 512/5000 | 32 | — | — | — | 0.0617331 | 9.42476 |
| [study/pilot/mnist-onn-s101/evaluations/validation](<experiments/evaluation-study--pilot--mnist-onn-s101--evaluations--validation.md>) | 512/5000 | 32 | — | — | — | 0.0714237 | 27.8529 |

## Completeness

Recorded diagnostics: 1. Recorded failed attempts: 2. Completed primary pairs: 6/6. Missing values mean unmeasured, not zero. Individual records and PDFs are in `experiments/`; raw records remain in the run directories and published artifact manifests.

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
