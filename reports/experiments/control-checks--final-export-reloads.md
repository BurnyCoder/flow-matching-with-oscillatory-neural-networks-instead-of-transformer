# Control: checks/final-export-reloads

## Purpose and interpretation

This diagnostic checks the contract recorded below, such as checkpoint reconstruction, metric repetition, or identity-intervention behavior. It is excluded from primary paired-seed inference and cannot establish an oscillator mechanism by itself.

## Recorded evidence

```json
{
  "status": "completed",
  "kind": "all-reported-checkpoint-export-reloads",
  "device": "cpu",
  "source_revision": "b49e52bb12834bd0f151a4f45a36462182130a8d",
  "publication": "exports/20261017T191030.039367Z/publication.json",
  "compared_checkpoints": 31,
  "all_state_tensors_equal": true,
  "prediction_max_absolute_difference": 0.0,
  "cpu_threads": 4,
  "procedure": "Strictly load original EMA checkpoints and exported safetensors; compare all state tensors and configurations; compare two fixed Gaussian inputs at times 0.25 and 0.75 per checkpoint. Full inputs and both outputs are preserved.",
  "tensor_artifact_sha256": "050793bd048da810ce6135f14dd5d23d4328f0b3488d8b7ecbb18a80ec123d43",
  "checks": [
    {
      "source": "runs/checks/cifar10-metric-smoke",
      "export": "exports/20261017T191030.039367Z/cifar10/checks/cifar10-metric-smoke",
      "tensor_prefix": "run_00",
      "input_seed": 20261016,
      "checkpoint_sha256": "8e8cafa8870ab0ff9c618a39ba1f368be3966ff67459009f4ca195f437cd1aec",
      "exported_weights_sha256": "c6ef33c9235e17feb0c2656fcc29364fa02d5dd9dabf27b05c314c01420090d9",
      "state_tensors": 65,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/checks/mnist-smoke",
      "export": "exports/20261017T191030.039367Z/mnist/checks/mnist-smoke",
      "tensor_prefix": "run_01",
      "input_seed": 20261017,
      "checkpoint_sha256": "8f38d4c79f09a651583cc67b6596ab438a75269ad8fcb6a89db4135257a2a9db",
      "exported_weights_sha256": "89560b05f7e7318c1e71d7f8436e6458e469d07b9544ab602c756368d1482404",
      "state_tensors": 65,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/ablations/cifar10-no_coupling-s101",
      "export": "exports/20261017T191030.039367Z/cifar10/study/ablations/cifar10-no_coupling-s101",
      "tensor_prefix": "run_02",
      "input_seed": 20261018,
      "checkpoint_sha256": "3ec78d32218e855970086aafa4e6930fbfe4204b39228bfb133280ba95086f0d",
      "exported_weights_sha256": "90e80072c9a90978f8f2a30d80412ba0fb7976a049d69dd09c97210c746a5313",
      "state_tensors": 61,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/ablations/cifar10-no_coupling-s102",
      "export": "exports/20261017T191030.039367Z/cifar10/study/ablations/cifar10-no_coupling-s102",
      "tensor_prefix": "run_03",
      "input_seed": 20261019,
      "checkpoint_sha256": "0b3cbc2ac4115efd57aed785b957b2270de7683d46d26a20ed6b0908bd329ad2",
      "exported_weights_sha256": "7b4006f783489f1f3eb11be8f7441aa92b5df640cf7a775ad5498422fb2ac54d",
      "state_tensors": 61,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/ablations/cifar10-no_frequency-s101",
      "export": "exports/20261017T191030.039367Z/cifar10/study/ablations/cifar10-no_frequency-s101",
      "tensor_prefix": "run_04",
      "input_seed": 20261020,
      "checkpoint_sha256": "5de69e1afb911e81cba8030f52489023a73a601e54690a5babf6d0663d9d1b26",
      "exported_weights_sha256": "e414bd1148c47d7eb29817e4d725bdbffb03b37df5abbbc41e78924c149ca58a",
      "state_tensors": 61,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/ablations/cifar10-no_frequency-s102",
      "export": "exports/20261017T191030.039367Z/cifar10/study/ablations/cifar10-no_frequency-s102",
      "tensor_prefix": "run_05",
      "input_seed": 20261021,
      "checkpoint_sha256": "14fa697dfda45dc97daac9ff222051777ca99779e234e429f4c6f1ef4fc2f315",
      "exported_weights_sha256": "395b21f842524dfe11c55360d2093be3d562406a87a0f3af51b154887f33285a",
      "state_tensors": 61,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/checks/fixed-batch",
      "export": "exports/20261017T191030.039367Z/mnist/study/checks/fixed-batch",
      "tensor_prefix": "run_06",
      "input_seed": 20261022,
      "checkpoint_sha256": "5d36c5c4faf91992c6cba4b3bdfb01880b396f817a2ee17924878ada5fe36de0",
      "exported_weights_sha256": "dbc1290adf03cf30958883e06410e5867c99f943e25bdd637684610e552a369f",
      "state_tensors": 65,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/controls/untrained-conv",
      "export": "exports/20261017T191030.039367Z/cifar10/study/controls/untrained-conv",
      "tensor_prefix": "run_07",
      "input_seed": 20261023,
      "checkpoint_sha256": "c3ef26854d2b6b534035ebf5993c7814172283e3a3a7febf89f703798283e6e9",
      "exported_weights_sha256": "a09c911b1517c717c7527fa511e42da3abe44ad546b1ac8c47726cf2dc96af82",
      "state_tensors": 69,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/controls/untrained-dit",
      "export": "exports/20261017T191030.039367Z/cifar10/study/controls/untrained-dit",
      "tensor_prefix": "run_08",
      "input_seed": 20261024,
      "checkpoint_sha256": "f88d4f36bb900a4e46ba2852061a0c8c8d1dcb0f465a8def494aee59c03793ca",
      "exported_weights_sha256": "e0e8e1d0ed79b23b982ae432972278986797979ab46c66c70acb115217b00c18",
      "state_tensors": 52,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/controls/untrained-onn",
      "export": "exports/20261017T191030.039367Z/cifar10/study/controls/untrained-onn",
      "tensor_prefix": "run_09",
      "input_seed": 20261025,
      "checkpoint_sha256": "cd7618119dc6162268c54e9a8b4484d41d52ca10dbdaf76798f2a2afbce56a04",
      "exported_weights_sha256": "c6ef33c9235e17feb0c2656fcc29364fa02d5dd9dabf27b05c314c01420090d9",
      "state_tensors": 65,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-conv-s101",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-conv-s101",
      "tensor_prefix": "run_10",
      "input_seed": 20261026,
      "checkpoint_sha256": "d283104b77bb38b474d78f48ee64bf89e90cd8ecb2fc32d2cced963261e28cbd",
      "exported_weights_sha256": "d20f3947bcf79794427f8215c597a57c4b5eef3b308ccaccf1960beb86b4188e",
      "state_tensors": 69,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-conv-s102",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-conv-s102",
      "tensor_prefix": "run_11",
      "input_seed": 20261027,
      "checkpoint_sha256": "ebb6eafd617b874a4fd559f32996118fdc64418f9206aa38faeb0645c4545b82",
      "exported_weights_sha256": "7c81f89410f40e356ad05be8cb60016c65181208541c303d8ba1665778d1ce3c",
      "state_tensors": 69,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-conv-s103",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-conv-s103",
      "tensor_prefix": "run_12",
      "input_seed": 20261028,
      "checkpoint_sha256": "4cb949dcbdf984ec4658182017a6f2be36da6eb9eb8d6891f6ebaf7119156a43",
      "exported_weights_sha256": "66eb47d34559e0a1127012efc6888a22546eb0b9e9b94e7e28821dbd6c404441",
      "state_tensors": 69,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-conv-s104",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-conv-s104",
      "tensor_prefix": "run_13",
      "input_seed": 20261029,
      "checkpoint_sha256": "384185eaffe897a164f9680f1506f8de0136c16b0d60ff9c491226f04488f574",
      "exported_weights_sha256": "c896d46723b1f6e52eb8b6ad1e8fbdfc9c889784b5e114fe73606f538ed3d1ac",
      "state_tensors": 69,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-conv-s105",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-conv-s105",
      "tensor_prefix": "run_14",
      "input_seed": 20261030,
      "checkpoint_sha256": "1f90d6fd0182341298cc1812aca672144633c8ea7c6ab57816bf9949e597352f",
      "exported_weights_sha256": "d872edbf5e8bf7c11f03b6637adfc2762a009a6b0ea659f48e595e4d853cd5b7",
      "state_tensors": 69,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-conv-s106",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-conv-s106",
      "tensor_prefix": "run_15",
      "input_seed": 20261031,
      "checkpoint_sha256": "fc27f7e6924da1c7426ca800f5916d5836ea6f8285d7fc926ba4ab981c156194",
      "exported_weights_sha256": "2057aaaf7d16b4220dfdc8bc9d8ef32a6e91388223e30e65c2d72252f4562cd4",
      "state_tensors": 69,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-dit-s101",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-dit-s101",
      "tensor_prefix": "run_16",
      "input_seed": 20261032,
      "checkpoint_sha256": "030d935262170666e1c5f3da7020026a573aa3de03b9f86a9c67f036afbe0dce",
      "exported_weights_sha256": "cc227d4e18a7e1d94a2a9e0d34e809f86a8798e0c8e283af84968e529a1c8a67",
      "state_tensors": 52,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-dit-s102",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-dit-s102",
      "tensor_prefix": "run_17",
      "input_seed": 20261033,
      "checkpoint_sha256": "a87b3e659c6398bf2b2791e05297293a8c154ada863778310148018b4f940c15",
      "exported_weights_sha256": "2e584a96062f1a454dce28bb6322e17731df8999e32045b8ed7f9fe8fb4453cc",
      "state_tensors": 52,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-dit-s103",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-dit-s103",
      "tensor_prefix": "run_18",
      "input_seed": 20261034,
      "checkpoint_sha256": "e737abf235c2b51f00a3e768fc58f991acaee1881c8e43ad60fa46de0b8ecdd8",
      "exported_weights_sha256": "75bca1cf3738374c420f34e36b0679e6369cf3fa86343aa762f0ff85bd6df64b",
      "state_tensors": 52,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-dit-s104",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-dit-s104",
      "tensor_prefix": "run_19",
      "input_seed": 20261035,
      "checkpoint_sha256": "931c047b6ac7ef2a657e898a26f94ab432497b05a07e2c281eef36f50e20588a",
      "exported_weights_sha256": "3aba95b3af00dd2cad93f206c5af6d73f07431b491ed2e358a2feb8c8a1991c1",
      "state_tensors": 52,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-dit-s105",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-dit-s105",
      "tensor_prefix": "run_20",
      "input_seed": 20261036,
      "checkpoint_sha256": "eec459e41060cbff6d07971d0c2fd363ccc1f525e23cc72c6b5c872e2cf832ac",
      "exported_weights_sha256": "38b99f5c030c04a12c83c9b41ee55c16a4cde20c557e29b274950dbeb6d4a292",
      "state_tensors": 52,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-dit-s106",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-dit-s106",
      "tensor_prefix": "run_21",
      "input_seed": 20261037,
      "checkpoint_sha256": "c05ea7f6e8db40d1fee8137f96251a02fd2c1ee199d91e2214aa46a17c51cc5e",
      "exported_weights_sha256": "f4b8c7eddf76c637e828ccb902517b2f2efbfb2e269986f81bd5ac965e541f3e",
      "state_tensors": 52,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-onn-s101",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-onn-s101",
      "tensor_prefix": "run_22",
      "input_seed": 20261038,
      "checkpoint_sha256": "e3eb0aabeaf1f372fbcb2f23bf94625b9fd2a1424504818a02b5d37a4522d6b1",
      "exported_weights_sha256": "bb7094d51557bb6d7ccfdbaebd393c3cc33bbfcb48963aba33e58c48af187f99",
      "state_tensors": 65,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-onn-s102",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-onn-s102",
      "tensor_prefix": "run_23",
      "input_seed": 20261039,
      "checkpoint_sha256": "e5cf67558e832212ded9b04225e34091cd62715d9df0773c5d22685da64e8366",
      "exported_weights_sha256": "06a34f3c2e2dae9947112142592c6747376ba44959be8efb0ba37bc3687276cd",
      "state_tensors": 65,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-onn-s103",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-onn-s103",
      "tensor_prefix": "run_24",
      "input_seed": 20261040,
      "checkpoint_sha256": "51090d973909f69682a6f2c126cf1377fbda43f249646733df76a4256e4566a1",
      "exported_weights_sha256": "93b25b900dff3df0f60701b0456a0b45c89eba9e2da7f539f2c290f575f5e4e7",
      "state_tensors": 65,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-onn-s104",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-onn-s104",
      "tensor_prefix": "run_25",
      "input_seed": 20261041,
      "checkpoint_sha256": "a0262716020a4118b0f750cf092274f8748ae86cb5ea23b3a700d75bc86e6322",
      "exported_weights_sha256": "b306ed41c2b415cf5be073c0bbf0c4fcd4fdfee714ac4eab05aff3061021f620",
      "state_tensors": 65,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-onn-s105",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-onn-s105",
      "tensor_prefix": "run_26",
      "input_seed": 20261042,
      "checkpoint_sha256": "6ae577c27cfe5120f2824baff526173d912511a47c56665da170c0b5a227251a",
      "exported_weights_sha256": "12d9c6bfed79fd4a812af0b18f053486c5852171826f6e0eb453c4868fa3bd00",
      "state_tensors": 65,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/main/cifar10-onn-s106",
      "export": "exports/20261017T191030.039367Z/cifar10/study/main/cifar10-onn-s106",
      "tensor_prefix": "run_27",
      "input_seed": 20261043,
      "checkpoint_sha256": "e9595ec0d0fe7e945b48301e630d059d2077a641c3da27f0b6c205899f4da0b8",
      "exported_weights_sha256": "c8b04e345990d920092c9b209471de71092020c363ac4004d631fddb476b3984",
      "state_tensors": 65,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/pilot/mnist-conv-s101",
      "export": "exports/20261017T191030.039367Z/mnist/study/pilot/mnist-conv-s101",
      "tensor_prefix": "run_28",
      "input_seed": 20261044,
      "checkpoint_sha256": "3fffc15466b2561ab3b0ebe9ad66272c4c4e3fe99e0f13602910d1f7da80b651",
      "exported_weights_sha256": "bffd1bfd5e174490442d518b745e2f1da3416b0e2594b714f47a5dd9f16c1e31",
      "state_tensors": 69,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/pilot/mnist-dit-s101",
      "export": "exports/20261017T191030.039367Z/mnist/study/pilot/mnist-dit-s101",
      "tensor_prefix": "run_29",
      "input_seed": 20261045,
      "checkpoint_sha256": "628a9d5c15326d0a89bb13ecab677c36bbe22b0de447d3e9bba981f82e164ee0",
      "exported_weights_sha256": "a72223e84aa002fcb68459de860540f70499c1964584c2b1b37b437d2e327366",
      "state_tensors": 52,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    },
    {
      "source": "runs/study/pilot/mnist-onn-s101",
      "export": "exports/20261017T191030.039367Z/mnist/study/pilot/mnist-onn-s101",
      "tensor_prefix": "run_30",
      "input_seed": 20261046,
      "checkpoint_sha256": "f11cdc5df1a58a9be3dd5973d540755c240dcb1c67246627de8aaf8229e0b7e6",
      "exported_weights_sha256": "06201135c2255dcc8b2ee988f9ce128ce6fe2e6eb35f3fd142b3a4db415fec96",
      "state_tensors": 65,
      "all_state_tensors_equal": true,
      "prediction_max_absolute_difference": 0.0
    }
  ],
  "id": "checks/final-export-reloads",
  "artifact": "runs/checks/final-export-reloads/control.json"
}
```

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
