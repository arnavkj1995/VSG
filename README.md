## Learning Robust Dynamics Through Variational Sparse Gating

####  [[Project Website]]() [[BringBackShapes Code]](https://github.com/arnavkj1995/BBS) [[Video]]() 

Learning world models from their sensory inputs enables agents to plan for actions by imagining their future outcomes. World models have previously been shown to improve sample-efficiency in simulated environments with few objects, but have not yet been applied successfully to environments with many objects. In environments with many objects, often only a small number of them are moving or interacting at the same time. In this paper, we investigate integrating this inductive bias of sparse interactions into the latent dynamics of world models trained from pixels. First, we introduce Variational Sparse Gating (VSG), a latent dynamics model that updates its feature dimensions sparsely through stochastic binary gates. Moreover, we propose a simplified architecture Simple Variational Sparse Gating (SVSG) that removes the deterministic pathway of previous models, resulting in a fully stochastic transition function that leverages the VSG mechanism. We evaluate the two model architectures in the BringBackShapes (BBS) environment that features a large number of moving objects and partial observability, demonstrating clear improvements over prior models.

<p align="center">
<img src="https://arnavkj1995.github.io/images/Jain22.png" width="700">
</p>

### Setup
The dependencies can be installed using the `requirements.txt` file:

```shell
cd VSG
virtualenv --no-download VSG
source VSG/bin/activate
pip install --upgrade pip
pip install tensorflow==2.4.1 tensorflow_probability==0.12.2
pip install -r requirements.txt
```

NOTE:
In case there are issues with numpy, specifically `NotImplementedError: Cannot convert a symbolic Tensor (strided_slice:0) to a numpy array.`, follow the fix mentioned [here](https://github.com/tensorflow/models/issues/9706#issuecomment-792106149).

### BringBackShapes

To conduct experiments on the proposed BringBackShapes environment, first install the environment following the instructions [here](https://github.com/arnavkj1995/BBS)

```shell
bash scripts/bringbackshapes.sh {MODEL} {SUFFIX} sparse 3000 {DISTRACTORS} 5 False False False 125 {SIZE} {GATE_PRIOR} {SEED}
```

| Variable       | Description                          |
|------------|-----------------------------------------------|
| DISTRACTORS | Number of stochastic distractors in the env        |
| SIZE        | Scale of the area to control partial observability, 1.0 refers to the Basic version |
| MODEL       | Name of the agent |
| SUFFIX      | Name of the experiment for logging |
| GATE_PRIOR  | Gate prior probabilities for sparse gating mechanism in VSG/SVSG |
| SEED        | Seed parameter |

An example run is
```shell
bash scripts/bringbackshapes.sh VSG baseline sparse 3000 0 5 False False False 125 1.0 0.4 1
```

### DeepMind Control Suite
For running experiments on tasks from DeepMind Control Suite, first install the `dm_control` repository following the instructions [here](https://github.com/deepmind/dm_control). Then use the command below to run with the appropriate config.

| Variable       | Description                          |
|------------|-----------------------------------------------|
| TASK       | DMC Task to train on                          |
| MODEL      | Choose from DreamerV1, DreamerV2, VSG or SVSG |
| SUFFIX     | Name of the experiment for logging            |
| GATE PRIOR | Prior gate probability for VSG or SVSG        |
| DIM        | Size of the latent state                      |
| SEED       | Seed parameter               |

```shell
bash scripts/dmc.sh {TASK} {MODEL} {SUFFIX} {GATE_PRIOR} {DIM} {SEED}
```

For example to run `VSG` on walker_walk,

```shell
bash scripts/dmc.sh walker_walk VSG baseline 0.4 1024 1
```
### Bibtex
If you find this code useful, please reference in your paper:

```
@InProceedings{Jain22,
    author    = "Jain, Arnav Kumar and Sujit, Shivakanth and Joshi, Shruti and Michalski, Vincent and Hafner, Danijar and Kahou, Samira Ebrahimi",
    title     = "Learning Robust Dynamics through Variational Sparse Gating",
    booktitle = {Advances in Neural Information Processing Systems},
    month     = {December},
    year      = {2022}
  }
```

### Acknowledgements
This code was developed using [DreamerV2](https://github.com/danijar/dreamerv2).