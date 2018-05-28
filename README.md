# TS-Net: Combining Modality Specific and Common Features for Multimodal Patch Matching

# Setting up the environement:
  - Python3 
  - Tensorflow 1.4 (GPU version)
  - Numpy
  - Scipy
  - matplotlib
  - skimage
# Dataset: 
To generate new patches for VeDAI, CUHK and NIR-Scene dataset, run the following command in the "data" folder: 
```sh
$ ./generate_dataset.sh
```
The program will automatically download the dataset if you run the code for the first time and prepare tfrecord files ready for training. 

To use the same patches as in our experiment, run the following command in "data" folder:
```sh
$ ./download_icip_dataset.sh
```

# Running the code:

All the codes used to train/evaluate the network are located in "network" folder. In case you are interested in re-training the network, the default checkpoint files should be removed first. These includes three folders: vedai, nirscene and cuhk in "network folder". 

# Training:
To train the network on the three datasets with default parameters, simply run the following command:

```sh
$ python3 matchnet.py --train_test_phase=train --experiment=multimodal
```
Replace matchnet.py with other tsnet.py for training TS-Net. All the tunable parameters are located/listed at the bottom of each model (*.py). For instance, to run Pseudo MatchNet, use: 
```sh
$ python3 matchnet.py --train_test_phase=train --pseudo  --experiment=multimodal
```

In case you are interested in running a single dataset, leave the "experiment" option blank and use "dataset" option. An example to run on vedai is: 
```sh
$ python3 matchnet.py --train_test_phase=train  --dataset=vedai --lr=0.001 batch_size=128
```

To see all the customizable parameters for each model, run with "--help" option. 
```sh
$ python3 matchnet.py --help
```

# Evaluation:
To evaluate the trained network, simply run the following command:
```sh
$ python3 matchnet.py --train_test_phase=test
```
# Notices:

We save checkpoint at every epoch. To evaluate on the test set, we choose the checkpoint that produces the best performance on the validation. To produce a more stable results, the results on the valiation are smoothed (average smoothing) before being chosen.
