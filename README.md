# GED-Net
GED-Net for Ultrasound Image Segmentation. still being organized and updated

## Requirements

- Python 3.10
- PyTorch 2.1
- torchvision
- torch
- numpy
- tqdm
- sklearn
- thop
- Pillow
- future
- opencv-python

Or you can install the required packages using pip:

```
pip install -r requirements.txt
```



## Dataset

The structure of the dataset is:

dataset/
├── imgs/
│   ├── image1.png
│   ├── image2.png
│   ├── image3.png
│   └── ...
├── masks/
│   ├── image1.png
│   ├── image2.png
│   ├── image3.png
│   └── ...

And the dataset must be correctly configured in `train.py`， `base = Path(os.environ['raw_data_base']) if 'raw_data_base' in os.environ.keys() else Path('./your dataset')`

## Usage

To train a model, run the following command:

```
python train.py -e [epochs] -b [batch_size] -l [learning_rate]  -nf [X fold cross-validation] --folds [fold_numbers] --cuda_device [device_index]
```

Here are the descriptions of the command line arguments:

- `-e`, `--epochs`: Number of epochs to train the model (default: 100)
- `-b`, `--batch-size`: Batch size for training (default: 8)
- `-l`, `--learning-rate`: Learning rate for the optimizer (default: 0.0001)
- `-s`, `--scale`: Downscaling factor for the input images (default: 0.4375)
- `-nf`: X fold cross-validation(default: 5)
- `--folds`: Specify the folds to train (e.g., --folds 0 1 2) if you want to run all folds --folds 0 1 2 3 4 (default: [0])
- `--cuda_device`: CUDA device index (default: 0)

Example:

```cmd
python train.py -e 50 -b 2 -l 0.001 -nf 5--folds 0 1 2 --cuda_device 0
```

This command will train the model for 50 epochs with a batch size of 2, a learning rate of 0.001, and a downscaling factor of 0.5. It will use 10% of the data for validation and train on folds 0, 1, and 2 using CUDA device 0.

## Models

You can add any models you want to run just import it and set the correct mode name ,and implemente it in this codebase.

You can choose which model to train by modifying the `model_name` variable in the `train.py` script.

## Checkpoints

The code saves checkpoints of the trained models in the `checkpoint/` directory. You can load a pre-trained model by setting the `-load` flag when running the `train.py` script:

```
python train.py -e 50 -b 8 -l 0.001 --load path/to/checkpoint.pth --cuda_device 0
```

This command will load the weights from the specified checkpoint and continue training the model.

## Results

The segmentation results are auto saved in the `result_output_BUSI_malignant_Pre/{model_name}/fold_{fold_number}/` directory. The results include  predicted masks. If you want to run it on you own datasets,  just reset the `result_output_BUSI_malignant`  to  you own dataset name.  You can use a separate segmentation metric to compare it with the mask



## Topology

If you want to get the connections or topology of the various parts of the graph, you can get it via `get_neighbours.py`, where edge_index stores 2 arrays, the first array is the id of the query and the second is the id of the node that is connected to the query; the ids of each patch are listed from left to right, top to bottom.

## Logging

The training process is logged using the Python `logging` library. The log messages are printed to the console and saved to a file named `logger.log` in the root directory.
