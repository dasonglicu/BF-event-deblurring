# BF Dvent Deblurring

### Installation
```
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Datasets

Follow EFNet to download the GoPro events raw dataset ([Raw_Dataset](https://data.vision.ee.ethz.ch/csakarid/shared/EFNet/GOPRO_rawevents.zip)) to ```./datasets```
  * it should be like:
    ```bash
    ./datasets/
    ./datasets/DATASET_NAME/
    ./datasets/DATASET_NAME/train/
    ./datasets/DATASET_NAME/test/
    ```

### Training
* ```python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train.yml --launcher pytorch```


### Testing
* ```python3 basicsr/test.py -opt options/test.yml```


### Visual results
* The visual results of GOPRO can be downloaded from [Gdrive](https://drive.google.com/file/d/1SpQjgOOqcrJP8ryrM-V36e1wNdyMmFez/view?usp=share_link).
