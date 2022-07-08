# MEANet

Pytorch implementation for [MEANet:  Multi-Modal Edge-Aware Network for Light Field Salient Object Detection](https://www.sciencedirect.com/science/article/pii/S0925231222003502).


# Requirements
* Python 3.6 <br>
* Torch 1.10.2 <br>
* Torchvision 0.4.0 <br>
* Cuda 10.0 <br>
* Tensorboard 2.7.0

# Usage

## To Train 
* Download the [training dataset](https://github.com/kerenfu/LFSOD-Survey) and modify the 'train_data_path'.
* Start to train with
```sh
python -m torch.distributed.launch --nproc_per_node=4 train.py 
```

## To Test
* Download the [testing dataset](https://github.com/kerenfu/LFSOD-Survey) and have it in the 'dataset/test/' folder. 
* Download the already-trained [MEANet model](#trained-model-for-testing) and have it in the 'trained_weight/' folder.
* Change the `weight_name` in `test.py` to the model to be evaluated.
* Start to test with
```sh
python test.py  
```

# Download

## Trained model for testing
We released two versions of the trained model: 

Trained with additional 100 samples from HFUT-Lytro on [baidu pan](https://pan.baidu.com/s/1kd2ZjhwNcB4cEdGFwIUgUg?pwd=0o0r) with fetch code: 0o0r

Trained only with DUTLF-FS on [baidu pan](https://pan.baidu.com/s/1f_lBt1tebq9oQzIeknw9cg?pwd=75bn) with fetch code: 75bn

## Saliency map
We released two versions of the saliency map: 

Trained with additional 100 samples from HFUT-Lytro on [baidu pan](https://pan.baidu.com/s/1SR6wXKgpBfw9izsZlI4lXw?pwd=x7xa) with fetch code: x7xa

Trained only with DUTLF-FS on [baidu pan](https://pan.baidu.com/s/1luKlhBIXL0HdqxwbZZkgqg?pwd=s7vn) with fetch code: s7vn


# Citation
Please cite our paper if you find the work useful: 

    @article{JIANG202278,
    title = {MEANet: Multi-modal edge-aware network for light field salient object detection},
    journal = {Neurocomputing},
    volume = {491},
    pages = {78-90},
    year = {2022},
    author = {Yao Jiang and Wenbo Zhang and Keren Fu and Qijun Zhao}
    }
