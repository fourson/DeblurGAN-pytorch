# DeblurGAN

An easy-to-read implementation of [DeblurGAN](https://arxiv.org/pdf/1711.07064.pdf) using PyTorch  

## Some demos of deblurring:  
- ![blurred1](demo_pic/1.png) ![deblurred1](demo_pic/deblurred1.png)
  
- ![blurred2](demo_pic/2.png) ![deblurred2](demo_pic/deblurred2.png) 

## Prerequisites
- NVIDIA GPU + CUDA cuDNN
- Python 3.7

## Folder Structure
  ```
  deblurGAN/
  │
  ├── deblur_image.py - deblur your own images
  ├── test.py - evaluation of trained model
  ├── train.py - main script to start training
  ├── make_aligned_data.py - make aligned data
  ├── config.json - demo config file
  ├── config_aligned.json - demo config file using aligned dataset
  ├── pretrained_weights/ - some pretrained weights for test
  │   ├── GAN/ - folder of pretrained weights using GAN loss
  │   └── WGAN_GP/ - folder of pretrained weights using WGAN_GP loss
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py - abstract base class for data loaders
  │   ├── base_model.py - abstract base class for models
  │   └── base_trainer.py - abstract base class for trainers
  │
  ├── data_loader/ - dataloader and dataset
  │   ├── data_loader.py
  |   └── dataset.py 
  │
  ├── data/ - default directory for storing input data, containing 2 directory for blurred and sharp
  │   ├── blurred/ - directory for blurred images
  │   └── sharp/ - directory for sharp images
  │
  ├── model/ - models, losses, and metrics
  │   ├── layer_utils.py
  │   ├── loss.py
  │   ├── metric.py
  │   └── model.py
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  └── utils/
      ├── logger.py - class for train logging
      ├── util.py
      ├── visualization.py - class for tensorboardX visualization support
      └── ...
  ```

## Config file format
```
{
    "name": "DeblurGAN",                         // training session name
    "n_gpu": 1,                                  // number of GPUs to use for training
    "data_loader": {                             // selecting data loader
        "type": "GoProDataLoader",
        "args": {
            "data_dir": "data/",
            "batch_size": 1,
            "shuffle": false,
            "validation_split": 0.1,
            "num_workers": 4
        }
    },
    "generator": {                               // architecture of generator
        "type": "ResNetGenerator",
        "args": {
            "input_nc": 3,
            "output_nc": 3
        }
    },
    "discriminator": {                           // architecture of discriminator
        "type": "NLayerDiscriminator",
        "args": {
            "input_nc": 3
        }
    },
    "loss": {                                    // loss function
        "adversarial": "wgan_gp_loss",
        "content": "perceptual_loss"
    },
    "metrics": [                                 // list of metrics to evaluate 
        "PSNR"
    ],
    "optimizer": {                               // configuration of the optimizer (both generator and discriminator)
        "type": "Adam",
        "args": {
            "lr": 0.0001,
            "betas": [
                0.5,
                0.999
            ],
            "weight_decay": 0,
            "amsgrad": true
        }
    },
    "lr_scheduler": {                            // learning rate scheduler
        "type": "LambdaLR",
        "args": {
            "lr_lambda": "origin_lr_scheduler"
        }
    },
    "trainer": {                                 // configuration of the trainer
        "epochs": 300,
        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 2,
        "monitor": "max PSNR",
        "tensorboardX": true,
        "log_dir": "saved/runs"
    },
    "others": {                                  // other hyperparameters
        "gp_lambda": 10,
        "content_loss_lambda": 100
    }
}
```

## How to run
* **Train**
 ```
    python train.py --config config.json
 ```

* **Resume**
```
    python train.py --resume path/to/checkpoint
```

* **Test**
```
    python test.py --resume path/to/checkpoint
```

* **Deblur**
```
    python deblur_image.py --blurred path/to/blurred_images --deblurred path/to/deblurred_images --resume path/to/checkpoint
```

* **Make aligned data first if you want to use aligned dataset**
```
    python make_aligned_data.py --blurred path/to/blurred_images --sharp path/to/sharp_images --aligned path/to/aligned_images
```

## Tips
- If you want to use gan_loss instead of wgan_gp_loss, use_sigmoid must be set to true in generator.
- Aligned dataset could boost the speed of data_loader a little bit. So run make_aligned_data.py to get aligned dataset before training.
- Pretrained weights of both GAN and WGAN_GP are available.
- **Download pretrained weights： https://drive.google.com/open?id=1w-u0r3hd3cfzSjFuvvuYAs9wA-E-B-11**

## Acknowledgements
The organization of this project is based on [PyTorch Template Project](https://github.com/victoresque/pytorch-template)