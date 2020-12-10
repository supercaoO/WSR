# Description of Test Options

Let us take [`test_WSR_Set5.json`](./test_WSR_Set5.json) as an example. 

**Note**: Before you run `python test.py -opt options/test/*.json`, please carefully check options: `"scale"`, `"degradation"`,  `"self_ensemble"`, `"dataroot_HR"`, `"dataroot_LR"`, `"networks"` and `"pretrained_path"`.

```c++
{
    "mode": "sr", // solver type (only "sr" is provided)
    "gpu_ids": [0], // GPU ID to use

    "scale": 4, // super resolution scale (*Please carefully check it*)
    "degradation": "BI", // degradation model for SR: "BI" | "BD" | "DN" (*Please carefully check it*)
    "is_train": false, // whether train the model
    "use_chop": true, // whether enable memory-efficient test
    "rgb_range": 255, // maximum value of images
    "self_ensemble": false, // whether use self-ensemble strategy
    
    // test dataset specifications (you can place more than one test dataset here) (*Please carefully check dateset mode/root*)
    "datasets": { 
        "test_set1": {
            "mode": "LRHR", // dataset mode: "LRHR" | "LR"
            "dataroot_HR": "./results/HR/Set5/x4", // HR dataset root (required by "LRHR" dataset mode) 
            "dataroot_LR": "./results/LR/LRBI/Set5/x4", // LR dataset root (required by "LRHR"/"LR" dataset mode) 
            "data_type": "img" // data type: "img" (image files) | "npy" (binary files), "npy" is recommended during training
        }
    },
    
    // networks specifications
    "networks": { 
        "which_model": "WSR", // network name
        "num_features": 64, // number of base feature maps
        "in_channels": 3, // number of input channels
        "out_channels": 3, // number of output channels
        "num_groups": 10 // number of S-IMDBs (N)
    },
    
    "solver": {
        "pretrained_path": "./models/WSR_x4_BI.pth" // pre-trained model directory (for test)
    }
}
```
