# S-Extension Patch: A simple and efficient way to extend an object detection model 

There are three parts to this code.
1. Feature extraction
2. Compatible classes selection
3. Model training and inference

## Code usage

### Folder management for feature extraction
You need to add your images into a directory called __database/__, so it will look like this:

    ├── src/            # Source files
    ├── cache/          # Generated on runtime for feature extraction file
    ├── models/         # Containing all the model training files
    ├── README.md       # Intro to the repo
    └── database/       # Directory of all your images

__all your images should be put into database/__

In this directory, each image class should have its own directory and the images belonging to that class should put into that directory.

To get started with feature extraction, run the feature extraction code through ```python resnet.py``` after following the env steps and folder management as described there. 
Once you run the above code, visit the cache/ directory where you will find hte extracted features file. The same file will be used in the next step.

**Note that the extension class should also be appended in the same directory for its features to be compared with base classes. For example in COCO there are 80 base classes and I wish to add1 extension class then total 81 folders inside the database directory**
 
### Compatible classes selection

To find the compatible classes you can use the SMG class from the file ```simmat_threshold.py```. From it you can find the compatible classes by comparing the distance values with your set similarity threshold (in the paper it was 0.05).

### Training and inference

In the paper a ResNet152 model was trained from the pre-trained PyTorch fine-tuning pipeline. After training it on the compatible classes the inference could be run (on a single thread) by running the ```detect.py``` file. You need to do the path changes for source and classifier model trained. 

## Credits

1. Original dataset credits are to their respective authors.
2. The enitre credit for the detector scripts used goes to the original author of Yolov5 repository (configuration, trianing, base inference). Can be found [here](https://github.com/ultralytics/yolov5).
3. Feature extraction is based on the work of Po-Chih Huang's CBIR system based on ResNet features.

If you want to cite the entire work of S-Extension Patch: A simple and efficient way toextend an object detection model please make sure to include the full citiation as follows:

>@article{parikh2021s,  
>  title={S-Extension Patch: A simple and efficient way to extend an object detection model},  
>  author={Parikh, Dishant},  
>  journal={arXiv preprint arXiv:2110.02670},  
>  year={2021}  
>}  

## Author
Dishant Parikh | [DishantP](https://github.com/Dishant-P)

