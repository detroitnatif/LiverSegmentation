# LiverSegmentation
Using PyTorch and MonAI to perform image segmentation

<div align="center">
  <h1>LiverSegmentation</h1>
  <p>Using PyTorch and MonAI to perform image segmentation</p>
  <img src="img/Visualization_Train2.png" alt="Description of the image" width="700" height="400">
</div>

# Data 
<blockquote>
  <p>Antonelli, M., Reinke, A., Bakas, S. et al. The Medical Segmentation Decathlon. Nat Commun 13, 4128 (2022). https://doi.org/10.1038/s41467-022-30695-9 </p>
</blockquote>

Starting with this pre-labled dataset, comprised of 200 patient CAT scans each containing 128 slices, I seperated the data into training and testing splits. From here I preprocessed the images by combining all 128 slices into a compresssed .nii file, and performed the necessary transformations.  
```
train_transforms = Compose(
    [
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstd(keys=["vol", "seg"]),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
        CropForegroundd(keys=["vol", "seg"], source_key="vol"),
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),   
        ToTensord(keys=["vol", "seg"])
    ]
)
```

 

## Model Architecture 


