# Pre-processing

## Image Pre-processing

We used min max normalization for all scans.
The so-called min max normalization can be formulated as:

precessed_image = image.clip(min, max)
precessed_image= (precessed_image - min)/(max - min)

where min and max are some predefined values, for example min can be -1000 and max can be 500.


In the implementation, as shown in Functions.py, we used 6 different combinations of min and max values to **augment** the training data.

frame = np.random.choice(['imagesTr_400.0_-1000.0', 'imagesTr_450.0_-1000.0', 'imagesTr_500.0_-1000.0','imagesTr_400.0_-900.0', 'imagesTr_450.0_-900.0', 'imagesTr_500.0_-900.0'])

## Key points Pre-processing

We understand there are multiple methods that use keypoints as supervision for network training, we however convert the keypoints into a mask image.
Specifically, to utilize the provided landmark / keypoints，we convert each keypoint into a  3x3x3 patch （intensity ==1）, while the background has zero intensity.
To this end, we have a HxWxD gray image with many 3x3x3 white patches.

**Why mask over key points?**

Because key points are labeled by human that are subjective or by classical methods that are not 100% correct, rounding each key point into a 3x3x patch provides some tolerance of the labeling/uncertainty.

## Mask Pre-processing

We combine the provided Mask and the generated key points  mask from the last step, we then get a new mask image, where the intensity of the lung region is 1 and that of key points patch is 2.

# Training

We use start channel == 16/32, using_l2 == 2, data:mask:smth= 1.0:1.0:4.0, iteration=100001

# Testing

We submit the model that achieved highest validation Dice score.

