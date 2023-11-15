We extend LKU-Net with an Affine Head for the recent RnR-Exm 2023 Challenge.

The affine head is adopted from **TransMorph**.

The code and detailed description will be updated in the following weeks.


## Updating
- [x] Update the Model.py, from which one can directly call LKU-Net-Affine. April 18 2023.\
      ```
      LKU-Net-Affine = UNet(2, 8)
      ```

- [x] Updated the pre-trained model on the mouse dataset, [see here](https://drive.google.com/drive/folders/1PETLjQ7jV6jUmtvZnfvBlAt1Zl_21TWP?usp=sharing). Oct 11 2023.\
- [ ] Update the training and testing configurations.
- [ ] Update detailed descriptions about the LKU-Net-Affine.
