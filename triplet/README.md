# Outline

## Determine which images are close and which are far

- For now, neighboring images are considered closer than far away images


## Collect triplet dataset

- Get list of files

- Have offset variable k=5

- Create positive pair from -k to k offset from original image

- Create negative pair from N/2-k to N/2+k (N is the size of the dataset)

- Split into test and train
