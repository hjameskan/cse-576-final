# Training Deep Neural Networks on Occluded Datasets

## Overview

Convolutional neural networks (CNNs) trained for object recognition often fail to classify images that have been distorted in ways that weren't present in the training set. This is partially because CNNs learn very specific features of each class. For example, a CNN may learn that a person is defined by having a face. But in an image where the face is occluded an object in the foreground, then the CNN may not recognize it as a person. In contrast, the human visual system can use a variety of cues to recognize objects despite occlusion. How might this same capability be incorporated into CNNs? We hypothesize that this can be accomplished by training a CNN with a modified image data set that includes occluded images. In doing so, the CNN will be forced to learn a wider variety of features as no single feature can reliably describe an object. The results of this work may act a powerful framework for training CNNs to function robustly in realistic environments, where occluded objects are common.

## Team Members

### Luke Bun

Luke has experience generating artificial stimuli for psychophysical and electrophysiolgical neuroscience experiments. He will use these skills to help generate occluded stimuli.

### Shao-Jung Kan

Shao-Jung has experience in network and systems development. He is most interested in exploring the industry trends of machine learning.

### Cameron McCarty

Cameron is most interested in the implication occlusion has on explainable AI or XAI. Many XAI methods rely on feature removal but suffer from occluded images being outside of their scope. Cameron will bring previous experience with XAI occlusion and implementing image classifiers.

## Project Goals

It is our goal to improve the accuracy of a CNN model on occluded images by training selection. We will implement a standard multi-class object classifier that can perform well under occlusion compared to a model of the same architecture without the occlusion training. Our final test is running them side by side on the same holdout set to determine the accuracy gain.

## Milestones

### Planning Specifics

The first step is to come to a consensus over basic details. These include data set, architecture, occlusion method, and optimiser.

### Basic Classifier

We will start our implementation with our control model that we will compare our model with, this will be include an initial hyper-parameter search to optimise accuracy on the base data set.

### Occlusion Augmentation

Next we will train another copy of this classifier with a modified set of data. We have a few different options from exhaustive occlusion over all parts of an image or implementing a randomized set of occlusions to be performed on a given training image. Since this data set will be larger than the set used on the basic this step may take some time.

### Hyper-parameter Search

After training an initial version of each model, we will search over hyper-parameters to get the optimal model for each. This will take the least amount of work, but the most amount of run time.

### Testing and Analysis

Our final step is to test our models on a holdout occluded testing set and compile our findings and draw conclusions.
