## AI-for-Web-Accessibility


This is the GitHub repository for my Masters dissertation titled: **Artificial Intelligence for Web Accessibility** which I completed as a part of my MSc in
Data Science course in the University of Southampton, UK under the supervision of Prof. Mike Wald

This project provided me an opportunity to apply my knowledge in machine Learning and Deep Learning to a problem that impacts people's lives.

The project mainly focuses on applying AI technologies to make the web more accessible to people who are differently abled. We take the knowledge and information
available on the internet for granted; but not everyone is so fortunate. This project is my modest attempt to work on this problem. There is a lot of scope to extend
this work and please feel free to contact me if you have any ideas/queries/suggestions!

LinkedIn: [Shaunak Sen](https://www.linkedin.com/in/shaunak-sen/)
Email: shaunak1105@gmailcom

The project mainly focuses on two parts:

1. Automatic Image Captioning System
2. Contextual Hyperlink Detection

This document only provides an overview. For details please refer to the full report [here](link)

### Automatic Image Captioning System

#### The Problem

The World Wide Web Consortium (W3C) is an organization responsible for developing and maintaining web standards such as HTML, CSS, etc. (57). The Web Content Accessibility Guidelines (WCAG) is developed through the W3C process and it aims to create and maintain a single set of guidelines and recommendations for individuals, organizations, and governments internationally to follow to make web content more accessible and inclusive, especially for people with disabilities (58; 64).

Guideline H37 (56) of the WCAG focuses on the proper use of alternative (alt) texts for images to help visually impaired people understand the message the image is trying to convey. Often developers fail to provide the above-mentioned alt texts and even if they do, the text does not really convey the message of the image.
Automatic Image captioning is a challenging task because it combines the workings of both CNNs and RNNs together. The CNN must understand the high-level features of the image and the RNN must translate these features into relevant captions

#### The Dataset

There are several options for a dataset of images accompanied by their corresponding captions. Some of these are Flickr8k (17), Microsoft COCO: Common objects in context (MSCOCO) (27), Conceptual captions dataset by Google (44). I have used the Flickr8k dataset for this task


#### Data Cleaning and pre-processing

In this task, we are dealing with both image data as well as textual data, which has been crowdsourced (17). Data cleaning and preprocessing is very important for the performance of the deep learning model.
The pre-processing steps vary for images and text. The pre-processing for the images involves:

1. Resize the images to dimensions: 224x224x3 - 224 is the image height and width (in pixels). 3 denotes the number of color channels (RGB)
2. Normalize the images by mean centering them. The mean RGB value was subtracted from each pixel value of the image

For natural language processing based tasks it is a good practice to clean text data and create a text cleaning pipeline using tools like python nltk (2; 6). The steps in the text cleaning pipeline suitable for our task include:

1. Tokenize the captions into separate words
2. Case Normalization - Convert all words to lowercase
3. Remove punctuations from the words
4. Remove non-alphanumeric characters from the words

For more traditional machine learning tasks additional steps like stemming and lemmatization need to be carried out, but because our model is going to have an embedding layer, it does not make sense to perform additional preprocessing (6).

#### Model for Image Classiﬁcation





