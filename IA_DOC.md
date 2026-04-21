# The Data
For the ai to work properly we need to train it on a [dataset]("https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false") . we will use the google quick draw dataset because it fits perfectly the project and provide numpy bitmap files that are already processed by google .

# The Training 
Because we don't have the google process power , we won't be able to use the full 50 million drawings provided by them ... Although we will use 10 distinct categories out of the 345 available . 

After selecting these categories we're going to merge them together into one big file . Then we'll use PyTorch as the machine learning framework .

To check that the ai is correctly training on the dataset and not only perfecting these exact drawings... We'll apply the 80/20 rule:

- **The Training Set (80%)**: The AI will look at these drawings and their labels to learn.

- **The Validation Set (20%)**: You hide these from the AI during learning.

The validation set will be used to check if the AI is able or not to recognize certain patterns and finally guess the drawing !  

# The Layering 

Basically , we'll tell the ai to analyse the drawing in multiple steps like layers :

- **Layer 1 (Convolution)**: The AI scans the 28x28 grid with a small magnifying glass, looking for basic shapes like straight lines or curves.

- **Layer 2 (Pooling)**: The AI shrinks the image slightly, keeping only the most important features it found in Layer 1. This drastically increases processing speed.

- **Layer 3 (Flatten)**: The grid is crushed into a single flat line of data.

- **Layer 4 (Dense Output)**: The final decision-maker. It outputs a list of probabilities (e.g., 90% chance it's an apple, 10% chance it's a circle).


# Repeating

Now that the process is in place we can start repeating it again and again and,  hopefully, we'll see the precision of the ai on the validation set increase . 

# Export 

Finally we'll save it's brain and plug it into the server loop  

## The execution

So after seelecting a set of 35 drawings saved into .npy files, we have to seperate them into the two set. To do this we use a simple program that extracts 1 000 drawing out of each numpy file and set them into an array. Then he shuffles them so we basically have 35 000 tuple as ({28x28 pixels}, dog) randomly set into another numpy set. So the x is the 784 pixels and the y is the label .

When we train with 5 repetitions we have a 66.74% precision so we go more.
With 10 repetition we have 69.59% precision.
With 20 repetition we have 68.64% precision.
With 40 repetition we have 65.59% precision. 

So here we can notice an issue where the more we repeat the less i have precision .. That's due to a lack of calculation power. Here's the explication :

Actually my computer can't run the training when i put 10 000 images per category , so i droped it to 1 000. But this means that the ai trains on way less data but way more. That means that when we test the 20% validation set the ai only knows an apple if it is in the 80% training set. 

so now let's take more data like 5 000 images per category and see what happens:
With 5 repetition we have 73.23% precision
With 10 repetition we have 73.97% precision
With 20 repetition we have 74.00% precision

I'm not going further in this test but as we can see the precision goes and dropes so the optimal number of repetition is 10 . 

Now we have to increase this number by using a better computer . 

