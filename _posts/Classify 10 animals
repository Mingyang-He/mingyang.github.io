# Classify 10 animals

This example aims to use Duck Duck Go to scrape sample images off the Web based on the fast.ai course
example on birds and design an appropriate multiclass loss function. 👽

**Step One: searching and downlowding images**
```
!pip install --upgrade duckduckgo_search

from duckduckgo_search import ddg_images
from fastcore.all import *

def search_images(term, max_images=200): return L(ddg_images(term, max_results=max_images)).itemgot('image')
```
📝

Install the duckduckgo search engine at the first.

```
from fastbook import *
from fastai.vision.widgets import *

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(search_images_ddg(term, max_images=max_images))
```
📝

Imports all of the fastai.vision and fastbook library.

define a function, **_search_images_**, which labels images based on a filename rule provided by the dataset creators.

```
searches = 'dog','bird','cat','goat','lizard','bull','snake','bear','fish','kangaroo'
path = Path('dog_or_not','bird_or_not','cat_or_not','goat_or_not','lizard_or_not',
           'bull_or_not','snake_or_not','bear_or_not','fish_or_not','kangaroo_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    resize_images(path/o, max_size=400, dest=path/o)
```
📝

Grab some examples of animal photos, and save each group of photos to a different folder.

**Step Two: train the model**
```
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
```
📝

Sometimes photos may not download correctly which could cause fail model training, so they have to be removed.

```
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)
```
📝

This part tells fastAI what kind of dataset we have and how it is structured. 
In fastAI we can create that easily using a DataBlock, and view sample images from it.

**_blocks_** cleassify our dataset. The inputs to our model are images, and the outputs are categories (in this case, "bird" or "cat").

To find all the inputs to our model, run the **_get_image_files_** function. it will return a list of all image files from that path.

The parameter **_valid_pct=0.2_** tells fastai to hold out **20%** of the data and not use it for training the model at all. This 20% of the data is called the **validation set**; the remaining **80%** is called the **training set**. The validation set is used to measurer the accuracy of the model. By default, the 20% is selected randomly.

The parameter **_seed=42_** sets the random seed to the same value every time we run this code, which means we get the same validation set every time we run it. 

Finally, **_resize_** the image to 192x192 pixels.

```
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```
📝

This part creates a convolutional neural network (CNN) and specifies what architecture to use, what data we want to train it on, and what metric to use.

**_resnet34_** refers to the number of layers in this variant of the architecture, and in this case we are using 18 layers vision model.

By using **_metrics=error_rate_**, it tells what percentage of images in the validation set are being classified incorrectly.

**_learn.fine_tune(3)_** tells fastAI how many times to look at each image.

After building and training the fastAI model in previous post, we can use this model to test and identify animals from our 10-animals-library. 👽

First of all, as an example, we can search for a random dog photo and seeing what result we can get.

```
urls = search_images('dog photos', max_images=1)
urls[0]

from fastdownload import download_url
dest = 'dog.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)
```

![dog_pic](http://www.publicdomainpictures.net/pictures/40000/velka/cute-dog-1362593345bn4.jpg)

Let's see what our model thinks about that image we downloaded. ❓

```
is_animal,_,probs = learn.predict(PILImage.create('dog.jpg'))
print(f"This is a: {is_animal}.")
print(f"Probability it's a dog: {probs[1].item():.4f}")
```

By running the code, it obtains the result that:

![](/images/10.png)

Similarly, we can try our model with other examples
```
download_url(search_images('bird photos', max_images=1)[0], 'bird.jpg', show_progress=False)
Image.open('bird.jpg').to_thumb(256,256)

download_url(search_images('stone photos', max_images=1)[0], 'stone.jpg', show_progress=False)
Image.open('stone.jpg').to_thumb(256,256)

is_animal,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_animal}.")
print(f"Probability it's a bird: {probs[1].item():.4f}")

is_animal,_,probs = learn.predict(PILImage.create('stone.jpg'))
print(f"Probability it's an animal : {probs[0].item():.4f}")
```

The result shows that a bird image can be identified successfully, while a stone, non-animal, image will be identified with low probability rate.

![](/images/11.png)


















