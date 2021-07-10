# WordNetSim
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

Python module implementing a semantic similarity metric via WordNet.

## Tools and libraries 🛠️

- [**NLTK**](http://nltk.org/): As a WordNet interface.
- [**FreeLing 4.2**](http://nlp.lsi.upc.edu/freeling/node/1): For tokenizing, sentence splitting and lemmatization. Also needed the **Python API (pyFreeLing)**.
- [**FastText 0.9.2**](https://fasttext.cc/docs/en/support.html): For language detection.
- [Numpy 1.19.5](https://www.numpy.org): For basic statistical functions.

## Setup 👨‍💻

Main dependencies for this module to work are Freeling and NLTK.

### Installing Freeling

To use Freeling **API for Python** (instead of native C++) we need to compile it from source as stated in their [**documentation**](https://freeling-user-manual.readthedocs.io/en/v4.2/toc/). I deeply encourage the reader to follow the installation steps described there since it is not a trivial process.

### NLTK Dependencies

In some versions of NLTK the WordNet corpus might not be pre-installed, so you would need to install it, if not installed the program should crash as soon as the first line:

```
from nltk.corpus import wordnet as wn
```
Depending on the language you are using, you may need to add a file with a list of stopwords to the installation folder of NLTK, this folder will usually be: 
```
/home/user/nltk_data/corpora/stopwords/
```

And the structure of the file is simply the name of the language (eg: catalan) and one stopword per line. As an example, you can take a look at the [catalan stopwords](./catalan) file.

Also, you may want to use WordNets that are not part of the [OMW](http://compling.hss.ntu.edu.sg/omw/) (the ones included in NLTK) such as [Mongolian WordNet](https://github.com/kbatsuren/monwn). For this, you only have to add the folder cointaining the .tab (if it is .tsv, change extension to .tab) file in the **omw** folder inside your NLTK data folder. Usual path for this is:

```
/home/user/nltk_data/corpora/omw/
```

You can find an example of how these folders are structured on [the Mongolian WordNet example](./mon).

## Some enchancements 🆙

As you might notice, some of the steps are slow. These are mainly finding the synsets corresponding to each word and computing distance between those. To solve this, the use of memoization techniques, such as precomputing the values and saving them on sets or any other fast-accessing data structure are good solutions, however the idea of this repo was offering the basic features.

Also, some overhead seems to happen when we launch the program. It seems to be a problem related to FreeLing, however I have still not found a solution for it, so suggestions are welcome.
## Author ✒️

* **Jose Francisco Domenech Gomis** ([dogo3](https://github.com/dogo3))