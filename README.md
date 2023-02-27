# ⭐ Detecting Inconsistent Amazon Reviews

In this project, we built a **Machine Learning model** that can detect when a review expresses a certain **sentiment** in the title or the body, but the **rating is inconsistent**.
In the following picture we can see an example of what we will refer to as **inconsistent review**:

<div align="center">

![epson-cinema-2150](https://user-images.githubusercontent.com/23276420/220408482-c6e1209b-5492-4784-be70-14415d16ef78.png)

</div>

In this case, we can see that the Anonymous reviewer thinks that the projector is not that great, since it has lots of problems, but he still rated it 4 stars, even if he asked for a refund.

This system could be **very useful for an e-commerce company**, because reviews are really important for customers, and having good quality reviews helps both
the customer to make more informed purchase decisions, but also helps the company to improve their quality of services.

In order to train the model, we used a [dataset of Amazon reviews](https://nijianmo.github.io/amazon/index.html) and in particular a subset of **1 million reviews** about **electronics products**.

For the reviews we classified them in the following way:
- Reviews with 1 or 2 stars are **negative**
- Reviews with 3 stars are **neutral**
- Reviews with 4 or 5 stars are **positive**

For the classification we considered different approaches:
- **Text-free approach**: baseline approach where we didn't consider text, but we only leveraged the length
of the content and all the other features offered by the dataset mentioned.
  This approach has been used with the following models:
  - Dummy Classifier
  - Logistic Regression
  - AdaBoost
  - Linear SVM
  - Random Forest
  - MLP
- **Bag-of-Words approach**: the text is turned into vectors by using both the Bag-of-Words representation and the TF-IDF Weighting scheme.
  For both of them we trained the following models both using all the features and also using text features only:
  - Random Forest
  - MLP
- **Embedding approach**: each word is turned into a vector by using a [pretrained GloVe model](https://nlp.stanford.edu/projects/glove/).
  For this approach the following models have been compared:
  - Convolutional Neural Network (CNN)
  - Long-Short Term Memory (LSTM)

For the text approaches, the title and the content of the reviews were concatened and processed like a single paragraph.

The whole code of the project is available in the notebook [`Inconsistent Reviews.ipynb`](https://github.com/SkyLionx/inconsistent-reviews/blob/main/Inconsistent%20Reviews.ipynb) contained in this repo but it is best viewed on the Colab platform.

<a href="https://colab.research.google.com/github/SkyLionx/inconsistent-reviews/blob/main/Inconsistent%20Reviews.ipynb" target="_blank">
<img src="https://img.shields.io/badge/Colab-Open%20Notebook-green?style=for-the-badge&logo=googlecolab&color=blue">
</a>
<br /><br />

Moreover, a detailed report with the pipeline of the models and evaluation is available in the file named [`Report.pdf`](https://github.com/SkyLionx/inconsistent-reviews/blob/main/Report.pdf).

## Results

In this section we report the results that we obtained with each approach.
We first evaluated the models on the validation set and then measured the performance of the best models on the test set.

### Text-free approaches

In this case we don't have many discriminative features for the task and so we can't achieve very high results on the validation set.

| Model name              | Accuracy Train Set  | Accuracy Validate Set |
|-------------------------|---------------------|-----------------------|
| Dummy Classifier        | 0.332               | 0.333                 |
| Logistic Regression     | 0.402               | 0.406                 |
| AdaBoost                | 0.423               | 0.428                 |
| Linear  SVM             | 0.401               | 0.402                 |
| **Random Forest**       | **0.915**           | **0.450**             |
| MLP                     | 0.427               | 0.425                 |


### Bag-of-Word approaches

As we can see, adding text features greatly helps to achieve higher accuracy.

| Model name    | Features                    | Accuracy Train Set | Accuracy Validate Set |
|---------------|-----------------------------|--------------------|-----------------------|
| Random Forest | BoW                         | 0.863              | 0.715                 |
| MLP           | BoW                         | 0.799              | 0.708                 |
| Random Forest | BoW + text-free features    | 0.878              | 0.714                 |
| MLP           | BoW + text-free features    | 0.732              | 0.719                 |
| Random Forest | Tf-Idf                      | 0.913              | **0.727**             |
| MLP           | Tf-Idf                      | 0.800              | 0.716                 |
| Random Forest | Tf-Idf + text-free features | **0.919**          | 0.724                 |
| MLP           | Tf-Idf + text-free features | 0.726              | 0.720                 |

### Embeddings approaches

| Model name | Accuracy Train Set | Accuracy Validate Set |
|------------|--------------------|-----------------------|
| CNN        | **0.87**           | 0.80                  |
| LSTM       | 0.84               | **0.81**              |

Using embeddings, the CNN model is performing pretty close to the LSTM on the validation set.

### Final comparison on Test Set

| Approach      | Best model             | Test Accuracy |
|---------------|------------------------|---------------|
| Text-free     | Random Forest          | 0.45          |
| BoW           | Random Forest (Tf-Idf) | 0.73          |
| **Embedding** | **LSTM**               | **0.81**      |

On the test set we are able to obtain a satisfactory accuracy of 81% using Embeddings and an LSTM architecture.

## Contributors

<a href="https://github.com/SkyLionx" target="_blank">
  <img src="https://img.shields.io/badge/Profile-Fabrizio%20Rossi-green?style=for-the-badge&logo=github&labelColor=blue&color=white">
</a>
<br /><br />
<a href="https://github.com/dotmat3" target="_blank">
  <img src="https://img.shields.io/badge/Profile-Matteo%20Orsini-green?style=for-the-badge&logo=github&labelColor=blue&color=white">
</a>
<br /><br />
<a href="https://github.com/ErVincit" target="_blank">
  <img src="https://img.shields.io/badge/Profile-Emanuele%20Vincitorio-green?style=for-the-badge&logo=github&labelColor=blue&color=white">
</a>

## Technologies

In this project the following Python libraries were adopted:
- TensorFlow and Scikit-learn for machine learning
- Numpy
- Matplotlib and Seaborn for plotting
- Pandas for data handling
