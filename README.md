# ML_Project58-TensorFlowCollabarativeFilteringModel

### Tensorflow Collaborative Filtering Model

#### Introduction:
This project implements a collaborative filtering model using TensorFlow. Collaborative filtering focuses on finding similarities between users based on their ratings for items (in this case, books) and recommends items to users based on the ratings of similar users.
#### Data:
The project uses three main datasets:
```
BX-Book-Ratings.csv: Contains user ratings for books.
BX-Users.csv: Contains user information.
BX-Books.csv: Contains book information.
```

##### Preprocessing:

The code reads the datasets and merges them based on the ISBN (International Standard Book Number) to create a single dataframe book_rating.

Unnecessary columns like year of publication, publisher, author, and image URLs are dropped from the book_rating dataframe.

##### Data Exploration:
The code displays the first few rows of the book_rating dataframe to show the structure of the data.

It also shows the number of ratings each book has received using a groupby operation.

Ratings with a count below a certain threshold (25 in this case) are filtered out to reduce noise.

Similarly, users with a low number of ratings (below a threshold, 20 in this case) are filtered out.

##### Data Normalization:
The ratings are normalized using MinMaxScaler from sklearn.preprocessing.

The normalized ratings are then used to build the user-book matrix, where rows represent users, columns represent books, and entries represent normalized ratings.

##### Model Architecture:
TensorFlow v1 compatibility mode is enabled since the code uses tf.placeholder.

The encoder-decoder architecture for the collaborative filtering model is defined, consisting of two hidden layers for both encoder and decoder.

The encoder function takes the input features and passes them through the hidden layers, applying sigmoid activation functions.

Similarly, the decoder function reconstructs the input features from the encoded representation.

##### Model Training:
Placeholders for input data and evaluation metrics are defined.

Loss function, optimizer, and evaluation metrics (mean squared error and precision) are defined.

TensorFlow variables are initialized.

The model is trained for a specified number of epochs using mini-batch gradient descent.

After training, the model predicts ratings for all user-book pairs in the dataset.

##### Generating Recommendations:
Predicted ratings are converted back to the original scale and stored in a dataframe.

Duplicate entries are removed, and the top-10 recommendations are generated for each user based on predicted ratings.

### How to Run
1. **Data Preparation:**
   - Ensure the book ratings dataset (BX-Book-Ratings.csv) and user/book information datasets (BX-Users.csv, BX-Books.csv) are available in the specified format.
2. **Project Setup:**
   - Install the required libraries and dependencies, including TensorFlow and scikit-learn.
3. **Data Preprocessing:**
   - Execute the data preprocessing script to load, clean, and filter the dataset.
4. **Model Training:**
   - Run the TensorFlow collaborative filtering model script to train the model on the preprocessed data.
5. **Model Evaluation:**
   - Analyze the model performance metrics and generated recommendations to assess the effectiveness of the collaborative filtering approach.

### Results and Analysis
- Evaluate the model's ability to recommend books by comparing predicted ratings with actual user ratings.
- Analyze the impact of different hyperparameters and model architectures on recommendation quality.
- Explore potential enhancements or extensions to improve recommendation accuracy or scalability.


### Acknowledgements
- This project utilizes the Book-Crossing dataset for book ratings and additional user/book information.
- Special thanks to the TensorFlow library for providing tools and resources for building and training deep learning models.
- Credits to the scikit-learn library for providing utilities for data preprocessing and evaluation.
