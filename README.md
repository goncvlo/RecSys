# RecSys | MovieLens Recommender System

**Overview**

RecSys is a collection of recommendation algorithms applied to the MovieLens ml-100k dataset. It demonstrates various collaborative filtering and machine learning techniques to provide personalized movie suggestions.

**Features**
- Multiple algorithms: User-based and item-based collaborative filtering, Matrix Factorization, etc.
- Detailed evaluation using metrics like RMSE and Precision@K.
- Jupyter notebooks for step-by-step model training and inference.

**Usage**
- Data Preparation: Run the data preprocessing script to prepare the MovieLens dataset.
- Model Training: Execute the train_model.ipynb notebook to train various models.
- Evaluation: Check best model performance using the evaluate_model.ipynb notebook.
- Inference: Use inference.ipynb for generating movie recommendations.

**Algorithms**
- Collaborative Filtering: User-based and item-based approaches.
- Matrix Factorization: Techniques like SVD.
- Baseline Models: Compare with simple models such as average rating predictors.

**Future Work**
- Incorporate deep learning models like Autoencoders.
- Improve scalability for larger datasets.
- Implement a web interface for real-time recommendations.

**References**
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Surprise](https://surprise.readthedocs.io/en/stable/)
