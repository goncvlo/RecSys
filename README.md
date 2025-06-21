# RecSys | MovieLens Recommender System

RecSys is a collection of recommendation algorithms applied to the MovieLens ml-100k dataset. It demonstrates various collaborative filtering and machine learning techniques to provide personalized movie suggestions.


### :nut_and_bolt: Features
- Multiple algorithms: User-based and item-based collaborative filtering, Matrix Factorization, etc.
- Detailed evaluation using metrics like RMSE and FCP.
- Jupyter notebooks for step-by-step model training and inference.

### :open_file_folder: Usage
- Data Preparation: Run the data preprocessing script to prepare the MovieLens dataset.
- Model Training: Execute the train_model.ipynb notebook to train various models.
- Evaluation: Check best model performance using the evaluate_model.ipynb notebook.
- Inference: Use inference.ipynb for generating movie recommendations.

### :scroll: Algorithms
- Collaborative Filtering: User-based and item-based approaches.
- Matrix Factorization: Techniques like SVD.
- Baseline Models: Compare with simple models such as average rating predictors.

### :test_tube: Installation
To install the RecSys repository, begin by creating and activating a virtual environment to isolate your project’s dependencies. Next, clone the repository from GitHub and navigate to the project directory. Finally, install the required packages specified in the requirements.txt file. This setup ensures you have a clean environment tailored for running the RecSys project effectively.
```
>> python -m venv recsys-env
>> recsys-env\Scripts\activate
```
```
>> git clone https://github.com/goncvlo/RecSys.git
>> cd RecSys
>> pip install -r requirements.txt
```

### :hourglass_flowing_sand: Future Work
- Incorporate deep learning models like Autoencoders.
- Improve scalability for larger datasets.
- Implement a web interface for real-time recommendations.

### :raising_hand: References
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Surprise](https://surprise.readthedocs.io/en/stable/)
