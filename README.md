# CS 486 Project

Our project is a recommendation engine for movies where instead of recommending movies that users similar to you enjoyed (like most engines), we recommend movies that users DISsimilar to you enjoyed. We found that this actually gave us better results. We used the Netflix Prize data from Kaggle that can be found here: https://www.kaggle.com/netflix-inc/netflix-prize-data



Some of the features include giving recommendations based on dissimilar users, recommendations based on similar users to compare to, displaying the average score of the movies from dissimilar users and similar users for their respective recommendations, and displaying the average score for each of the movies recommended.

## How to get started
- Install Python 3.7+
    - Python 3.7 is the newest version available on `student.cs.uwaterloo.ca`
    - Just make sure we're maintaining compatibility
- [Optional] Create a virtual environment
    - This prevents you from upgrading system packages by accident
      (this is more of an issue on Linux, but still a best practice)
- Install the necessary packages for running our project
    - `pip3 install -r requirements.txt`
- Run the following command
    - `python3 d3.py`
