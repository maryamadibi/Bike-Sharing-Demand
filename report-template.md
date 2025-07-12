# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Maryam Adibi

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
That I needed to change the format. I had to merge the datetime column from the test set with the predicted values and save the result in the correct format.Kaggle refuses the submissions containing negative predictions values so I changed negative values with 0. 

### What was the top ranked model that performed?
The best performing model was from the add_features stage, called WeightedEnsemble_L3. It achieved:
Validation RMSE: ~37.98
Best Kaggle score: 0.44798
Interestingly, even though hyperparameter tuning (HPO) helped improve some model metrics, this ensemble model (with good feature engineering but without HPO) performed best on the test dataset.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
During exploratory data analysis (EDA), I made several improvements to the dataset to help the model better understand the patterns in bike rental demand:
The datetime column was originally a single string. I converted it into a proper date-time format and extracted useful time-based features like hour, day of the week, month, and year. Once these were created, the original datetime column was removed.
The season and weather columns were stored as numbers, but these values represent categories (like spring, summer, etc.). I converted them to categorical variables so the model could treat them properly.
The columns casual and registered were found to be highly correlated with the target variable (count). However, these columns were only present in the training set and not available in the test set. Since we can't use them for prediction, I removed them.
I created a new feature called day_type by combining the holiday and workingday columns. This feature groups each day into one of three types: weekday, weekend, or holiday. This helps the model understand demand changes based on the type of day.
The features temp (temperature) and atemp (feels-like temperature) were almost identical — they had a 98% correlation. To avoid confusion and redundancy, I dropped atemp.
In addition, I used visualizations to look at patterns in demand over time, such as changes by hour, season, or weekends, which helped guide the feature engineering process.

### How much better did your model preform after adding additional features and why do you think that is?
After adding new features and cleaning the dataset, the model's RMSE improved from ~55.03 to ~34.38 — that's about a 138% improvement!
This helped because of:
1. Extracted time-related patterns like rush hours.
2. Cleaned up variables that could confuse the model.
3. Converted features to the correct data types.
4. Reduced multicollinearity by dropping duplicate-like columns.
These steps helped the model better understand real-world patterns in the data.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Hyperparameter tuning contributed to model improvement over the initial training run. Three different HPO configurations were tested. The best one (HPO2) achieved a Kaggle score of 0.49440, which was an improvement over the initial model (1.83835), but still not as strong as the model trained after feature engineering (0.44798).
Observations:
Hyperparameter tuning was performed using the AutoGluon TabularPredictor.fit() function with different presets and model types.
Although tuning helped optimize the learning process, the improvements were limited due to the prescribed hyperparameter ranges, which restricted AutoGluon's ability to explore the full search space.
The parameter presets='optimize_for_deployment' enabled faster and more reliable experiments, especially when hardware limitations made high_quality configurations impractical.
Some HPO trials failed entirely when the time_limit was too short or memory usage exceeded system resources — particularly with "high_quality" and auto_stack=True.
Choosing lighter presets allowed models to train successfully within time and memory constraints, but came at the cost of potentially lower accuracy.
verall, the effectiveness of HPO in this project was highly dependent on the choice of model types, the time budget, and resource availability.
The trade-off between exploration (broad search) and exploitation (fine-tuning known good values) 

### If you were given more time with this dataset, where do you think you would spend more time?
I would:
Run AutoGluon with longer time limits and "high_quality" presets.
Try other feature interactions or seasonal adjustments.
Explore time-series-specific models, if supported.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
| Model         | HPO1                                | HPO2               | HPO3                                        | Kaggle Score |
| ------------- | ----------------------------------- | ------------------ | ------------------------------------------- | ------------ |
| initial       | prescribed\_values                  | prescribed\_values | presets: 'high\_quality' (auto\_stack=True) | 1.83835      |
| add\_features | prescribed\_values                  | prescribed\_values | presets: 'high\_quality' (auto\_stack=True) | 0.44798      |
| hpo           | Tree-Based Models: GBM, XT, XGB, RF | KNN                | presets: 'optimize\_for\_deployment'        | 0.49440      |


### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](img/Figure_1.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.


![model_test_score.png](img/Figure_2.png)

## Summary
This project used AutoGluon, a powerful AutoML tool, to build models for predicting bike-sharing demand. Here’s what I learned:
AutoGluon makes it easy to try multiple models with minimal code.
Careful feature engineering often outperforms basic hyperparameter tuning.
Understanding your data and fixing problems like data type mismatches or highly correlated variables can significantly improve results.
HPO can help but depends a lot on time limits, presets, and available resources.
AutoGluon handles a lot automatically but tuning and experimentation are still key for better results.
In short, combining AutoML tools with thoughtful feature engineering produced the best performance in this task.

