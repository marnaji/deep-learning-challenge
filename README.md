# deep-learning-challenge
## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* EIN and NAME: Identification columns
* APPLICATION_TYPE: Alphabet Soup application type
* AFFILIATION: Affiliated sector of industry
* CLASSIFICATION: Government organization classification
* USE_CASE: Use case for funding
* ORGANIZATION: Organization type
* STATUS: Active status
* INCOME_AMT: Income classification
* SPECIAL_CONSIDERATIONS: Special considerations for application
* ASK_AMT: Funding amount requested
* IS_SUCCESSFUL: Was the money used effectively

## Instructions
#### Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
   * What variable(s) are the target(s) for your model?
   *  What variable(s) are the feature(s) for your model?
2. Drop the EIN and NAME columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

6. Use pd.get_dummies() to encode categorical variables.

7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

#### Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Create the first hidden layer and choose an appropriate activation function.

4. If necessary, add a second hidden layer with an appropriate activation function.

5. Create an output layer with an appropriate activation function.

6. Check the structure of the model.

7. Compile and train the model.

8. Create a callback that saves the model's weights every five epochs.

9. Evaluate the model using the test data to determine the loss and accuracy.

10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

#### Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.
1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

#### Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

# Report 
### Overview: 
   The nonprofit foundation, Alphabet Soup, is looking to improve its grant allocation process by harnessing the power of machine learning and neural networks. My role was to create a binary classifier using the provided dataset to help identify potential grant recipients with the highest likelihood of success when supported by Alphabet Soup.
   Working alongside Alphabet Soup's business team, I have been given access to a CSV file containing data on more than 34,000 organizations that have previously been recipients of Alphabet Soup's funding. This dataset includes multiple columns containing vital metadata for each of these organizations.

### Results: 
   * Data Preprocessing
   1. What variable(s) are the target(s) for your model? 
      Target variable is the 'IS_SUCCESSFUL' column from the DataFrame.
   2. What variable(s) are the features for your model?
      The feature variables consist of all columns from the DataFrame except for the 'IS_SUCCESSFUL' column, which has been removed as part of the data preprocessing.
   3. What variable(s) should be removed from the input data because they are neither targets nor features?
      Both the 'EIN' and 'NAME' columns were excluded from the dataset because they did not serve as either target variables or features for the dataset.
   
![DroppedCol](/images/droppedCol.png)

   * Compiling, Training, and Evaluating the Model
   1. How many neurons, layers, and activation functions did you select for your neural network model, and why?
      
      In the initial attempt, I selected 8 hidden nodes for the first layer and 5 hidden nodes for the second layer as initial choices, which were essentially arbitrary selections. These choices were intended as starting points for further iterations and adjustments in the second attempt to fine-tune the neural network model.

![firstAttempt](/images/layesFirtsAttempt.png)

   2. Were you able to achieve the target model performance?
      Regrettably, I was unable to attain the desired 75% model accuracy target.

   3. What steps did you take in your attempts to increase model performance?
      I made several modifications to the model to improve its accuracy, including adding more layers, removing additional columns, incorporating additional hidden nodes, and experimenting with different activation functions for each layer. These adjustments were aimed at enhancing the model's predictive performance.

### Summary:
In summary, the deep learning model achieved an accuracy of approximately 73% in predicting the classification problem. To achieve higher prediction accuracy, it would be advisable to improve the correlation between the input and output. This could involve more comprehensive data preprocessing, exploring different activation functions, and iterating on the model until a higher level of accuracy is achieved.

![summary](/images/Summary.png)