# data_app/views.py
from django.shortcuts import render, redirect
from django.http import HttpResponse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings #for ignoring warnings from module
from django.urls import reverse

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username == 'team-1' and password == 'URCE':
            return redirect('Home')
    return render(request, 'index.html')

def Home_view(request):
    return render(request, 'home.html')
'''
old view
def wine_quality_predictions(request):
    if request.method == 'POST':
        residual_sugar = request.POST.get('residual_sugar')
        chlorides = request.POST.get('chlorides')
        free_sulfur_dioxide = request.POST.get('free_sulfur_dioxide')
        total_sulfur_dioxide = request.POST.get('total_sulfur_dioxide')
        density = request.POST.get('density')
        pH = request.POST.get('pH')
        alcohol = request.POST.get('alcohol')
        values = (residual_sugar, chlorides, free_sulfur_dioxide,total_sulfur_dioxide, density, pH, alcohol)
        return HttpResponse("sucessfully recieved data")
    return render(request, 'wine_quality.html')
'''


#this works
def wine_quality_view(request):
    if request.method == 'POST':
        volatile_acidity = request.POST.get('volatile_acidity')
        fixed_acidity = request.POST.get('fixed_acidity')
        residual_sugar = request.POST.get('residual_sugar')
        chlorides = request.POST.get('chlorides')
        values = (volatile_acidity,fixed_acidity,residual_sugar, chlorides)


        # Convert the tuple to a comma-separated string
        values_str = ','.join(map(str, values))

        # Construct the URL with the tuple of values
        redirect_url = reverse('data', args=[values_str])
        #return HttpResponse("data read sucessfully")
        return redirect(redirect_url)
        '''
        #return redirect('data',values)


        # Construct the URL with parameters
        redirect_url = reverse('data', kwargs={
            'volatile_acidity':volatile_acidity,
            'fixed_acidity':fixed_acidity,
            'residual_sugar': residual_sugar,
            'chlorides': chlorides,
        })

        return redirect(redirect_url)'''
    return render(request, 'predictor.html')

def success_view(request):
    return HttpResponse("Data read successfully")


def your_view(request, values):
    # Convert the comma-separated string back to a tuple
    values_tuple = tuple(values.split(','))

    csv_path = 'static/data/winequality-red.csv'  # Path to your CSV file


    # Read CSV file using pandas
    wine_dataset = pd.read_csv(csv_path)

    #to get the shape of the data
    #print(wine_dataset.shape) #expected output: (1599, 12)

    #note: we have 12 col in which 11 are attributes(featurized parameters) and 12th is the label i.e quality
    #to get the head values of our data set i.e first 5 table values of our dataset
    #wine_dataset.head()

    #3rd step cleaning and preproceesing the data

    #to check weather the dataset had any missing values
    #wine_dataset.isnull().sum()

    #to check weather the dataset had duplicate row values
    #num_duplicate_rows = wine_dataset.duplicated().sum()

    #print("Number of duplicate rows:", num_duplicate_rows)

    #to remove the duplicate data rows from the dataset and give the shape of the dataset
    wine_dataset_cleaned = wine_dataset.drop_duplicates()
    #print("Shape of cleaned DataFrame:", wine_dataset_cleaned.shape)
    wine_dataset=wine_dataset_cleaned

    #to get the statistical measures of the dataset
    #wine_dataset.describe()

    #5th step seperation of labels and binarizing them into 0 or 1 for simple understanding
    #droping the lables into a variable name x
    x= wine_dataset.drop('quality',axis=1)
    #to see the quality values print them in next step
    #print(x)

    #Label binarization i
    #note :Label binarization into good or bad from the above quality numbers from 3 to 8 ie 6 values into
    #2 particular values i.e 0 or 1 , o for bad and 1 for good

    y=wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
    #to check weather the the labels of quality had changed print the next statement.
    #print(y)

    x=wine_dataset[['volatile acidity','fixed acidity','residual sugar','chlorides']]

    #6th step split the data into train and test data
    #note we need to create 4 variables i.e shown below
    x_train, x_test, y_train, y_test = sm.train_test_split(x,y,test_size=0.2,random_state=2)
    #to check weather they are splited exactly check with next statement
    #print(y.shape, y_train.shape, y_test.shape) # here y is the original data points

    #7th step Model training
    #Random forest classifier:
    #It is an ensemble model (which used more than a 2 or 3 model combination to  prediction)

    model = RandomForestClassifier()
    # to train the model
    model.fit(x_train,y_train) #x_train contains all the training data and y train contains the label data
    #that is quality values (0,1)

    #8th step  Evaluate the model (model evaluation)
    #We are going to use accuracy score values
    # accuracy on test data
    x_test_prediction = model.predict(x_test)
    #we are predicting the label values
    # now compare the values predicted by model with actual data values
    test_data_accuracy = accuracy_score(x_test_prediction, y_test) #y_test contain real actual values
    #to print the accuracy of the model
    #print('The accuracy of our trained model is: ', test_data_accuracy*100,'%')

    #step 8 Building a predictive system and preditc
    #Build a code that will predict all the chemical values
    # Your processing logic here, for example:
    Input_data = values_tuple
    Input_data_as_numpy_array = np.asarray(Input_data)
    Input_data_reshape = Input_data_as_numpy_array.reshape(1,-1)
    # Suppress the specific warning using exception handling
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
            prediction = model.predict(Input_data_reshape)
    except Warning:
            # Handle the warning if needed
            pass  # You can customize this part to handle the warning in a specific way
    #print(prediction)
    #[0]
    if (prediction[0]==1):
        #print("it is a good quality one")
        return render(request, 'good.html')
    else:
        #print("it is a bad quality wine")
        return render(request, 'bad.html')

    #return HttpResponse("csv data file Uploaded sucessfully")
