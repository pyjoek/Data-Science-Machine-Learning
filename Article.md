# UNDERSTANDING LINEAR REGRETION
## Linear Regretion
    Using line to do prediction

## Steps
### Importing Libraries
    sklearn.datasets - fetch_openml
        - for testing boston, you'll need that library
    sklearn.linear_model - LinearRegression
    sklearn.model_selection - train_test_split
    sklearn.metrics - mean_squared_error
    pandas
    numpy
    matplotlib.pyplot

### Initiating the boston datas into data frame
    boston_dataset = fetch_openml(name='boston', version=1)
    boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)

### Setting the target column
    boston['MEDV'] = boston_dataset.target

### Cleaning your data
    boston.describe(include='all')
    boston[['RM', 'MEDV']].isnull().sum()
    boston['rm'].fillna(boston['rm'].median(), inplace=True)
    boston['medv'].fillna(boston['medv'].median(), inplace=True)

### Check for correlating columns
    corr_matrix = boston.corr().round(2)
    corr_matrix

### You can test visual presentation.(hist, scatter, line) Optional
    # Hist
        boston.hist(column='CRIM')
        plt.show()
    # Scatter
        boston.plot(kind = 'scatter', x = 'RM', y = 'MEDV', figsize=(8,6))
        plt.show()

### Init your model
    model = LinearRegression()

### split your dataset to train and test datas
    X = boston['RM']
    Y = boston['MEDV']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

### fit your model
    model.fit(X_train, Y_train)

### get your intercept and coefficient
    model.intercept_.round(2), model.coef_.round(2)

### Test Prediction
    # Method 1;
        new_RM = np.array([6.5]).reshape(-1, 1)
        model.predict(new_RM)

    # Method 2;
        model.intercept_ + model.coef_*6.5

    # Method 3; --  Used This for Visualizing the graph
        y_test_prediction = model.predict(X_test)
        y_test_prediction.shape
        type(y_test_prediction)

### Visuaize your prediction
    plt.scatter(X_test, Y_test, label='testing data')
    plt.plot(X_test, y_test_prediction, label='prediction', linewidth=3)
    plt.xlabel('RM')
    plt.ylabel('MEDV')
    plt.legend(loc='upper left')
    plt.show()

