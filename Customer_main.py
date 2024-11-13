import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,f1_score,r2_score

data = pd.read_csv('customer_feedback_satisfaction.csv')

class Customer:
    def __init__(self,data:pd.DataFrame,model):
        self.data = data
        self.model = model
        
    def preprocessing(self,x_drop_column:list,y_column:str,label_column:list):
        label = LabelEncoder()
        for col in label_column:
            self.data[col] = label.fit_transform(self.data[col])
            
        x = self.data.drop(columns=x_drop_column)
        y = self.data[y_column]
        
        X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)
        
        return X_train,X_test,Y_train,Y_test
        
    def model_train(self,X_train,Y_train):
        self.model.fit(X_train,Y_train)
        return self.model
        
    def evaluate_model(self,X_test,Y_test):
        Y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(Y_test,Y_pred)
        mse = mean_squared_error(Y_test,Y_pred)
        r2 = r2_score(Y_test,Y_pred)

        print(f'mean_absolute_error : {mae:.2f}')
        print(f'mean_squared_error : {mse:.2f}')
        print(f"Your Model Accuracy : {round(r2*100,2)}")

if __name__ == "__main__":
    
    model = GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,max_depth=5,min_samples_leaf=10,min_samples_split=2)
    customer = Customer(data,model)
    
    x_drop_column = ['CustomerID','SatisfactionScore','Gender','Country','FeedbackScore','LoyaltyLevel']
    label_column = ['Gender','Country','FeedbackScore','LoyaltyLevel']
    
    X_train,X_test,Y_train,Y_test = customer.preprocessing(x_drop_column,'SatisfactionScore',label_column)
    
    customer.model_train(X_train,Y_train)
    
    customer.evaluate_model(X_test,Y_test)


# Heigh accuracy that got by GradientBoostingRegressor

# mean_absolute_error : 5.50
# mean_squared_error : 59.36
# Your Model Accuracy : 78.81