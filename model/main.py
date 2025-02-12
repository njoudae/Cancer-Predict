
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

def create_model(data):
    y = data['diagnosis']
    x = data.drop(['diagnosis'], axis=1)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_hat = model.predict(x_test)
    print("Accuracy: ", accuracy_score(y_hat, y_test))
    print("Report: ", classification_report(y_hat, y_test))
    return model, scaler

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data

def main():
    data = get_clean_data()
    #print(data.head())
    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()
