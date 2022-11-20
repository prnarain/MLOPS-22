

from sklearn.model_selection import train_test_split

X_data = range(10)
y_data = range(10)

def train():
    trainlist=[]
    for i in range(0,5):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3,random_state = i) # zero or any other integer
        print(y_test)
        trainlist.append(y_test)

    trainlist1=[]
    for i in range(0,5):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3,random_state = i) # zero or any other integer
        print(y_test)
        trainlist1.append(y_test)

    assert trainlist == trainlist1

def val():
    vallist = []
    for i in range(6,7): 
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3,random_state = i)
        print(y_test)
        vallist.append(y_test)

    vallist1 = []
    for i in range(6,7): 
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3,random_state = i)
        print(y_test)
        vallist1.append(y_test)

    assert vallist == vallist1

def test():
    testlist = []
    for i in range(8,10):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3,random_state = i) # zero or any other integer
        print(y_test)
        testlist.append(y_test)

    testlist1 = []
    for i in range(8,10):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3,random_state = i) # zero or any other integer
        print(y_test)
        testlist1.append(y_test)

    assert testlist == testlist1

