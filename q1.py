

from sklearn.model_selection import train_test_split

X_data = range(10)
y_data = range(10)

for i in range(0,5):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3,random_state = i) # zero or any other integer
    print(y_test)

print("-"*30)

for i in range(6,7): 
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3,random_state = i)
    print(y_test)

print("-"*30)

for i in range(8,10):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3,random_state = i) # zero or any other integer
    print(y_test)

print("-"*30)
