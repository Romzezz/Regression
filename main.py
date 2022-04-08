# Оптимизационные задачи в машинном обучении.
#Проект №4: Regression.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error


def Lin_reg(x, y, test_size =0.2, graph = False, in_pairs = False):
    '''
    Функция Lin_reg строит линейную регрессию для заданных данных.
        Параметры:
            x - pandas Датафрейм экзогенных переменных 
            x - pandas Датафрейм эндогенных переменных
            test_size - размер тестовой выборки. По умолчанию - 0.2
            graph - Если True, выводит график регрессии. По умолчанию - False
            in_pairs - Если True, строит парные регрессии для всех столбцов из x. По умолчанию -  False
            
    Функция возвраащает массив коэфициэнтов и свободный член регрессионной модели.
    
    
    '''
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    r2=[]
    if in_pairs:
        for i in x_train.columns:
            print(f'\nПарная регрессия для {y.columns[0]} и {i}')
            x_train1 = pd.DataFrame(x_train[i])
            y_train1 = pd.DataFrame(y_train)
            prm = LinearRegression()
            prm.fit(x_train1, y_train1)
            r2.append(r2_score(y_train1,prm.predict(x_train1)))
            print('Coef:',prm.coef_[0][0])
            print('Intercept:',prm.intercept_[0])
            print(f'R^2 for {i}: ', r2_score(y,prm.predict(pd.DataFrame(x[i]))))
            print('Mean squared error: ', mean_squared_error(y, prm.predict(pd.DataFrame(x[i]))))
            if graph:    
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,4))
                ax1.scatter(x_train1, y_train1,color = 'green', alpha=0.7, label = 'Обучающая выборка')
                ax1.scatter(x_test[i], y_test,color = 'blue', alpha=0.7,label = 'Тестовая выборка')
                ax1.plot(x_train1,prm.predict(x_train1), color = 'red', linewidth = 2 )
                ax1.legend()
                ax2.scatter(x_test[i], y_test,color = 'blue', label = 'Тестовые данные')
                ax2.scatter(np.array(x_test[i]).reshape(-1,1), prm.predict(pd.DataFrame(x_test[i])),color = 'red', label = 'Прогноз модели')
                ax2.legend()
                plt.show() 
    prm = LinearRegression()
    prm.fit(pd.DataFrame(x_train),pd.DataFrame(y_train))
    print(f'\nМножественная линейная регрессия для {y.columns[0]} и X')
    print('Coef:',prm.coef_[0])
    print('Intercept:',prm.intercept_[0])
    print('R^2 for all X: ', prm.score(x,y))
    print('Mean squared error: ', mean_squared_error(y, prm.predict(x)))
    print('\n\n[MODEL] Y = ',end='')
    for i in range(len(x.columns)):
        print(round(prm.coef_[0][i],4),'*',x.columns[i],end = ' + ')
    print(round(prm.intercept_[0],3))
    if graph:    
        if len(x.columns[0])==2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs = x[x.columns[0]].values, ys =x[x.columns[1]].values, zs = y)
            ax.scatter(x[x.columns[0]].values, x[x.columns[1]].values,prm.predict(x).reshape(1,-1), color = 'r')
        if len(x.columns[0])==1:    
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,4))
                ax1.scatter(x_train1, y_train1,color = 'green', alpha=0.7, label = 'Обучающая выборка')
                ax1.scatter(x_test[i], y_test,color = 'blue', alpha=0.7,label = 'Тестовая выборка')
                ax1.plot(x_train1,prm.predict(x_train1), color = 'red', linewidth = 2 )
                ax1.legend()
                ax2.scatter(x_test[i], y_test,color = 'blue', label = 'Тестовые данные')
                ax2.scatter(np.array(x_test[i]).reshape(-1,1), prm.predict(pd.DataFrame(x_test[i])),color = 'red', label = 'Прогноз модели')
                ax2.legend()
                plt.show() 

    return prm.coef_[0],prm.intercept_[0]




def Poly_reg(x, y,test_size =0.2, degree = 2, graph = False):
    '''
        Функция Poly_reg строит парные полиномиальные регрессии для каждого х и y из заданных данных.
        Параметры:
            x - pandas Датафрейм экзогенных переменных 
            x - pandas Датафрейм эндогенных переменных
            test_size - размер тестовой выборки. По умолчанию - 0.2
            graph - Если True, выводит график регрессии. По умолчанию - False
            degree - Степень полинома. По умолчанию -  2
    
    '''
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)    
    y_train1 = pd.DataFrame(y_train)
    for i in x_train.columns:
        print(f'\nПолиномиальные регрессии {degree} порядка для {i}')
        x_train1 = pd.DataFrame(x_train[i])
        poly_reg = PolynomialFeatures(degree=degree)
        poly = poly_reg.fit_transform(x_train1) 
        new_reg = LinearRegression().fit(poly, y_train1)
        print('Coef:',new_reg.coef_[0][1:])
        print('Intercept:',new_reg.intercept_[0])
        print('Train data R^2:',new_reg.score(poly, y_train)) #train
        print('Test data R^2:',new_reg.score(poly_reg.fit_transform(pd.DataFrame(x_test[i])), y_test)) # test
        print('MSE:',mean_squared_error(y_train, new_reg.predict(poly)))
        print('MAE:',mean_absolute_error(y_train, new_reg.predict(poly)))
        if graph:
            fig = plt.figure()
            plt.scatter(x[i], y)
            plt.plot(x[i].sort_values(), new_reg.predict(poly_reg.fit_transform(x[i].sort_values().values.reshape((-1, 1)))), color='r')
            plt.show()