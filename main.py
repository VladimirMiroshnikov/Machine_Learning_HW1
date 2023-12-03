from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Literal
import pandas as pd
import pickle
import re
import numpy as np
from sklearn import impute
from datetime import date
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import json
from io import BytesIO
from fastapi.responses import FileResponse


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def prepare_col(x):
    if type(x) != float and any(map(str.isdigit, x)):
        s = x.split()[0]
    elif type(x) == float:
        s = x
    else:
        s = np.nan
    return s


def prepare_torque(x):
    if type(x) != float:
        try:
            s1 = x.replace('at', '@').split('@')[0]
            s2 = x.replace('at', '@').split('@')[1]
        except:
            s1 = x.replace('at', '@').split('@')[0]
            s2 = x.replace('at', '@').split('@')[0]
        if any(map(str.isalpha, s1)) == False:  # Если не указаны единицы измерения сразу за первым числом
            try:
                s1 = s1 + s2.split('(')[
                    1]  # Добавим единицы измерения из правой части строки в позицию за первым числом
            except:
                pass
    else:
        s1 = x
        s2 = x
    return s1


def torque_convert(x):
    if type(x) != float and len(re.findall('[(/]', x)) == 0:
        s = re.sub('[^a-zA-Z]', '', x)
        if s.upper() == 'KGM':
            x_new = float(re.sub('[a-zA-Z]', '', x)) * 9.80665
        else:
            x_new = float(re.sub('[a-zA-Z]', '', x))
    elif type(x) == float:
        x_new = x
    else:
        x_new = np.nan
    return x_new


def prepare_max_torque_rpm(x):
    if type(x) != float and len(re.findall('[+]', x)) == 0:
        try:
            s2 = x.replace('at', '@').split('@')[1]
            try:
                s2 = s2.split('-')[1]
            except:
                try:
                    s2 = s2.split('~')[1]
                except:
                    s2 = s2
        except:
            s2 = np.nan
    elif type(x) == float:
        s2 = x
    else:
        s2 = np.nan
    return s2


def convert_max_torque_rpm(x):
    if type(x) != float:
        x_new = re.sub('[^0-9]', '', x)
    else:
        x_new = x
    return x_new


def prepare_df_no_mis(X_train_: pd.DataFrame, X_test_: pd.DataFrame,
                      strategy: Literal['mean', 'median', 'most_frequent']):
    cat_features_mask = (X_train_.dtypes == 'object').values

    X_train_real = X_train_[X_train_.columns[~cat_features_mask]]
    X_test_real = X_test_[X_test_.columns[~cat_features_mask]]

    mis_repl = impute.SimpleImputer(strategy=strategy)

    X_train_real_no_mis = pd.DataFrame(data=mis_repl.fit_transform(X_train_real), columns=X_train_real.columns)
    X_test_real_no_mis = pd.DataFrame(data=mis_repl.transform(X_test_real), columns=X_test_real.columns)

    X_train_no_mis = pd.concat([X_train_real_no_mis, X_train_[X_train_.columns[cat_features_mask]]], axis=1)
    X_test_no_mis = pd.concat([X_test_real_no_mis, X_test_[X_test_.columns[cat_features_mask]]], axis=1)

    X_train_no_mis[['engine', 'seats']] = X_train_no_mis[['engine', 'seats']].astype('int16')
    X_test_no_mis[['engine', 'seats']] = X_test_no_mis[['engine', 'seats']].astype('int16')

    return X_train_no_mis, X_test_no_mis  # каждая функция, где есть transform должна возвращать и train и test


def gen_new_features(X: pd.DataFrame):
    X['hp_per_1l_engine'] = X.max_power / X.engine
    X['age'] = date.today().year - X.year
    X['age_squared'] = X['age'] ** 2
    X['km_driven_per_year'] = (X.km_driven / X['age']).astype('int32')
    X['brand'] = X.name.apply(lambda x: x.split()[0])
    X['1_or_2_ownrs_Dealer'] = np.where(
        X.owner.isin(['First Owner', 'Second Owner']) & (X.seller_type == 'Dealer'), 1, 0)

    return X


def scale_encode2(X: pd.DataFrame, scaler_: StandardScaler, ohe_: OneHotEncoder):
    cat_features_mask_ = (X.dtypes == 'object').values

    X_real = X[X.columns[~cat_features_mask_]].drop('1_or_2_ownrs_Dealer', axis=1)

    X_real_scaled = pd.DataFrame(scaler_.transform(X_real), columns=X_real.columns)

    X_cat = X[X.columns[cat_features_mask_]].drop('name', axis=1)

    X_cat_enc = pd.DataFrame(ohe_.transform(X_cat).toarray(),
                             columns=ohe_.get_feature_names_out())

    X_full = pd.concat([X_real_scaled, X_cat_enc, X['1_or_2_ownrs_Dealer']], axis=1)

    return X_full


sorted_feature_list = ['age', 'year', 'brand_Lexus', 'owner_Test Drive Car', 'brand_Volvo', 'brand_BMW', 'brand_Land',
                       'brand_Jaguar', 'brand_Mercedes-Benz', 'brand_Audi', 'brand_Jeep', 'brand_MG', 'brand_Isuzu',
                       'age_squared', 'brand_Toyota',
                       'brand_Kia', 'brand_Force', 'fuel_LPG', 'brand_Mitsubishi', 'seller_type_Individual',
                       'seller_type_Trustmark Dealer',
                       'brand_Datsun', 'brand_Peugeot', 'torque', 'brand_Tata', 'brand_Daewoo', 'engine',
                       '1_or_2_ownrs_Dealer',
                       'brand_Fiat', 'brand_Mahindra', 'brand_Volkswagen', 'transmission_Manual',
                       'owner_Fourth & Above Owner', 'brand_Chevrolet',
                       'brand_Maruti', 'max_power', 'brand_Renault', 'max_torque_rpm', 'fuel_Petrol',
                       'owner_Second Owner', 'km_driven',
                       'hp_per_1l_engine', 'brand_Skoda', 'mileage', 'owner_Third Owner', 'brand_Honda', 'fuel_Diesel',
                       'brand_Ford',
                       'brand_Nissan', 'brand_Hyundai', 'seats', 'km_driven_per_year']

with open('model.pkl', 'rb') as file:
    model_p = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler_p = pickle.load(file)

with open('ohe.pkl', 'rb') as file:
    ohe_p = pickle.load(file)


L = []
L_dfs = []


@app.post('/upload_json')
def upload_json(file: UploadFile):
    content = file.file.read() #считываем байтовое содержимое
    buffer = BytesIO(content) #создаем буфер типа BytesIO
    df = pd.read_json(buffer, typ='series').to_frame().transpose()
    buffer.close()
    file.close()
    L.append(df)
    return df.to_dict(orient='index')[0]


@app.post('/upload_csv')
def upload_csv(file: UploadFile):
    content = file.file.read() #считываем байтовое содержимое
    buffer = BytesIO(content) #создаем буфер типа BytesIO
    df = pd.read_csv(buffer, index_col=0)
    buffer.close()
    file.close()
    #df_ = pd.DataFrame(df.to_dict(orient='index').values())
    L_dfs.append(df.to_dict(orient='index'))
    return df.to_dict(orient='index')


@app.post("/predict_item")
def predict_item():
    car_from_json = L[0] #pd.read_json(L[0], typ='series').to_frame().transpose()
    for c in ['mileage', 'engine', 'max_power']:
        car_from_json[c] = car_from_json[c].apply(prepare_col).astype('float')
    for c in ['year', 'selling_price', 'km_driven']:
        car_from_json[c] = car_from_json[c].astype('int64')
    car_from_json['seats'] = car_from_json['seats'].astype('float64')
    car_from_json['max_torque_rpm'] = car_from_json['torque'].apply(prepare_max_torque_rpm).apply(
        convert_max_torque_rpm).astype('float')
    car_from_json['torque'] = car_from_json['torque'].apply(prepare_torque).apply(torque_convert).astype('float')

    car_from_json.drop('selling_price', axis=1, inplace=True)
    car_from_json_fe = gen_new_features(prepare_df_no_mis(car_from_json, car_from_json, 'median')[0])
    pred = model_p.predict(scale_encode2(car_from_json_fe, scaler_p, ohe_p)[sorted_feature_list[:42]])[0]
    return pred


@app.post("/predict_items")
def predict_items():
    cars_from_csv = pd.DataFrame(L_dfs[0]).transpose() #pd.read_json(L[0], typ='series').to_frame().transpose()
    for c in ['mileage', 'engine', 'max_power']:
        cars_from_csv[c] = cars_from_csv[c].apply(prepare_col).astype('float')
    for c in ['year', 'selling_price', 'km_driven']:
        cars_from_csv[c] = cars_from_csv[c].astype('int64')
    cars_from_csv['seats'] = cars_from_csv['seats'].astype('float64')

    cars_from_csv['max_torque_rpm'] = cars_from_csv['torque'].apply(prepare_max_torque_rpm).apply(
        convert_max_torque_rpm).astype('float')
    cars_from_csv['torque'] = cars_from_csv['torque'].apply(prepare_torque).apply(torque_convert).astype('float')

    cars_from_csv.drop('selling_price', axis=1, inplace=True)
    cars_from_csv_fe = gen_new_features(prepare_df_no_mis(cars_from_csv, cars_from_csv, 'median')[0])
    cars_from_csv['predicted_price'] = model_p.predict(scale_encode2(cars_from_csv_fe, scaler_p, ohe_p)[sorted_feature_list[:42]])
    cars_from_csv.to_csv('cars_pred_prices.csv')
    response = FileResponse(path='cars_pred_prices.csv', media_type='text/csv', filename='cars_pred_prices_dwnld.csv')
    return response
