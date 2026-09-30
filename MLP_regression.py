from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

from data_utils import load_split


def get_data(feature, target):
    x_tr, x_ts, y_tr, y_ts, _ = load_split(feature, target)

    print(f'Train size : {len(y_tr)}, Test size : {len(y_ts)}')

    return (x_tr, y_tr, x_ts, y_ts)


def mlp(dataset):
    x_tr, y_tr, x_ts, y_ts = dataset
    print('Training starts...')
    reg = MLPRegressor(hidden_layer_sizes=(256, 512, 256), activation='relu',
    random_state=42, max_iter=10000).fit(x_tr, y_tr)
    print('Training ends...')

    tr_pred = reg.predict(x_tr)
    ts_pred = reg.predict(x_ts)

    tr_rmse = mean_squared_error(y_tr, tr_pred) ** 0.5
    tr_r2 = r2_score(y_tr, tr_pred)
    ts_rmse = mean_squared_error(y_ts, ts_pred) ** 0.5
    ts_r2 = r2_score(y_ts, ts_pred)

    print('Training results')
    print(f'RMSE : {tr_rmse}, R-square : {tr_r2}')
    print('Test results')
    print(f'RMSE : {ts_rmse}, R-square : {ts_r2}')


if __name__ == '__main__':
    mlp(get_data('high', 'Heat of formation'))
    mlp(get_data('mid', 'Heat of formation'))
    mlp(get_data('low', 'Heat of formation'))
