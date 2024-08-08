from river import preprocessing
from river import compose
from river import feature_extraction
from river import stats


# x1: station
# x2: clouds
# x3: humidity
# x4: pressure
# x5: temperature
# x6: wind
# x7: hour
# y: bikes

def get_hour(x):
    return x['x7']


def set_prep_model():
    # selector = compose.Select('x2', 'x3', 'x4', 'x5', 'x6')
    # selector += (
    #     get_hour |
    #     feature_extraction.TargetAgg(by=['x1', 'x7'], how=stats.Mean())
    # )
    # model = compose.Pipeline(selector, preprocessing.StandardScaler())
    # model = preprocessing.StandardScaler()
    selector = compose.Select('x2', 'x3', 'x4')
    model = compose.Pipeline(selector, preprocessing.StandardScaler())
    return model
