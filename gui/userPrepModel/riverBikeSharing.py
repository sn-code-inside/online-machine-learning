from river import preprocessing
from river import compose
from river import feature_extraction
from river import stats

# Original data has the following columns:
# x1: {'moment': datetime.datetime(2016, 4, 1, 0, 0, 7),
# x2:  'station': 'metro-canal-du-midi',
# x3:  'clouds': 75,
# x4:  'description': 'light rain',
# x5:  'humidity': 81,
# x6:  'pressure': 1017.0,
# x7:  'temperature': 6.54,
# x8:  'wind': 9.3}


def get_hour(x):
    x['hour'] = x['x1'].hour
    return x


def set_prep_model():
    selector = compose.Select('x3', 'x5', 'x6', 'x7', 'x8')
    # model += (
    #     get_hour |
    #     feature_extraction.TargetAgg(by=['x2', 'hour'], how=stats.Mean())
    # )
    model = compose.Pipeline(selector, preprocessing.StandardScaler())
    return model