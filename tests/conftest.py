import pandas as pd
import pytest


@pytest.fixture
def tiny_raw_frames():
    """Minimal raw rows matching FAA column names (before cleaning)."""
    train = pd.DataFrame(
        {
            "INDEX_NR": [1, 2, 3, 4],
            "INCIDENT_DATE": ["1/15/2010", "2/1/10", "3/1/10", "4/1/10"],
            "INCIDENT_MONTH": [1, 2, 3, 4],
            "INCIDENT_YEAR": [2010, 2010, 2010, 2010],
            "TIME": ["12:00", "", "8:30", "15:30"],
            "TIME_OF_DAY": ["Day", "Night", "Day", "Day"],
            "AIRPORT_ID": ["KAAA", "KBBB", "KAAA", "KCCC"],
            "AIRPORT": ["A", "B", "A", "C"],
            "LATITUDE": [40.0, 41.0, 40.0, 42.0],
            "LONGITUDE": [-74.0, -75.0, -74.0, -76.0],
            "RUNWAY": ["9", "27", "9", "18"],
            "STATE": ["NY", "CA", "NY", "TX"],
            "FAAREGION": ["E", "W", "E", "C"],
            "LOCATION": [pd.NA, pd.NA, pd.NA, pd.NA],
            "OPID": ["X", "Y", "X", "Z"],
            "OPERATOR": ["U1", "U2", "U1", "U3"],
            "REG": ["N1", "N2", "N1", "N3"],
            "FLT": ["100", "200", "100", "300"],
            "AIRCRAFT": ["B737", "A320", "B737", "E170"],
            "AMA": ["148", "04A", "148", "332"],
            "AMO": [10.0, 11.0, 10.0, 12.0],
            "EMA": [22.0, 22.0, 22.0, 22.0],
            "EMO": [4.0, 4.0, 4.0, 4.0],
            "AC_CLASS": ["A", "A", "A", "A"],
            "AC_MASS": [4.0, 4.0, 4.0, 4.0],
            "TYPE_ENG": ["D", "D", "D", "D"],
            "NUM_ENGS": [2.0, 2.0, 2.0, 2.0],
            "ENG_1_POS": [1.0, 1.0, 1.0, 1.0],
            "ENG_2_POS": [1.0, 1.0, 1.0, 1.0],
            "ENG_3_POS": [pd.NA, pd.NA, pd.NA, pd.NA],
            "ENG_4_POS": [pd.NA, pd.NA, pd.NA, pd.NA],
            "PHASE_OF_FLIGHT": ["Approach", "Climb", "Landing Roll", "Approach"],
            "HEIGHT": [100.0, 200.0, 0.0, 150.0],
            "SPEED": [140.0, 160.0, 120.0, 145.0],
            "DISTANCE": [0.0, 1.0, 0.0, 0.0],
            "SKY": ["Clear", "Clear", "Clear", "Some Cloud"],
            "PRECIPITATION": [pd.NA, pd.NA, pd.NA, pd.NA],
            "BIRD_BAND_NUMBER": [pd.NA, pd.NA, pd.NA, pd.NA],
            "SPECIES_ID": ["S1", "S2", "S1", "S3"],
            "SPECIES": ["sparrow", "goose", "sparrow", "gull"],
            "OUT_OF_RANGE_SPECIES": [0, 0, 0, 0],
            "REMARKS": ["a", "bb", "ccc", "dddd"],
            "REMAINS_COLLECTED": [1, 0, 1, 0],
            "REMAINS_SENT": [0, 0, 0, 0],
            "WARNED": ["No", "Yes", "Unknown", "No"],
            "NUM_SEEN": ["1", "1-2", "10-Feb", "1"],
            "NUM_STRUCK": ["1", "1", "1", "1-2"],
            "SIZE": ["Small", "Large", "Small", "Medium"],
            "ENROUTE_STATE": [pd.NA, pd.NA, pd.NA, pd.NA],
            "COMMENTS": ["x", "y", "z", "w"],
            "SOURCE": ["FAA", "FAA", "FAA", "FAA"],
            "PERSON": ["Pilot", "Pilot", "Pilot", "Pilot"],
            "LUPDATE": ["1/1/20", "1/1/20", "1/1/20", "1/1/20"],
            "TRANSFER": [0, 0, 0, 0],
            "INDICATED_DAMAGE": [0, 1, 0, 1],
        }
    )
    test = train.drop(columns=["INDICATED_DAMAGE"]).copy()
    test["INDEX_NR"] = [10, 11, 12, 13]
    return train, test
