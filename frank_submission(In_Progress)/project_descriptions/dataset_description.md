Dataset Description
Files
train.csv - the training set
test.csv - the test set
sample_submission.csv - a sample submission file in the correct format
Columns
INDEX_NR - Unique record identifier assigned to each strike report in the FAA database.
INCIDENT_DATE - Full date of the strike (MM/DD/YY).
INCIDENT_MONTH - Month the strike occurred (1–12).
INCIDENT_YEAR - Year the strike occurred.
TIME - Local time of the strike in HH:MM format (blank if unknown).
TIME_OF_DAY - Light conditions at time of strike (Day, Night, Dawn, Dusk).
AIRPORT_ID - ICAO identifier code for the nearest airport (e.g. KDFW, KLAX).
AIRPORT - Name of the nearest airport.
LATITUDE - Latitude coordinate of the airport.
LONGITUDE - Longitude coordinate of the airport.
RUNWAY - Runway designation where the strike occurred (blank if airborne).
STATE - US state or territory where the strike occurred (FN/FGN = foreign location).
FAAREGION - FAA administrative region code (e.g. AWP = Western Pacific, AEA = Eastern).
LOCATION - Free text description of location if the aircraft was enroute rather than at an airport.
OPID - Three-letter ICAO code identifying the aircraft operator.
OPERATOR - Name of the airline or operator (BUS = business, PVT = private, MIL = military).
REG - Aircraft tail/registration number.
FLT - Flight number.
AIRCRAFT - Aircraft model name (e.g. B-737-300, A-320, C-172).
AMA - ICAO aircraft make code.
AMO - ICAO aircraft model code.
EMA - Engine make code.
EMO - Engine model code.
AC_CLASS - Aircraft class (A = airplane, B = helicopter, C = glider, J = ultralight, Y = other).
AC_MASS - Aircraft weight class (1 = 2,250 kg or less through 5 = above 272,000 kg).
TYPE_ENG - Engine type (A = piston, B = turbojet, C = turboprop, D = turbofan, F = turboshaft, E = none).
NUM_ENGS - Number of engines on the aircraft.
ENG_1_POS - Mounting position of engine 1 on the aircraft.
ENG_2_POS - Mounting position of engine 2 on the aircraft.
ENG_3_POS - Mounting position of engine 3 on the aircraft.
ENG_4_POS - Mounting position of engine 4 on the aircraft.
PHASE_OF_FLIGHT - Phase of flight when the strike occurred (Approach, Climb, Descent, En Route, Landing Roll, Parked, Take-off Run, Taxi).
HEIGHT - Height above ground level in feet at time of strike (0 = on the ground).
SPEED - Indicated airspeed in knots at time of strike.
DISTANCE - Distance from the airport in miles at time of strike.
SKY - Sky condition at time of strike (No Cloud, Some Cloud, Overcast).
PRECIPITATION - Precipitation present at time of strike (None, Rain, Fog, Snow, Hail, etc.).
BIRD_BAND_NUMBER - Federal band ID number if the bird was banded.
SPECIES_ID - ICAO species code identifying the type of bird or wildlife.
SPECIES - Common name of the bird or other wildlife involved in the strike.
OUT_OF_RANGE_SPECIES - Flag indicating the species was reported outside its normal geographic range (1 = yes).
REMARKS - Notes from the report form or database manager.
REMAINS_COLLECTED - Whether physical remains of the bird/wildlife were recovered at the scene (0/1).
REMAINS_SENT - Whether remains were sent to the Smithsonian Institution for identification (0/1).
WARNED - Whether the pilot received a prior warning of bird/wildlife activity (Yes, No, Unknown).
NUM_SEEN - Number of birds or animals seen by the pilot just before the strike.
NUM_STRUCK - Number of birds or animals actually struck.
SIZE - Pilot's perceived size of the bird or animal (Small, Medium, Large).
ENROUTE_STATE - State where the strike occurred if the aircraft was enroute rather than at an airport.
COMMENTS - Additional administrative notes about the record.
SOURCE - Type of report filed (FAA Form 5200-7, Air Transport Report, Airport Report, Multiple, Other).
PERSON - Role of the person who filed the report (Pilot, Tower, Air Transport Operations, Airport Operations, etc.).
LUPDATE - Date the record was last updated in the FAA database.
TRANSFER - Internal database flag
INDICATED_DAMAGE - Target variable. Whether the wildlife strike caused damage to the aircraft (1 = damage, 0 = no damage).