# =================================
# Initial comments for preprcessing
# =================================

#drop ReservationStatus and ReservationStatusDate. Check-Out always means IsCanceled=0, and Canceled or No-Show always means 1. The status date is when that outcome was recorded
#Text columns are padded with whitespace. Strip them before any grouping or encoding.
#Agent and Company use the string 'NULL' instead of real missing values, so isna() won’t find them.
#Children is float, but should be an integer.
#167 rows with zero guests and 331 rows with zero nights: remove them.
#Undefined categories: MarketSegment (2 rows), DistributionChannel (4 rows). Merge into the most common category.
#296 repeat guests have no prior bookings recorded, and 4,647 non-repeat guests have previous cancellations.
#DaysInWaitingList is 0 for 96% of rows. Convert to a binary flag.
#Lead times over 500 days (455 rows) and stays over 30 nights (15 rows) are rare but plausible, cap them rather than delete.

#Feature engineering ideas:
#TotalNights, TotalGuests, HasKids, RoomTypeChanged
#An actual arrival date, which gives weekday and season
#Total cost (ADR × TotalNights)
#The redundant columns ArrivalDateWeekNumber and ArrivalDateDayOfMonth can be dropped once we have a proper date.

#The 2015 data covers only 6 months, 2016 covers 12 and 2017 covers 8. Don’t treat ArrivalDateYear as a generalisable feature, and be careful with month effects because the years aren’t balanced.
#Stratify on IsCanceled.

#Duplicates and Categorical encoding 


# Import the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Import the data
booking_data = pd.read_csv("data/Hotel_bookings.csv")

# Remove Missing values in ['Children', 'Country'] columns
booking_data.dropna(subset=['Children', 'Country'], inplace=True)
print (booking_data.shape) # New shape => (79302, 31)

# Remove invalid bookings
booking_data = booking_data[
    ~(
        ((booking_data['StaysInWeekNights'] == 0) & (booking_data['StaysInWeekendNights'] == 0) & (booking_data['IsCanceled'] == 0)) |
        ((booking_data['Adults'] == 0) & (booking_data['IsCanceled'] == 0)) | 
        (booking_data['MarketSegment'] == 'Undefined') | 
        (booking_data['DistributionChannel'] == 'Undefined')
    )
]
print (booking_data.shape) # New shape => (78907, 31)

# Convert "NULL" in ['Company'] into "Unknown" as a category
booking_data['Company'] = booking_data['Company'].str.strip()
booking_data['Company'] = booking_data['Company'].str.replace('NULL', 'Unknown', regex=False)
print (booking_data['Company'].value_counts())
print (booking_data.shape)

# Remove row with ADR 5400 [Very outlier]
booking_data = booking_data[
    ~(
        ((booking_data['ADR'] == 5400)))
]
print (booking_data.shape)

# drop duplicates
booking_data.drop_duplicates(inplace=True)
print(booking_data.shape)
print (booking_data['Babies'].value_counts())


# Clip extreme outliers in non-continous numeric columuns to specific upper limit
# booking_data['Babies'] = booking_data['Babies'].clip(upper=1)
# booking_data['Children'] = booking_data['Children'].clip(upper=2)  
train_data, test_data = train_test_split(
    booking_data, 
    test_size=0.2, 
    random_state=42, 
    stratify=booking_data['IsCanceled']
) 
print (train_data.columns)
print (train_data['ArrivalDateYear'].value_counts()) 
print (train_data['ArrivalDateMonth'].value_counts()) 
print (train_data['ArrivalDateWeekNumber'].value_counts())
print (train_data['ArrivalDateDayOfMonth'].value_counts())
print (train_data['StaysInWeekendNights'].value_counts())
print (train_data['StaysInWeekNights'].value_counts())
print (train_data['Adults'].value_counts())
print (train_data['Children'].value_counts())
print (train_data['Babies'].value_counts())

# Remove outliers from babies
train_data = train_data[train_data['Babies'] <2]

# 'Meal', 'Country', 'MarketSegment', 'DistributionChannel',
#        'IsRepeatedGuest', 'PreviousCancellations',
#        'PreviousBookingsNotCanceled', 'ReservedRoomType', 'AssignedRoomType',
#        'BookingChanges', 'DepositType', 'Agent', 'Company',
#        'DaysInWaitingList', 'CustomerType', 'ADR', 'RequiredCarParkingSpaces',
#        'TotalOfSpecialRequests', 'ReservationStatus', 'ReservationStatusDate'
print (train_data['Meal'].value_counts())
print (test_data['Meal'].value_counts())
print (train_data['Country'].value_counts())
print (test_data['Country'].value_counts())
# Handle rare countries
threshold = 0.02
country_proportions = train_data['Country'].value_counts(normalize=True)
frequent_countries = country_proportions[country_proportions >= threshold].index.tolist()
train_data['Country'] = train_data['Country'].apply(lambda val: val if val in frequent_countries else 'Other')
test_data['Country'] = test_data['Country'].apply(lambda val: val if val in frequent_countries else 'Other')
print (train_data['Country'].value_counts())
print (test_data['Country'].value_counts())
print (train_data['MarketSegment'].value_counts())
print (test_data['MarketSegment'].value_counts())
print (train_data['DistributionChannel'].value_counts())
print (test_data['DistributionChannel'].value_counts())
print (train_data['IsRepeatedGuest'].value_counts())
print (test_data['IsRepeatedGuest'].value_counts())
print (train_data['PreviousCancellations'].value_counts())
print (test_data['PreviousCancellations'].value_counts())
#Remove outluers from PreviousCancellations
train_data = train_data[train_data['PreviousCancellations'] <= 6]
print (train_data['PreviousBookingsNotCanceled'].value_counts())
print (test_data['PreviousBookingsNotCanceled'].value_counts())

print (train_data[train_data['PreviousBookingsNotCanceled'] >= 5]['ADR'].count())
# Handle outliers in PreviousBookingsNotCanceled by clipping to a maximum value of 5
max_val = 7
train_data['PreviousBookingsNotCanceled'] = train_data['PreviousBookingsNotCanceled'].clip(upper=max_val)
test_data['PreviousBookingsNotCanceled'] = test_data['PreviousBookingsNotCanceled'].clip(upper=max_val) 
print (train_data['PreviousBookingsNotCanceled'].value_counts())
print (test_data['PreviousBookingsNotCanceled'].value_counts())
print (train_data['ReservedRoomType'].value_counts())
print (test_data['ReservedRoomType'].value_counts())

frequent_rooms = ['A', 'D', 'F', 'E', 'B', 'G']
train_data['ReservedRoomType'] = train_data['ReservedRoomType'].apply(lambda val: val if val in frequent_rooms else 'Other')
test_data['ReservedRoomType'] = test_data['ReservedRoomType'].apply(lambda val: val if val in frequent_rooms else 'Other')

print (train_data['AssignedRoomType'].value_counts())
print (test_data['AssignedRoomType'].value_counts())


frequent_assigned_rooms = ['A', 'D', 'E', 'F', 'B', 'G', 'C', 'K']
train_data['AssignedRoomType'] = train_data['AssignedRoomType'].apply(lambda val: val if val in frequent_assigned_rooms else 'Other')
test_data['AssignedRoomType'] = test_data['AssignedRoomType'].apply(lambda val: val if val in frequent_assigned_rooms else 'Other')

print (train_data['BookingChanges'].value_counts())
print (test_data['BookingChanges'].value_counts()) 

max_val = 5
train_data['BookingChanges'] = train_data['BookingChanges'].clip(upper=max_val)
test_data['BookingChanges'] = test_data['BookingChanges'].clip(upper=max_val)

print (train_data['DepositType'].value_counts())
print (test_data['DepositType'].value_counts()) 

pd.set_option('display.max_rows', None)       
pd.set_option('display.max_columns', None)     
pd.set_option('display.max_colwidth', None)
print (train_data['Agent'].value_counts())
print (test_data['Agent'].value_counts()) 
print (train_data['Agent'].nunique())
print (test_data['Agent'].nunique()) 

# Chosing the most frequent agents and grouping the rest into 'Other'
frequent_agents = ['9', 'NULL', '14', '7', '28', '1', '8']
def clean_agent(val):
    val_str = str(val).strip()
    if val_str in frequent_agents:
        return val_str
    else:
        return 'Other'

train_data['Agent'] = train_data['Agent'].apply(clean_agent)
test_data['Agent'] = test_data['Agent'].apply(clean_agent) 
print (train_data['Agent'].value_counts())
print (test_data['Agent'].value_counts()) 
print (train_data['Agent'].nunique())
print (test_data['Agent'].nunique())   

print (train_data['Company'].value_counts())
print (test_data['Company'].value_counts()) 
print (train_data['Company'].nunique())
print (test_data['Company'].nunique())   

frequent_companies = ['Unknown', '40', '45', '153', '219', '174', '233', '51']

train_data['Company'] = train_data['Company'].apply(lambda val: val if val in frequent_companies else 'Other_company')
test_data['Company'] = test_data['Company'].apply(lambda val: val if val in frequent_companies else 'Other_company') 
print (train_data['CustomerType'].value_counts())
print (test_data['CustomerType'].value_counts()) # No issues with this feature 
print (train_data['RequiredCarParkingSpaces'].value_counts())
print (test_data['RequiredCarParkingSpaces'].value_counts()) 

# Required car parking spaces with values 2, 3 are very rare. with only 5 values in  the train set so, we will remove them to avoid noise. 
max_parking = 1
train_data['RequiredCarParkingSpaces'] = train_data['RequiredCarParkingSpaces'].clip(upper=max_parking)
test_data['RequiredCarParkingSpaces'] = test_data['RequiredCarParkingSpaces'].clip(upper=max_parking) 
print (train_data['TotalOfSpecialRequests'].value_counts())
print (test_data['TotalOfSpecialRequests'].value_counts()) #No issues with this feature.
print (train_data['ReservationStatus'].value_counts())
print (test_data['ReservationStatus'].value_counts()) # No issues with this feature. 
# weird column with 864 unique values.
print (train_data['ReservationStatusDate'].value_counts())
print (test_data['ReservationStatusDate'].value_counts())
print (train_data['ReservationStatusDate'].nunique())
print (test_data['ReservationStatusDate'].nunique()) 
print(booking_data['ReservationStatusDate'].nunique())

