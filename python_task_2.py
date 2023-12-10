import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

def calculate_distance_matrix(csv_file):
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Create a list of unique IDs
    unique_ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))

    # Initialize an empty distance matrix
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)
    distance_matrix = distance_matrix.fillna(0)  # Fill NaN with 0

    # Iterate over rows in the DataFrame to populate the distance matrix
    for index, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']

        # Update the distance matrix for bidirectional routes
        distance_matrix.at[id_start, id_end] += distance
        distance_matrix.at[id_end, id_start] += distance

    return distance_matrix

# Example usage
csv_file_path = ('C:\\Users\\user\\Desktop\\Mapup\\MapUp-Data-Assessment-F-main\\datasets\\dataset-3.csv')

resulting_matrix = calculate_distance_matrix(csv_file_path)
print(resulting_matrix)



def unroll_distance_matrix(distance_matrix):
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """

    # Get the upper triangular part of the distance matrix (excluding the diagonal)
    upper_triangle = distance_matrix.where(np.triu(np.ones(distance_matrix.shape), k=1).astype(bool))
    
    # Stack the upper triangle to create a Series
    stacked_distances = upper_triangle.stack().reset_index()
    stacked_distances.columns = ['id_start', 'id_end', 'distance']

    
    return stacked_distances

# Example usage
csv_file_path = ('C:\\Users\\user\\Desktop\\Mapup\\MapUp-Data-Assessment-F-main\\datasets\\dataset-3.csv')
distance_matrix = calculate_distance_matrix(csv_file_path)
unrolled_distances = unroll_distance_matrix(distance_matrix)
print(unrolled_distances)


def find_ids_within_ten_percentage_threshold(unrolled_distances, reference_value):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter rows based on the reference value
    reference_rows = unrolled_distances[unrolled_distances['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    average_distance = reference_rows['distance'].mean()

    # Calculate the threshold range
    lower_threshold = average_distance - (average_distance * 0.1)
    upper_threshold = average_distance + (average_distance * 0.1)

    # Filter rows within the 10% threshold range
    within_threshold_rows = unrolled_distances[
        (unrolled_distances['distance'] >= lower_threshold) &
        (unrolled_distances['distance'] <= upper_threshold)
    ]

    # Get unique values from the 'id_start' column and sort them
    result_ids = sorted(within_threshold_rows['id_start'].unique())

    return result_ids

# Example usage
csv_file_path = ('C:\\Users\\user\\Desktop\\Mapup\\MapUp-Data-Assessment-F-main\\datasets\\dataset-3.csv')
distance_matrix = calculate_distance_matrix(csv_file_path)
distance_matrix = calculate_distance_matrix(csv_file_path)
unrolled_distances = unroll_distance_matrix(distance_matrix)

reference_value = 1001400  # Replace with your desired reference value
resulting_ids = find_ids_within_ten_percentage_threshold(unrolled_distances, reference_value)
print(resulting_ids)

def calculate_toll_rate(df):
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Adding toll rate columns with respective coefficients
    df['moto'] = df['distance'] * 0.8
    df['car'] = df['distance'] * 1.2
    df['rv'] = df['distance'] * 1.5
    df['bus'] = df['distance'] * 2.2
    df['truck'] = df['distance'] * 3.6

    return df

# Example usage
# Assuming df is your DataFrame containing the data from Question 2
df = pd.read_csv('C:\\Users\\user\\Desktop\\Mapup\\MapUp-Data-Assessment-F-main\\datasets\\dataset-3.csv')
updated_df = calculate_toll_rate(df)
print(updated_df)

input_df = pd.read_csv('C:\\Users\\user\\Desktop\\Mapup\\MapUp-Data-Assessment-F-main\\datasets\\dataset-3.csv')
def calculate_time_based_toll_rates(input_df):
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
   # Define time ranges and discount factors
    weekday_discounts = [(time(0, 0, 0), time(10, 0, 0), 0.8),
                         (time(10, 0, 0), time(18, 0, 0), 1.2),
                         (time(18, 0, 0), time(23, 59, 59), 0.8)]

    weekend_discount = 0.7

    # Create a list to store the new rows
    new_rows = []

    # Iterate through each row in the input DataFrame
    for index, row in input_df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']

        # Iterate through each day of the week
        for day in range(7):
            start_datetime = datetime.combine(datetime.today(), time(0, 0, 0)) + timedelta(days=day)
            end_datetime = start_datetime + timedelta(days=1) - timedelta(seconds=1)

            # Apply discount factors based on time ranges
            for start_time, end_time, discount_factor in weekday_discounts if day < 5 else [(time(0, 0, 0), time(23, 59, 59), weekend_discount)]:
                start_datetime_range = datetime.combine(start_datetime, start_time)
                end_datetime_range = datetime.combine(start_datetime, end_time)

                # Check if the time range overlaps with the given row
                if not (end_datetime_range < datetime.combine(start_datetime, time(0, 0, 0)) or start_datetime_range > datetime.combine(end_datetime, time(23, 59, 59))):
                    # Calculate new toll rate based on the discount factor
                    new_distance = distance * discount_factor

                    # Append a new row with updated information
                    new_rows.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'distance': new_distance,
                        'start_day': start_datetime.strftime('%A'),
                        'start_time': start_time,
                        'end_day': end_datetime.strftime('%A'),
                        'end_time': end_time,
                        'moto' : df['distance'] * 0.8,
                        'car' : df['distance'] * 1.2,
                        'rv' : df['distance'] * 1.5,
                        'bus' : df['distance'] * 2.2,
                        'truck' : df['distance'] * 3.6
                    })

    # Create a new DataFrame with the added rows
    result_df = pd.DataFrame(new_rows)

    return result_df

# Example usage:
# Assuming you have a DataFrame named 'input_df' with the provided data
result_df = calculate_time_based_toll_rates(input_df)
print(result_df)


  
