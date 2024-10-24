import pandas as pd
import numpy as np



def calculate_distance_matrix(df) -> pd.DataFrame(): # type: ignore
    """Calculate a distance matrix based on the dataframe, df.
    
    Args:
        df (pandas.DataFrame)
    
    Returns:
        pandas.DataFrame: Distance matrix
    """
    ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    
    dist_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)
    
    np.fill_diagonal(dist_matrix.values, 0)
    
    for _, row in df.iterrows():
        dist_matrix.at[row['id_start'], row['id_end']] = row['distance']
        dist_matrix.at[row['id_end'], row['id_start']] = row['distance']  
    
    for k in ids:
        for i in ids:
            for j in ids:
                if dist_matrix.at[i, j] > dist_matrix.at[i, k] + dist_matrix.at[k, j]:
                    dist_matrix.at[i, j] = dist_matrix.at[i, k] + dist_matrix.at[k, j]
    
    return dist_matrix

df = pd.read_csv('dataset-2.csv')

distance_matrix = calculate_distance_matrix(df)


print("Distance Matrix:")
print(distance_matrix)



def unroll_distance_matrix(df) -> pd.DataFrame(): # type: ignore
    """Unroll a distance matrix to a DataFrame in the style of the initial dataset.
    
    Args:
        df (pandas.DataFrame)
    
    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    df_unrolled = df.reset_index().melt(id_vars=['index'], var_name='id_end', value_name='distance')
    df_unrolled.columns = ['id_start', 'id_end', 'distance']
    
    df_unrolled = df_unrolled[df_unrolled['id_start'] != df_unrolled['id_end']]
    
    return df_unrolled

unrolled_df = unroll_distance_matrix(distance_matrix)

print("\nUnrolled Distance Matrix:")
print(unrolled_df)




def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame(): # type: ignore
    """Find all IDs whose average distance lies within 10% of the average distance of the reference ID.
    
    Args:
        df (pandas.DataFrame)
        reference_id (int)
    
    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    ref_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    
    lower_bound = ref_avg_distance * 0.9
    upper_bound = ref_avg_distance * 1.1
    
    avg_distances = df.groupby('id_start')['distance'].mean()
    
    within_threshold_ids = avg_distances[(avg_distances >= lower_bound) & (avg_distances <= upper_bound)].index.tolist()
    
    within_threshold_ids.sort()
    
    return within_threshold_ids

df = pd.read_csv('dataset-2.csv')

reference_id = 1001400  
threshold_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)

print("\nIDs within 10% threshold of reference ID's average distance:")
print(threshold_ids)



def calculate_toll_rate(df)->pd.DataFrame(): # type: ignore
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame(): # type: ignore
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    return df
