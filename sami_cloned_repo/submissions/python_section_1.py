from typing import Dict, List

import pandas as pd

from itertools import permutations

import re

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    result = []

    for i in range(0,len(lst), n):
        segment = []

        for j in range(i,min(i+n, len(lst))):
            segment.insert(0,lst[j])
        result.extend(segment)
    return result
    
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))           
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict ={}

    for string in lst:
        length = len(string)

        if length not in length_dict:
            length_dict[length] =[]

        length_dict[length].append(string)
    return dict(sorted(length_dict.items()))

print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))  
print(group_by_length(["one", "two", "three", "four"]))


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    
    def flatten(current: Dict, parent_key: str = '') -> Dict:
        items = []
        for k, v in current.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(item, dict):
                        items.extend(flatten(item, list_key).items())
                    else:
                        items.append((list_key, item))
            else:
                items.append((new_key, v))
        return dict(items)
    
    return flatten(nested_dict)


nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened = flatten_dict(nested_dict)
print(flattened)



def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    unique_perm = set(permutations(nums))

    return [list(perm) for perm in unique_perm]
nums = [1,1,2]
result = unique_permutations(nums)
print("[")
for perm in result:
    print(f"    {perm},")

    


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',   
        r'\b\d{2}/\d{2}/\d{4}\b',   
        r'\b\d{4}\.\d{2}\.\d{2}\b'  
    ]

    combined_pattern = '|'.join(date_patterns)
    
    matches = re.findall(combined_pattern, text)
    
    return matches
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))



def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    return pd.Dataframe()


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    
    n = len(matrix)
    
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    final_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[row][j] for row in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
result = rotate_and_multiply_matrix(matrix)
print(result)


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df.set_index(['id', 'id_2'], inplace=True)
    
    results = pd.Series(dtype=bool, index=df.index.unique())
    
    def create_full_time_range():
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hours = [f'{h:02}:00:00' for h in range(24)]
        return [(day, hour) for day in days for hour in hours]

    full_time_set = set(create_full_time_range())
    
    for group_index, group in df.groupby(level=['id', 'id_2']):
        group_time_set = set(zip(group['startDay'], group['startTime']))
        group_time_set.update(zip(group['endDay'], group['endTime']))
        
        is_complete = full_time_set.issubset(group_time_set)
        
        results[group_index] = not is_complete
    
    return results

df = pd.read_csv('dataset-1.csv')
result = time_check(df)
print(result)
