import math


def get_landmark_x_y_row(num_landmark, landmark_list):
    row = {}
    j = 0;
    for i in range(num_landmark * 2):
        index = int(i / 2)
        row[i] = landmark_list[index][4]
        i += 1
        row[i] = landmark_list[index][5]
    return row


def calculate_distance(x1, y1, x2, y2):
	return math.dist([x1, y1], [x2, y2])


def get_landmark_distance_row(num_landmark, landmark_list):
    row = {}
    row_index = 0;
    for i in range(1, num_landmark):
        row[row_index] = calculate_distance(landmark_list[0][4], landmark_list[0][5], landmark_list[i][4], landmark_list[i][5])
        row_index += 1  
    i = 2
    while i < num_landmark - 4:
    	row[row_index] = calculate_distance(landmark_list[i][4], landmark_list[i][5], landmark_list[i + 4][4], landmark_list[i + 4][5])
    	row_index += 1 
    	i += 4
    i = 3
    while i < num_landmark - 4:
    	row[row_index] = calculate_distance(landmark_list[i][4], landmark_list[i][5], landmark_list[i + 4][4], landmark_list[i + 4][5])
    	row_index += 1 
    	i += 4
    i = 4
    while i < num_landmark - 4:
    	row[row_index] = calculate_distance(landmark_list[i][4], landmark_list[i][5], landmark_list[i + 4][4], landmark_list[i + 4][5])
    	row_index += 1 
    	i += 4
    return row