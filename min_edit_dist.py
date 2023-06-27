import numpy as np

def min_edit_distance(source, target, ins_cost=1, del_cost=1):
    print(source, '-->' , target)

    # Determine lengths of two strings
    n = len(source)
    m = len(target)

    # Create distance matrix
    dist_matrix = np.zeros((n + 1, m + 1))

    # Initialize 0th row and column; meat of matrix is all 0s
    # (distance from this position to empty string)
    for i in range(1, n + 1):
        dist_matrix[i][0] = dist_matrix[i-1][0] + del_cost
    for j in range(1, m + 1):
        dist_matrix[0][j] = dist_matrix[0][j-1] + ins_cost

    # Recurrence relation
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if source[i-1] == target[j-1] else ins_cost + del_cost
            dist_matrix[i, j] = min(
                dist_matrix[i-1, j] + del_cost,
                dist_matrix[i-1, j-1] + sub_cost,
                dist_matrix[i, j-1] + ins_cost)

    print(dist_matrix)
    print(dist_matrix[n, m])
    return dist_matrix[n, m]    # return minimum distance


def main():
    print('\n')
    source = str(input(f"Source: "))
    target = str(input(f"Target: "))
    min_edit_distance(source, target)

if __name__ == "__main__":
    main()