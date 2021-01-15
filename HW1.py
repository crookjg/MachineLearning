# 1a. Write a basic Python function (no imports) that creates a the following list of values,
# given n as an input argument. As n gets very large, what does the sum of this series approach?
#       As n gets very large, the SUM of this series approaches 2.
def series(n):
    #lst = [1]
    #for i in range(1, n+1):
    #    lst.append(1/(2 ** i))
    lst = [i for i in range(0, n+1) if 1/(2** i)]
    return lst
# 1b. Write a basic Python function (no imports) that computes the dot-product between two lists. 
# You can assume that u and v have the same length.
def dot_product(u, v):
    prod = 0
    """
    for i in range(len(u)):
            a = u[i]
            b = v[i]
            prod += a * b
    """
    for ui, vi in zip(u,v):
        prod += ui * vi
    return prod
# 2. Dot products: For reference, the definition of the dot product: v · w = ‖v‖‖w‖cosθ
#   (a) What is the length of the projection of v on w?
#           ||v|| cosθ
#   (b) For a given vector v, what is the maximum projection of v on w (by choosing w)?
#           ||v||
#   (c) For a given vector v, what is the minimum projection of v on w (by choosing w)?
#           0
#   (d) For a given vector v, what vector w maximizes the projection of v on w?
#           v = w
#   (e) For a given vector v, what vector w minimizes the projection of v on w?
#           w = ?
#   (f) For a given vector v, what angle between v and w maximizes the projection of v on w?
#           θ = 0
#   (g) For a given vector v, what angle between v and w minimizes the projection of v on w?
#           θ = 90
#   (h) For a given vector v, what angle between v and w produces a projection of 0?
#           θ = 180

# Additional Candidate Questions
# 1a. Write a basic Python function (no imports) that multiplies A (m x p) and B (p x n) and returns C (m x n):
def matrix_multiply(a, b):
    return [[sum(ai * bi for ai, bi, in zip(a_row, b_col)) for b_col in zip(*b)] for a_row in a]

# 1b. Write a basic Python function (no imports) that computes the transpose of A (m × n): A [Transpose] (i, j) = A (j, i)
def transpose(a):
    return [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]

# Write a basic Python function (no imports) that computes the outer product of two vectors:
def outer_product(a, b):
    return [[i * j for j in b] for i in a]
    #prod = []
    #for i in a:
    #    row = []
    #    for j in b:
    #        row.append(i * j)
    #    prod.append(row)
    #return prod

#x = ['a','b','c']
#y = [1,2,3]
#print(outer_product(x, y))

# Write a basic python function (no imports) that takes a vector 'v' as an input argument and returns its magnitude (length):
def magnitude(v):
    mag = 0
    for i in range(len(v)):
        mag += v[i] ** 2
    return mag ** .5

x = [1, 3, 5]
print(magnitude(x))