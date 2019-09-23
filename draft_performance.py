import timeit

number = 10 ** 6

# time_main = timeit.timeit("from draft_main import performance_dominance_new; performance_dominance_new()",
#                           number=number)

# time_vector = timeit.timeit("from draft_main import test_vector; test_vector()", number=number)
#
# print("Test vector: {}".format(time_vector))

# for i in range(10):
#     time_list_map = timeit.timeit("from draft_main import array_list_map; array_list_map()", number=number)
#     time_list_comprehension = timeit.timeit(
#         "from draft_main import array_list_comprehension; array_list_comprehension()", number=number)
#     time_list_vectorize = timeit.timeit("from draft_main import array_vectorize; array_vectorize()", number=number)
#
#     print("Time list map: {:.5f}".format(time_list_map))
#     print("Time list comprehension: {:.5f}".format(time_list_comprehension))
#     print("Time list vectorize: {:.5f}".format(time_list_vectorize))
#     print()

for _ in range(5):
    time_math_is_close = timeit.timeit("from draft_main import array_math_is_close; array_math_is_close()",
                                       number=number)
    time_math_own_function = timeit.timeit("from draft_main import array_math_own_function; array_math_own_function()",
                                           number=number)
    time_decimals = timeit.timeit("from draft_main import array_decimals; array_decimals()", number=number)

    print('Math is close: {}s'.format(time_math_is_close))
    print('Math own function: {}s'.format(time_math_own_function))
    print('Math decimals: {}s'.format(time_decimals))

pass
