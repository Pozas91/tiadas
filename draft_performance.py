import timeit

number = 1000

time_main = timeit.timeit("from draft_main import performance_dominance_new; performance_dominance_new()",
                          number=number)

# time_a1 = timeit.timeit("from draft import a1; a1()", number=number)
# time_a2 = timeit.timeit("from draft import a2; a2()", number=number)
# time_a3 = timeit.timeit("from draft import a3; a3()", number=number)
# time_a4 = timeit.timeit("from draft import a4; a4()", number=number)
# time_a5 = timeit.timeit("from draft import a5; a5()", number=number)
# time_a6 = timeit.timeit("from draft import a6; a6()", number=number)

# print("A1 time: {}".format(time_a1))
# print("A2 time: {}".format(time_a2))
# print("A3 time: {}".format(time_a3))
# print("A4 time: {}".format(time_a4))
# print("A5 time: {}".format(time_a5))
# print("A6 time: {}".format(time_a6))

print("Main time: {}".format(time_main))
