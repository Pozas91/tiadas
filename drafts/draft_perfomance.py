import timeit

import numpy as np

# Set number of iterations
number = 15000

time_dominance = []

for _ in range(30):
    time_dominance.append(
        timeit.timeit('from draft_main import performance_dominance; performance_dominance()', number=number)
    )

print('Dominance:')
print('\tAVG: {}s'.format(np.average(time_dominance)))
print('\tSTD: {}s'.format(np.std(time_dominance)))
print('\tMIN: {}s'.format(np.min(time_dominance)))
print('\tMAX: {}s'.format(np.max(time_dominance)))
print('')

# time_sum_integer = timeit.timeit("from draft_main import sum_integer; sum_integer()", number=number)
# time_sum_decimal = timeit.timeit("from draft_main import sum_decimal; sum_decimal()", number=number)
#
# for _ in range(15):
#     print('Sum integer: {}state'.format(time_sum_integer))
#     print('Sum decimal: {}state'.format(time_sum_decimal))

# time_copy = list()
# time_deep = list()
# time_hypervolume = list()
#
# for _ in range(10):
#     time_copy.append(timeit.timeit("from draft_main import simple_v_s_0; simple_v_s_0()", number=number))
#     time_deep.append(timeit.timeit("from draft_main import deep_v_s_0; deep_v_s_0()", number=number))
#     time_hypervolume.append(timeit.timeit("from draft_main import hypervolume; hypervolume()", number=number))
#
# print('Copy:')
# print('\tAVG: {}state'.format(np.average(time_copy)))
# print('\tSTD: {}state'.format(np.std(time_copy)))
# print('\tMIN: {}state'.format(np.min(time_copy)))
# print('\tMAX: {}state'.format(np.max(time_copy)))
# print('')
#
# print('Deep:')
# print('\tAVG: {}state'.format(np.average(time_deep)))
# print('\tSTD: {}state'.format(np.std(time_deep)))
# print('\tMIN: {}state'.format(np.min(time_deep)))
# print('\tMAX: {}state'.format(np.max(time_deep)))
# print('')
#
# print('Hypervolume:')
# print('\tAVG: {}state'.format(np.average(time_hypervolume)))
# print('\tSTD: {}state'.format(np.std(time_hypervolume)))
# print('\tMIN: {}state'.format(np.min(time_hypervolume)))
# print('\tMAX: {}state'.format(np.max(time_hypervolume)))

# time_normal_train = timeit.timeit('from drafts.algorithm_w import draft_b; draft_b()', number=50)
#
# print('Normal Train Time: {:.3f}'.format(time_normal_train))
