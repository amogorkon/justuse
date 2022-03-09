from itertools import permutations

print(len([x for x in permutations("123456789") if int(''.join(x)) % 2 == 0]))