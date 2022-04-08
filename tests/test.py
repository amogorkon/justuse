s = 'abcdefgabcd'
set(s)

print(''.join(x for x in s if x not in set(s)))