s = 'abcdefgabcd'
set(s)

print(''.join(x for x in s if not x in set(s)))