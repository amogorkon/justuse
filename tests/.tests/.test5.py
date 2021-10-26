import pydantic

s = "a b\tc\nd\re\x0bf\x0cg"
print(s)
print("".join(s.split()))