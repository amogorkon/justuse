import use

np = use("numpy")

for k, v in np.__dict__.items():
    if k.startswith("__"):
        print(k, str(v)[:20], type(v))
        