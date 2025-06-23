from justuse import URL, recklessness, use

mod4 = use(
    URL("https://raw.githubusercontent.com/amogorkon/q/main/q.py"),
    modes=recklessness,
    import_as="q",
).Q()
print(mod4)

print("end")
