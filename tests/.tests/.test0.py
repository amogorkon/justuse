import use
#use(use.Path(".test1.py"))

#mod1 = use("numpy")
#print(mod1)
#print(mod1.__version__)

#mod2 = use("numpy", version="1.19.2", modes=use.auto_install)
#print(mod2)
#print(mod2.__version__)

#mod3 = use("math")
#print(mod3)

mod4 = use(
    use.URL("https://raw.githubusercontent.com/amogorkon/q/main/q.py"),
    modes=use.recklessness,
    import_as="q",
).Q()
print(mod4)

print("end")
