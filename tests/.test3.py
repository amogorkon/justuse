import use

def decorator(func):
   print("decorated", func)
   def wrapper(*args, **kwargs):
      return func(*args, **kwargs)
   return wrapper


for x in range(5):
    print(x)
    for x in range(9,12):
        print(x)


#use(use.Path("../../../watnu/master/src/classes.py", aspectize={(use.aspect.property, "Task.*"): decorator}))