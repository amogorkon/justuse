import json
registry= {"foo": 23, "bar": 93}

with open(".test2.py", "w") as file:
    #print(file.readlines())
    text = "### WARNING ###\n" + json.dumps(registry)
    print(text)
    file.write(text)
    
with open(".test2.py", "r") as file:
    registry2 = json.loads('\n'.join(filter(lambda s: not s.startswith("#"), file.readlines())))
    

print(registry2, registry == registry2)