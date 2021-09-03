def foo():
    try:
        return 34
    finally:
        print(345)
        
print(foo())