from unittest.mock import patch
import webbrowser


def test():
    with patch("webbrowser.open"):
        return foobar() == 43
    

def foobar():
    webbrowser.open("google.com")
    return 43

print(test())