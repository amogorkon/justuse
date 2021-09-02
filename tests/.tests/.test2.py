from enum import Enum

class Message(Enum):
    problem_at_the_foo= lambda: f"{foo} is bar!"