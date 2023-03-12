from copy import deepcopy


class Hello:
    time = None

hello = Hello()

hello.time= 1


hi = deepcopy(hello)

hi.time = 2

print(hello.time)
