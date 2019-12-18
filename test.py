a = [1,2,3]


def tdfdf():
    for i in a:
        yield i

c = tdfdf()


next(c)
next(c)
next(c)
