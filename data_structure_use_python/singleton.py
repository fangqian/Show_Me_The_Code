def singleton(_cls):
    insts = {}

    def getinstance(*args, **kwargs):
        if _cls not in insts:
            insts[_cls] = _cls(*args, **kwargs)
        return insts[_cls]
    return getinstance


# @singleton
class A(object):
    a = 1

    def __init__(self, x=5, y=1):
        self.x = y


a1 = A(2)
a2 = A(3)

print(id(a1))
print(id(a2))
print(a1.x)
print(a2.x)