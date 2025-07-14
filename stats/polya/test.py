from core import *


"""
又称为罐子模型，可以引出放回抽样、无放回抽样、传染病模型以及安全模型。
设罐中有b个蓝球、r个红球，每次随机取出一个球，取出后将原球放回，还加进c个同色球和d个异色球
无放回抽样：c=-1，d=0
放回抽样：  c=0，d=0
传染病模型： c>0，d=0
安全模型：  c=0，d>0
"""
b = 90
r = 10
m = 8
n = 2
c = 3
d = 3

print(sampling_without_replacement(b, r, m, n, ))
print(sampling_with_replacement(b, r, m, n, ))
print(infection(b, r, m, n, c))
print(security(b, r, m, n, d))

