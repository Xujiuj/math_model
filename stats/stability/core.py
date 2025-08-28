# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/7/14 11:13
    @file  : core.py
"""

from .classes import *


def series(num: int, p: float | list[float]) -> float:
    """
        用于一键计算串联系统的算放框架
    Parameters
    ----------
    num: int
        该系统中的元素数量
    p: float | list[float]
        故障率，若为一个[0, 1]之间的常数，表示各个零部件故障率相同

        若为一个列表，则列表长度应该与num的值相同，表示每个零件对应的故障率

    Returns
    -------
    out: float
        该系统的可靠性

    Examples
    --------
    >>> series(3, 0.1) # 三个零件，每个零件的故障率都是0.1
    0.729

    >>> series(3, [0.1, 0.2, 0.3])  # 三个零件，故障率分别为0.1、0.2、0，3
    0.504
    """
    # 为了后续复杂混联系统的计算与开发，这里对各系统做了类的封装
    ser = Series(p, num)
    return round(ser.compute(), 5)

def parallel(num: int, p: float | list[float]) -> float:
    """
        用于一键计算并联系统的可靠性
    Parameters
    ----------
    num: int
        该系统中的元素数量
    p: float | list[float]
        故障率，若为一个[0, 1]之间的常数，表示各个零部件故障率相同

        若为一个列表，则列表长度应该与num的值相同，表示每个零件对应的故障率

    Returns
    -------
    out: float
        该系统的可靠性

    Examples
    --------
    >>> parallel(3, 0.9) # 必须全部坏，该系统才会坏，所以这里把故障率设为0.9，那么全坏的概率就是0.9^3=0.729，对应可靠性0.271
    0.271

    >>> parallel(3, [0.9, 0.8, 0.7])  # 可以看到，概率与串联系统取反的情况下，所得到的值也是1-串联系统的概率
    0.496
    """
    para = Parallel(p, num)
    return round(para.compute(), 5)

def vote(num: int, p: float | list[float], r: int| float) -> float:
    """
        用于一键计算表决系统的可靠性
    Parameters
    ----------
    num: int
        该系统中的元素数量
    p: float | list[float]
        故障率，若为一个[0, 1]之间的常数，表示各个零部件故障率相同

        若为一个列表，则列表长度应该与num的值相同，表示每个零件对应的故障率
    r: int| float
        若为整数类型，则表示该系统中最少需要多少台机器正常才正常，此时r<=num

        若为小数类型，则表示该系统中最少需要多少占比的机器正常才正常，算得结果向上取整，此时 r \in [0, 1]

    Returns
    -------
    out: float
        该系统的可靠性

    Examples
    --------
    >>> vote(3, 0.1, 2)
    0.972

    >>> vote(10, 0.1, 9)
    0.7361
    """
    vo = Vote(p, num, r)
    return round(vo.compute(), 5)
