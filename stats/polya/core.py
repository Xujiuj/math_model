# coding: utf-8

"""
    author:数模加油站
    波利亚模型通解

b : int
    抽样类型1的数量
r : int
    抽样类型2的数量
m : int
    抽到m个类型1
n : int
    抽到n个类型2
c : int
    每次抽完之后，再放入c个同类型的样本
d : int
    每次抽完之后，再放入d个不同类型的样本
"""

from math import comb

def sampling_without_replacement(b, r, m, n, /) -> float | None:
    """
    sampling_without_replacement(b, r, m, n, /) -> float | None \n

    无放回抽样(超几何分布) \n
    只涉及两种类型的抽样计算，不涉及多维超几何分布的计算  \n
    例如，检测次品率，在100个零件中进行抽取，其中好零件有90个，坏零件有10个，
    现在抽取了10次，抽取的结果分别是8个好零件和2个坏零件 \n
    那么计算的结果就是sampling_without_replacement(90, 10, 8, 2)
    Parameters
    ----------
    b : int
        抽样类型1的数量
    r : int
        抽样类型2的数量
    m : int
        抽到m个类型1, m < b
    n : int
        抽到n个类型2, n < r

    Returns
    -------
    out : float
        算得的概率值

    See Also
    --------
    scipy.stats.hypergeom.pmf(m, b+r, b, m+n)

    Examples
    --------
    >>> sampling_without_replacement(5, 6, 2, 3)
    0.4329004329004329
    >>> sampling_without_replacement(90, 10, 8, 2)
    0.20150988480897875

    如例子所示，100个零件中进行抽取，好零件有90个，抽取10次，抽到了8个好零件的概率
    """
    if m < b and n < r:
        return comb(b, m) * comb(r, n) / comb(b + r, m + n)
    raise ValueError("be sure m < b and n < r")


def sampling_with_replacement(b: int, r: int, m: int, n: int, /) -> float:
    r"""
    放回抽样，计算方法为$C_(m+n)^m\frac{b^m r^n}{b+r}^{m+n}$
    Parameters
    ----------
    b : int
        抽样类型1的数量
    r : int
        抽样类型2的数量
    m : int
        抽到m个类型1
    n : int
        抽到n个类型2

    See Also
    --------
    scipy.stats.binom.pmf(k, n, p)

    Returns
    -------
    out : float
        算得的概率值

    Examples
    --------
    >>> sampling_with_replacement(5, 6, 2, 3)
    0.33529751445194367
    >>> sampling_with_replacement(90, 10, 8, 2)
    0.19371024449999993
    """
    p = b / (b + r)
    return comb(m + n, m) * (p ** m) * (1 - p) ** n


def infection(b: int, r: int, m: int, n: int, c: int, /) -> float:
    """
    infection(b, r, m, n, c, /) -> float \n
    计算方法：C_(m+n)^m\prod{\prod^{m-1}{i=0}(r+ic)\prod^{n-1}{j=0}(b+ic)}{\prod^{m+n-1}{k=0}(b+r+kc)} \n
    波利亚模型中的传染病模型，每次发现一个传染病都会增加再感染的概率，每次发现一个正常则会减少再感染的概率

    Parameters
    ----------
    b : int
        抽样类型1的数量
    r : int
        抽样类型2的数量
    m : int
        抽到m个类型1
    n : int
        抽到n个类型2
    c : int
        每次抽完之后，会增加c个同类型的数量

    Returns
    -------
    result : float
        算得的传染病模型的概率值

    Examples
    --------
    >>> infection(5, 6, 1, 2, 3)
    0.3093964858670741
    """
    result = comb(m + n, m)
    for i in range(m):
        # 概率的计算中，分子是两个累乘的结果，分母是一个累乘的结果，共累乘了m+n次
        up = b + i * c
        down = b + r + i * c
        result *= up / down
    for j in range(n):
        up = r + j * c
        # 注意这里分母的计算要加上m，不能又从0开始计算
        down = b + r + (m + j) * c
        result *= up / down
    return result


def security(b: int, r: int, m: int, n: int, d: int, name1: str |int=0, name2 : str| int=1, verbose=False) -> dict | float:
    """
    security(b, r, m, n, d, /) -> float \n
    每次抽样都会放入d个不同类型的样本，即检测安全，那么就放松，下一次的事故率就增加，
    检测事故，那么就加紧，下一次的事故率就下降  \n
    但是注意，这个模型没有通解，抽取的顺序影响概率的计算，因此需要先得到要计算的全排列
    Parameters
    ----------
    b : int
        抽样类型1的数量
    r : int
        抽样类型2的数量
    m : int
        抽到m个类型1
    n : int
        抽到n个类型2
    d : int
        每次抽完之后，会增加d个不同类型的数量
    name1 : str | int
        类型1的名称
    name2 : str | int
        类型2的名称
    verbose : bool
        打印运算过程

    Returns
    -------
    out : dict | float
        如果verbose=True，计算不同组合下的安全模型的概率值
        如果verbose=False，直接输出所有组合下的概率和

    Examples
    --------
    >>> security(5, 6, 1, 2, 3, verbose=False)
    0.47097020626432384
    >>> security(5, 6, 1, 2, 3, verbose=True)
    {'1 0 1': 0.16501145912910617, '0 1 1': 0.15469824293353704, '1 1 0': 0.15126050420168066}


    每次抽完之后，放入3个不同类型的样本
    """
    import itertools
    name1, name2 = str(name1), str(name2)
    array = [name1] * m + [name2] * n
    result = {}
    # 重复的组合没必要计算，所以这里对他们排列组合之后，选用set类型而不是list类型
    permutations = set(itertools.permutations(array))
    # 遍历所有排列
    for group in permutations:
        _r = r
        _b = b
        out = 1
        # 遍历排列中的每一个元素
        for sample in group:
            # 如果是第一个类型的话
            if sample == name1:
                # 分母是样本总数
                # 分子是当前类型的数量
                out *= _b / (_b + _r)
                # 计算结束之后，另一类型的数量要加上d，这样计算的时候就不需要重新累加分子和分母了
                _r += d
            # 同上
            elif sample == name2:
                out *= _r / (_b + _r)
                _b += d
        result[" ".join(group)] = out
    if verbose:
        return result
    else:
        return sum(result.values())
