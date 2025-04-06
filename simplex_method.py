#本程序用单纯形法求解线性规划问题，在求解之前要求问题已经转化为标准问题，使用两阶段法求解的问题需要先写为辅助问题求解一次后，人工定义新的问题再次求解
#单纯形法是迭代算法，在每一轮迭代中，本程序都将输出完整的系数矩阵、右侧向量、判别式、基变量指标集和非基变量指标集
#请特别注意，中间过程的指标集都是有序的，因此右侧向量和判别式各分量分别与基变量指标集和非基变量指标集一一对应，单纯形表遵循以下格式：
#假设基变量为[2,5]，非基变量指标集[1,4,6,3]，记判别式为d，系数矩阵为a，右侧向量为b
#   x1 | x2 | x3 | x4 | x5 | x6
#   d1 | 0  | d4 | d2 | 0  | d3
# 2 a11| a12| a13| a14| a15| a16 | b1
# 5 a21| a22| a23| a24| a25| a26 | b2
#但最终输出的A, b, x都是按照正常顺序（指标从小到大）排列，这是为了方便两阶段法的计算，确保A,b与原问题的c顺序对应
#完整算法可参考 最优化方法/杨庆之编著.-北京:科学出版社,2015, 算法2.5.5 修正单纯形法
import numpy as np

def find_min_ratio_index(B_inv_b, B_inv_a_k):
    """
    找到满足条件的最小比率对应的索引
    
    参数:
    B_inv_b -- 向量 (B^{-1}b)
    B_inv_a_k -- 向量 (B^{-1}a_k)
    
    返回:
    最小比率对应的索引
    """
    # 找到B_inv_a_k中为正的元素的索引
    positive_indices = np.where(B_inv_a_k > 0)[0]
    
    if len(positive_indices) == 0:
        raise ValueError("没有正值元素，无法计算比率")
    
    # 计算这些正元素对应的比率
    ratios = B_inv_b[positive_indices] / B_inv_a_k[positive_indices]
    
    # 找到最小比率对应的索引（在positive_indices中的位置）
    min_ratio_pos = np.argmin(ratios)
    
    # 返回原始数组中的索引
    return positive_indices[min_ratio_pos]

def gauss_jordan_elimination(A, b, i_r, k):
    """
    对增广矩阵 [A | b] 进行 Gauss-Jordan 消元：
    1. 以 (i_r, k) 为基准，消去其他行第 k 列的元素
    2. 归一化 i_r 行，使 A[i_r, k] = 1
    3. 返回更新后的 A 和 b

    参数:
        A: 系数矩阵 (n x m)
        b: 右侧向量 (n x 1)
        i_r: 基准行索引
        k: 基准列索引

    返回:
        A_new: 消元后的系数矩阵
        b_new: 消元后的右侧向量
    """
    A = A.astype(float)  # 转换为浮点型以避免整数除法问题
    b = b.astype(float)
    
    # 1. 归一化 i_r 行，使 A[i_r, k] = 1
    pivot = A[i_r, k]
    if pivot == 0:
        raise ValueError("主元为 0，无法归一化")
    
    A[i_r, :] /= pivot  # 整行除以主元
    b[i_r] /= pivot
    
    # 2. 消去其他行的第 k 列元素
    n_rows = A.shape[0]
    for i in range(n_rows):
        if i != i_r:
            factor = A[i, k]  # 当前行第 k 列的元素
            A[i, :] -= factor * A[i_r, :]  # 消去操作
            b[i] -= factor * b[i_r]
    
    return A, b

def simplex_method(A, b, c, IB):
    n = 1
    run = True
    IN = [x for x in range(A.shape[1]) if x not in IB]
    while run:
        print(f"当前第{n}轮,基变量指标集: {[x+1 for x in IB]}, 非基变量指标集: {[x+1 for x in IN]}")
        B = A[:, IB]
        N = A[:, IN]
        cN = c[IN]
        cB = c[IB]
        delta = cN - cB@np.linalg.inv(B)@N
        print("系数矩阵为:", A)
        print("右侧向量为:", b)
        print("判别式为:", delta)

        if np.all(delta >= 0):
            x = np.zeros(A.shape[1])
            x[IB] = b
            # 堆叠并排序
            IB = np.array(IB)
            stacked = np.hstack([IB.reshape(-1, 1), A, b.reshape(-1, 1)])
            sorted_stack = stacked[stacked[:, 0].argsort()]
            # 恢复
            IB = sorted_stack[:, 0].astype(int).tolist()
            A = sorted_stack[:, 1:-1]
            b = sorted_stack[:, -1]
            print("问题有解,基变量指标集为:", [x+1 for x in IB])
            print("最优解为:", x)
            run = False
            return A, b, x, IB
        else:
            k_position = np.argmin(delta)
            k = IN[k_position]
            factor = np.linalg.inv(B)@A[:,k]
            if np.all(factor <= 0):
                print("问题有无界解")
                run = False
            else:
                ir_position = find_min_ratio_index(np.linalg.inv(B)@b, factor)
                ir = IB[ir_position]
                print("进基变量指标:", k+1, "出基变量指标:", ir+1)
                IB[ir_position] = k
                IN[k_position] = ir
                A, b = gauss_jordan_elimination(A, b, ir_position, k)

        n += 1