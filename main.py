# lab01 测试代码 - 可运行、可调试
def test():
    a = 10
    b = 20
    c = a + b
    print("计算结果：", c)
    return c

# 主程序
if __name__ == "__main__":
    print("=== lab01 运行成功 ===")
    result = test()
    print("最终返回值：", result)