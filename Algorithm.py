# -*- coding: utf-8 -*-

from itertools import combinations
import tkinter as tk
def main(m:list,n,j,s):
    k=6
    data=m[:n]
    k_tps=list(combinations(data,k))  
    j_tps = list(combinations(data, j))  
    lst=[k_tps[0]]  #第一个k

    templst=[]
    ks = k_tps[0]
    for js in j_tps:   #j_tps 一个j里所有的s
        s_tps=list(combinations(js,s))
        flag=True   
        for ss in s_tps:
            if cover(ks,ss):
                flag=False  #是否循环进程走完
                break
        if flag:
            templst.append(s_tps) #所有没有被匹配的j的全部ss组合 存储的为s的组合 每一个组合对应一个j
    k_tps.pop(0) #弹第一个k
    while len(templst)!=0:  #贪心算法
        numdit = {}         #创建一个字典 (k,ss)
        for ks in k_tps:    #找最优的k
            dr=[]           
            for ss in templst:
                for s in ss:
                    if cover(ks, s):
                        dr.append(ss[:])
                        break
            if len(dr)!=0:
                numdit[ks]=dr  #ks
        numdit=sorted(numdit.items(),key=lambda x:len(x[1]),reverse=True)  # 比较x1的大小
        lst.append(numdit[0][0])
        for i in numdit[0][1]:
            templst.remove(i)
        k_tps.remove(numdit[0][0])  

    print(len(lst))
    return lst

def cover(a,b):
    return set(b).issubset(set(a))

def calculate_result():
    # 获取输入框中的值
    m = m_entry.get().split(',')
    n = int(n_entry.get())
    j = int(j_entry.get())
    s = int(s_entry.get())

    # 计算结果
    result = main(m,n,j,s)
    rst=f'{len(result)}\n'
    for i in result:
        rst+=str(i)+'\n'

    # 显示结果
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, rst)




# 创建主窗口
root = tk.Tk()
root.title("Optimal Cover Set Finder")
root.geometry("570x350+500+350")
# 创建输入框和标签

m_label = tk.Label(root, text="m:" , height=2, width=10, font=25 , anchor= "e" )
m_label.grid(row=1, column=0, padx=1 , pady=15 )
m_entry = tk.Entry(root, font=25 , width=20)
m_entry.grid(row=1, column=1)

n_label = tk.Label(root, text="n:" , height=2, width=10, font=25 , anchor= "e")
n_label.grid(row=2, column=0, padx=1 , )
n_entry = tk.Entry(root , font=25 , width=20)
n_entry.grid(row=2, column=1)

j_label = tk.Label(root, text="j:", height=2, width=10, font=25 , anchor= "e")
j_label.grid(row=1, column=2, padx=1 , pady=15)
j_entry = tk.Entry(root , font=25 , width=20)
j_entry.grid(row=1, column=3)

s_label = tk.Label(root, text="s:" , height=2, width=10, font=25 , anchor= "e")
s_label.grid(row=2, column=2, padx=1 , )
s_entry = tk.Entry(root ,font=52, width=20 )
s_entry.grid(row=2, column=3)

# 创建按钮
calculate_button = tk.Button(root, text="Calculate", command=calculate_result , font = 25 , anchor= "w" )
calculate_button.grid(row=6, column=0, columnspan=4, pady= 10)

# 创建输出框
output_text = tk.Text(root, height=10, width=40 , font = 25 , )
output_text.grid(row=7, column=1, columnspan=4, pady=10)

# 进入消息循环
root.mainloop()


if __name__ == '__main__':
    pass
    # m=['A','B','C','D','E','F','G','H','I','J','k','l','m','n']
    # n=12
    # j=4
    # s=4
    # lst=main(m,n,j,s)
    # for l in lst:
    #     print(l)
    # j_tps = list(combinations(m[:7], j))
    # for i in j_tps:
    #     print(i)
    # pass
