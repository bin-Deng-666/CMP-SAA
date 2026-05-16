# 步骤 1：申请交互式 GPU 节点

## 方式一：`srun --pty`（进 interactive 分区，约 2 小时）

**40G 内存 + 指定 A800：**

```bash
srun --cpus-per-task=4 --mem=40G --gres=gpu:a800:1 --pty bash
```

**40G 内存 + 任意 1 张 GPU：**

```bash
srun --cpus-per-task=4 --mem=40G --gres=gpu:1 --pty bash
```

## 方式二：`salloc`（进 cluster-1，适合长时间挂 Streamlit）

先申请资源，等分配成功后会进入计算节点 shell（无 `--pty` 时不会被改到 interactive）：

```bash
salloc --partition=cluster-1 --cpus-per-task=4 --mem=40G --gres=gpu:a800:1

srun --jobid=59317 --pty bash
```

进入节点后执行步骤 2。用完后退出 shell 即释放资源；也可 `scancel <jobid>`。

# 步骤 2：启动 Streamlit

在 **GraduationProject** 目录下：

```bash
conda activate grad-project
streamlit run frontend/home.py
```

