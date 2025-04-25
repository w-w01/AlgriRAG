# 使用官方 Python 镜像并指定平台
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 复制项目文件(.dockerignore中定义的文件将自动忽略)
COPY . .

# 设置默认启动命令
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]