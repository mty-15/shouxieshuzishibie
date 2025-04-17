from waitress import serve

from app import app

if __name__ == '__main__':
    print("启动生产服务器 Waitress...")
    print("访问 http://localhost:5002 使用应用")
    serve(app, host='0.0.0.0', port=5002, threads=4)
