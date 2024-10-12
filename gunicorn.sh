# gunicorn -w 4 -b 0.0.0.0:80 manage:app
#!/bin/bash

# 启动 Gunicorn 并将输出日志重定向
gunicorn -b 0.0.0.0:80 manage:app --daemon --log-file gunicorn_output.log --log-level debug
