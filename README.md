# bot_hawk

**start**

```
git clone https://github.com/abbeyokgo/flask_template.git #下载源码
cd flask_template #切换到目录下
pip install -r requirements.txt
#init
export FLASK_APP=manage.py #linux
set FLASK_APP=manage.py #windows cmd
$env:FLASK_APP = "manage"  #windows powershell
flask db init #Initialize database
flask db migrate #Migration
flask db upgrade #upgrade
flask run #run
```

After running the website, prompt
```
[root@centos flask_template]# python manage.py runserver
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 128-717-467
```

然后浏览器打开：`127.0.0.1:5000`即可
