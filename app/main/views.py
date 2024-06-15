#-*- coding=utf-8 -*-
import json
import os
import pickle
import uuid

import numpy as np
import pandas as pd
from flask import render_template, session, redirect, url_for, current_app, jsonify, request, \
    copy_current_request_context
from .. import db
from ..models import User, TaskStatus
from ..email import send_email
from . import main
from .forms import NameForm
from ..utils.ClickHouseUtils import ClickHouseUtils
from ..utils.TFIDFUtil import TFIDFUtil
from ..utils.githubUtil import get_user_info_by_name, get_user_info_by_id
from concurrent.futures import ThreadPoolExecutor
import redis

# 连接到Redis服务器
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

executor = ThreadPoolExecutor(8)  # 设置线程池

@main.route('/user', methods=['GET', 'POST'])
def user():
    current_app.logger.info(f'{request.method} main.user')
    task_id = str(uuid.uuid4())
    if request.method == 'GET':
        data = request.args
    else:
        data = request.json     # for request that POST with application/json

    task_status_json = json.dumps({"status": "in progress", "progress": 5})
    # 在任务字典中为此任务设置初始状态
    redis_client.set(task_id, task_status_json)
    # tasks[task_id] = {"status": "in progress", "progress": 5}

    @copy_current_request_context
    def async_task(task_data, task_ids):
        task = TaskStatus(id=task_ids, status="in progress", progress=0)
        db.session.add(task)
        db.session.commit()
        current_app.logger.info(f'async_task_id:{task_ids} task_data:{task_data}')
        try:
            user, prediction_list = perform_long_running_task(task_id, task_data)  # 模拟长时间运行的任务
            task.status = "completed"
            task.progress = 100
            task.user = json.dumps(user)
            task.result = prediction_list[0]
            # tasks[task_ids] = {"status": "completed", 'data': user, 'prediction_list':prediction_list}
            task_status_json = json.dumps({"status": "completed", 'data': user, 'prediction_list':prediction_list})
            redis_client.set(task_id, task_status_json)

            # task.result = {'user': user, 'prediction_list': prediction_list}
            current_app.logger.info(f'async_task_id:{task_ids} tasks[task_ids]:{task_data}')
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            print(e)
            # tasks[task_ids] = {"status": "failed", "result": "result"}
            task_status_json = json.dumps({"status": "failed"})
            redis_client.set(task_id, task_status_json)

        db.session.commit()

    # user, prediction_list = perform_long_running_task(task_id, data)
    # 在线程池中提交异步任务，传递数据和任务ID
    current_app.logger.info(f'submit {task_id}')
    executor.submit(async_task, data, task_id)
    # return user, prediction_list
    # return jsonify({"success": True, 'data': user, 'prediction_list':prediction_list})
    # 立即响应，返回任务ID
    return jsonify({"success": True, "message": "Task submitted", "task_id": task_id})


def update_task_progress(task_id, progress):
    # task_status = tasks.get(task_id)
    task_status = redis_client.get(task_id)
    current_app.logger.info(f'task_id:{task_id}, process:{progress}')
    if task_status:
        task_status_json = json.dumps({"status": "in progress", "progress": progress})
        redis_client.set(task_id, task_status_json)
        # task_status["progress"] = progress
        db.session.query(TaskStatus).filter_by(id=task_id).update({"progress": progress})
        db.session.commit()

@main.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    # 根据任务ID获取任务状态
    task_status = redis_client.get(task_id)
    if task_status is None:
        return jsonify({"error": "Task not found"}), 404
    task_status_result = json.loads(task_status)
    return jsonify(task_status_result)

@main.route('/', methods=['GET', 'POST'])
def home():
    form = NameForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.name.data).first()
        if user is None:
            user = User(username=form.name.data)
            db.session.add(user)
            db.session.commit()
            session['known'] = False
            if current_app.config['FLASKY_ADMIN']:
                send_email(current_app.config['FLASKY_ADMIN'], 'New User',
                           'mail/new_user', user=user)
        else:
            session['known'] = True
        session['name'] = form.name.data
        return redirect(url_for('.home'))
    return render_template('home.html',
                           form=form, name=session.get('name'),
                           known=session.get('known', False))


def perform_long_running_task(task_id, data):
    account_type = data.get("accountType", "username")  # 默认为"username"
    account = data.get("account")

    current_app.logger.info(f"perform_long_running_task:{task_id}")

    base_dir = os.path.dirname(__file__)  # 获取当前文件的目录
    model_path = os.path.join(base_dir, '..', '..', 'training', 'model', 'bothawk_model.pickle')

    # 加载预训练的模型
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    update_task_progress(task_id, 10)

    # 根据account_type决定调用的函数
    if account_type == "username":
        user_info = analyze_user_by_name(task_id, account)
    else:
        user_info = analyze_user_by_id(task_id, account)
    current_app.logger.info(f'user_info:{user_info}')

    update_task_progress(task_id, 100)

    print('user_info', user_info)
    # 限制user信息只包含指定的列，并更新列名
    columns_mapping = {
        "login": "login",
        "name": "name",
        "email": "email",
        "bio": "bio",
        "followers": "Number of followers",
        "following": "Number of following",
        "similarity": "tfidf_similarity",
        "event_count": "Number of Activity",
        "issue_count": "Number of Issue",
        "pull_request_count": "Number of Pull Request",
        "repo_count": "Number of Repository",
        "push_count": "Number of Commit",
        "active_days": "Number of Active day",
        "fft": "Periodicity of Activities",
        "connect_accounts": "Number of Connection Account",
        "response_time": "Median Response Time"
    }
    user = {new_key: user_info[old_key] for old_key, new_key in columns_mapping.items() if old_key in user_info}

    # 将user信息转换为DataFrame以便模型使用
    user_df = pd.DataFrame([user])
    print(user_df)
    # 预测
    # 注意: 根据您的模型和数据结构，您可能需要对user_df进行适当的预处理
    prediction = model.predict(user_df)
    # 修改这里，将numpy数组转换为列表
    prediction_list = prediction.tolist()
    return user, prediction_list
    # return jsonify({"success": True, 'data': user, 'prediction_list':prediction_list})


#
# @main.route('/', methods=['GET', 'POST'])
# def index():
#     form = NameForm()
#     if form.validate_on_submit():
#         user = User.query.filter_by(username=form.name.data).first()
#         if user is None:
#             user = User(username=form.name.data)
#             db.session.add(user)
#             db.session.commit()
#             session['known'] = False
#             if current_app.config['FLASKY_ADMIN']:
#                 send_email(current_app.config['FLASKY_ADMIN'], 'New User',
#                            'mail/new_user', user=user)
#         else:
#             session['known'] = True
#         session['name'] = form.name.data
#         return redirect(url_for('.index'))
#     return render_template('index.html',
#                            form=form, name=session.get('name'),
#                            known=session.get('known', False))
#


def analyze_user_by_id(task_id, id):
    user_info = get_user_info_by_id(id, current_app.config['GITHUB_API_TOKEN'])

    for user in user_info:
        print(user)
    update_task_progress(task_id, 10)
    user = clickhouse_quey(task_id, user)
    bot_str = ['bot ', '-ci', '-io', '-cla', '-bot', '-test', 'bot@']
    user['login'] = 1 if any(s in user['login'].lower() for s in bot_str) else 0
    user['name'] = 1 if any(s in user['name'].lower() for s in bot_str) else 0
    user['bio'] = 1 if any(s in user['bio'].lower() for s in bot_str) else 0
    user['email'] = 1 if any(s in user['email'].lower() for s in bot_str) else 0

    return user

def analyze_user_by_name(task_id, name):
    user_info = get_user_info_by_name(name, current_app.config['GITHUB_API_TOKEN'])
    for user in user_info:
        print(user)
    update_task_progress(task_id, 10)
    user = clickhouse_quey(task_id, user)
    bot_str = ['bot ', '-ci', '-io', '-cla', '-bot', '-test', 'bot@']
    user['login'] = 1 if any(s in user['login'].lower() for s in bot_str) else 0
    user['name'] = 1 if any(s in user['name'].lower() for s in bot_str) else 0
    user['bio'] = 1 if any(s in user['bio'].lower() for s in bot_str) else 0
    user['email'] = 1 if any(s in user['email'].lower() for s in bot_str) else 0

    return user

def clickhouse_quey(task_id, user):
    clickhouse = ClickHouseUtils(
        host=current_app.config['CLICKHOUSE_HOST'],
        user=current_app.config['CLICKHOUSE_USER'],
        password=current_app.config['CLICKHOUSE_PASSWORD'],
        database=current_app.config['CLICKHOUSE_DATABASE']
    )

    query = '''
    SELECT body
    FROM events
    where
        body != ''
    and
        actor_id = {id}
    limit 100
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    # for row in result:
    #     print(row)

    tfidf_util = TFIDFUtil()
    documents = result
    similarity = tfidf_util.calculate_similarity(documents)
    user['similarity'] = similarity

    update_task_progress(task_id, 20)

    query = '''
    SELECT count(*)
    FROM events
    where
        actor_id = {id}
    '''.format(id=user['id'])

    result = clickhouse.execute_query(query)
    user['event_count'] = result[0][0]

    update_task_progress(task_id, 30)

    query = '''
    SELECT count(*)
    FROM events
    where
        actor_id = {id}
    and type = 'IssuesEvent' and type='IssueCommentEvent'
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    user['issue_count'] = result[0][0]

    update_task_progress(task_id, 40)

    query = '''
    SELECT count(*)
    FROM events
    where
        actor_id = {id}
    and type = 'PullRequestEvent' and type='PullRequestReviewEvent' and type='PullRequestReviewCommentEvent'
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    user['pull_request_count'] = result[0][0]

    update_task_progress(task_id, 50)

    #Number of Repository
    query = '''
    SELECT count(distinct(repo_id))
    FROM events
    where
        actor_id = {id}
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    user['repo_count'] = result[0][0]

    update_task_progress(task_id, 60)


    query = '''
    SELECT count(distinct(repo_id))
    FROM events
    where
        actor_id = {id}
    and type= 'PushEvent'
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    user['push_count'] = result[0][0]

    update_task_progress(task_id, 70)


    query = '''
    SELECT
        count(*) as active_days
    FROM
        (
        SELECT
            toDate(created_at) as date
        FROM
            events
        where
            actor_id = {id}
            AND created_at >= toDate(now()) - 365
        GROUP BY
            date
		)
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    print(result)
    user['active_days'] = result[0][0]
    # user['event_count']/user['active_days']

    update_task_progress(task_id, 75)


    query = '''
    SELECT
        created_at
    FROM
       events
    where
        actor_id = {id}
        AND created_at >= toDate(now()) - 365
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    # 将这些日期转换为DataFrame
    df = pd.DataFrame(result, columns=['created_at'])
    df['created_at'] = pd.to_datetime(df['created_at'])
    # 为简化，我们假设每个日期对应一个事件，生成每日计数
    # 实际情况下，你应该使用从数据库获取的真实事件数据
    df['count'] = 1

    # 将'created_at'设置为索引并对日期进行分组计数
    df.set_index('created_at', inplace=True)
    daily_counts = df.resample('D').count()

    # 使用之前的FFT分析代码
    # 计算FFT
    # 检查数据是否为空
    if len(daily_counts['count'].values) == 0:
        print("No data available for FFT analysis.")
        # 在这里可以选择继续你的逻辑，比如返回错误消息或空的结果
        user['fft'] = 0
    else:
        # 其余的FFT分析代码
        fft_result = np.fft.fft(daily_counts['count'].values)

        # 计算频率
        freq = np.fft.fftfreq(len(fft_result))

        # 找到幅度最大的成分（除去直流分量即freq=0的情况）
        idx = np.argmax(np.abs(fft_result[1:])) + 1  # 加1是因为我们跳过了第一个成分

        # 最显著的周期性频率
        dominant_freq = freq[idx]

        # 计算周期（时间单位取决于原始数据的时间单位）
        period = 1 / dominant_freq if dominant_freq != 0 else 0  # 避免除以零的错误

        # 幅度作为周期性的强度指标
        amplitude = np.abs(fft_result[idx])

        print(f"主要周期: {period} 时间单位")
        print(f"周期性强度（幅度）: {amplitude}")
        # Assuming `user` is a dictionary where you want to store the ACF values
        user['fft'] = amplitude  # Store the ACF values as a list

    update_task_progress(task_id, 80)


    query = '''
    SELECT
    SUM(other_actor_count) AS total_other_actor_count
    FROM (
        SELECT
            issue_id,
            COUNT(DISTINCT actor_id) - 1 AS other_actor_count
        FROM
            events
        WHERE
            issue_id IN (
                SELECT DISTINCT issue_id
                FROM events
                WHERE
                    actor_id = 15813364
                    AND (type='IssuesEvent'
                        OR type='PullRequestEvent'
                        OR type='IssueCommentEvent'
                        OR type='PullRequestReviewCommentEvent')
            )
            AND actor_id != 15813364
        GROUP BY
            issue_id
    ) AS subquery
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    # print(result)
    user['connect_accounts'] = result[0][0]

    update_task_progress(task_id, 85)


    query = '''
    SELECT
    AVG(dateDiff('second', NULLIF(prev_created_at, '1970-01-01 00:00:00'), created_at)) AS avg_time_diff
    FROM (
        SELECT
            actor_id,
            issue_id,
            created_at,
            lagInFrame(created_at) OVER (PARTITION BY actor_id, issue_id ORDER BY created_at) AS prev_created_at
        FROM
            events
        WHERE
            actor_id = {id}
            AND (type='IssuesEvent'
                 OR type='PullRequestEvent'
                 OR type='IssueCommentEvent'
                 OR type='PullRequestReviewCommentEvent')
    ) AS subquery
    '''.format(id=user['id'])
    result = clickhouse.execute_query(query)
    print(result)
    user['response_time'] = result[0][0]

    update_task_progress(task_id, 90)

    return user

