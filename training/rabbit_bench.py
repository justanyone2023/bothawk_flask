import os
from rabbit.rabbit import get_results

split_dir = 'data/rabbit/splits'
split_files = sorted([f for f in os.listdir(split_dir) if f.startswith('filtered_clean_data_actors_part_')])

apikey = os.getenv('GITHUB_API_KEY')
min_events = 10
min_confidence = 0.75
max_queries = 100
output_type = 'csv'
verbose = True
incremental = True

# 遍历前10个文件，调用get_results函数
for i, file_name in enumerate(split_files):
    contributors_name_file = os.path.join(split_dir, file_name)
    save_path = f'data/rabbit/results/result_part_{i+1}.csv'  # 结果保存路径

    print(f'Processing {file_name}...')

    # 调用get_results函数处理每个文件
    get_results(
        contributors_name_file=contributors_name_file,
        contributor_name=[],
        apikey=apikey,
        min_events=min_events,
        min_confidence=min_confidence,
        max_queries=max_queries,
        output_type=output_type,
        save_path=save_path,
        verbose=verbose,
        incremental=incremental
    )

print('Finished processing all files.')
