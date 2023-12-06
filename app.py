from flask import Flask, render_template, request, redirect, url_for, send_file
import csv
# from templates.image.mmdetection.video_analysis import analysisMain
import random
import os
# from fastapi import FastAPI

app = Flask(__name__)
# app = FastAPI()

# ホームページ
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/mypage')
def mypage():
    return render_template('home.html')

# PWAの実装
# https://medium.com/@tristan_4694/how-to-create-a-progressive-web-app-pwa-using-flask-f227d5854c49
# https://github.com/umluizlima/flask-pwa/tree/master
@app.route('/manifest.json')
def serve_manifest():
    return send_file('manifest.json', mimetype='application/manifest+json')
@app.route('/sw.js')
def serve_sw():
    return send_file('sw.js', mimetype='application/javascript')


## 色覚検査

# 最初のセットの問題数
num_tests_first_set = 3
# 二番目のセットの問題数
num_tests_second_set = 3

# 最初のセットの問題と正解
first_set_color_tests = [
    {'question': '赤', 'options': ['赤', '青', '黄'], 'answer': '赤', 'display_answer': '赤'},
    {'question': '青', 'options': ['赤', '青', '黄'], 'answer': '青', 'display_answer': '青'},
    {'question': '黄', 'options': ['赤', '青', '黄'], 'answer': '黄', 'display_answer': '黄'},
]

# 二番目のセットの問題と正解
second_set_color_tests = [
    {'question': '紫', 'options': ['赤', '青', '黄'], 'answer': '青', 'display_answer': '紫'},
    {'question': 'オレンジ', 'options': ['赤', '青', '黄'], 'answer': '黄', 'display_answer': 'オレンジ'},
    {'question': 'ピンク', 'options': ['赤', '青', '黄'], 'answer': '赤', 'display_answer': 'ピンク'},
]

# セットごとに問題をシャッフルする関数
def shuffle_questions():
    # ランダムな順番で問題をシャッフル
    random.shuffle(first_set_color_tests)
    random.shuffle(second_set_color_tests)

# シャッフル関数を初回実行
shuffle_questions()

# 現在の問題インデックス
current_question_index = 0

# 正解数を格納するリスト
correct_answers_list = [0] * (num_tests_first_set + num_tests_second_set)

# ユーザーの回答を格納するリスト
user_answers_list = []

# 問題を開始するページ
@app.route('/color/index')
def index():
    global current_question_index, correct_answers_list
    current_question_index = 0
    correct_answers_list = [0] * (num_tests_first_set + num_tests_second_set)
    return render_template('color/index.html', question=first_set_color_tests[current_question_index])

# チェックボタンを押した際の処理
@app.route('/color/check_answer', methods=['POST'])
def check_answer():
    global current_question_index, correct_answers_list
    user_answer = request.form['user_answer']

    # インデックスが範囲内かどうかを確認
    if current_question_index < num_tests_first_set + num_tests_second_set:
        if current_question_index < num_tests_first_set:
            current_question = first_set_color_tests[current_question_index]
        else:
            current_question = second_set_color_tests[current_question_index - num_tests_first_set]

        # ユーザーが選択した回答を比較
        # 表示は赤、青、黄だが、内部的には紫、青、オレンジとして処理
        if user_answer == current_question['answer']:
            correct_answers_list[current_question_index] = 1

        # ユーザーの回答をリストに追加
        user_answers_list.append(user_answer)   

        current_question_index += 1

    if current_question_index < num_tests_first_set + num_tests_second_set:
        if current_question_index < num_tests_first_set:
            current_question = first_set_color_tests[current_question_index]
        else:
            current_question = second_set_color_tests[current_question_index - num_tests_first_set]

        return render_template('color/index.html', question=current_question)
    else:
        return redirect(url_for('result'))

# 結果画面
@app.route('/color/result')
def result():
    global correct_answers_list
    correct_answers = sum(correct_answers_list)
    
    # 各問題に対する正解/不正解と正解の色を取得
    results = []
    for i in range(num_tests_first_set + num_tests_second_set):
        current_question = first_set_color_tests[i] if i < num_tests_first_set else second_set_color_tests[i - num_tests_first_set]
        result = {
            'question': current_question['question'],
            'correct': correct_answers_list[i] == 1,
            'correct_answer': current_question['answer'],
            'user_answer': user_answers_list[i],
        }
        results.append(result)

    # 問題をシャッフル
    shuffle_questions()
    
    return render_template('color/result.html', correct_answers=correct_answers, num_tests=(num_tests_first_set + num_tests_second_set), results = results)

## 動画分析
@app.route('/image')
def imageIndex():
    message = 0
    return render_template('image/index.html', upload_message = message)

@app.route('/upload', methods=['POST'])
def upload():
    message = 0
    if 'video' in request.files:
        video = request.files['video']
        if video.filename != '':
            # video.save(os.path.join('uploads', video.filename))
            video.save(os.path.join('uploads', 'video.mp4'))
            return render_template('image/index.html', upload_message = 1)
    return render_template('image/index.html', upload_message = 2)

@app.route('/image/analysis')
def imageAnalysis():
    score, analysis_result = analysisMain('./uploads/video.mp4')
    return render_template('image/index.html', upload_message = 3, result = analysis_result, score = score)


## 知識問題
question_count = 5
# Load questions from CSV file
def load_questions():
    questions = []
    with open('static/csv/driving_questions.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        # print(csv_reader)
        for row in csv_reader:
            # print(row)
            options = row['選択肢'].split('/')
            question = {
                'id': row['\ufeffid'],
                'question': row['問題文'],
                'options': options,
                'correct_option': row['正解'],
                'explanation': row['解説']
            }
            questions.append(question)
    return questions

all_questions = load_questions()

@app.route('/questions', methods=['GET', 'POST'])
def questionsIndex():
    if request.method == 'POST':
        question_count = int(request.form.get('question_count', 5))
        return redirect(url_for('driving_exam'))
    return render_template('questions/index.html')


@app.route('/questions/test', methods=['GET', 'POST'])
def driving_exam():
    if request.method == 'POST':
        return check_exam(request.form, all_questions, question_count)
    else:
        selected_questions = random.sample(all_questions, question_count)
        return render_template('questions/driving_exam.html', questions=selected_questions, question_count=question_count)

def check_exam(user_answers, questions, question_count):
    score = 0
    result_details = [] 
    for i, question in enumerate(questions, 1):
        # G et correct option as string
        correct_option_str = question['correct_option']
        
        # Get user's answer as string
        user_answer_str = user_answers.get(f'question{i}')
        
        if user_answer_str == correct_option_str:
            score += 1

        if user_answer_str is not None:
            question_details = {
                'question': question['question'],
                'correct_answer': correct_option_str,
                'user_answer': user_answer_str,
                'explanation': question.get('explanation', 'No explanation available.')
            }
            result_details.append(question_details)

    result_message = f'正解数: {score}/{question_count}'
    print(result_message)
    return render_template('questions/result.html', result_message=result_message, result_details=result_details)



## 反射神経

@app.route('/reflexes')
def reflexesIndex():
    return render_template('reflexes/index.html')


if __name__ == '__main__':
    app.run(debug=True)
